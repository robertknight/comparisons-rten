mod audio;
mod tokenizers;
mod utils;

use audio::{get_mel_filteres, read_audio};
use ndarray::{
    concatenate, s, Array, Array2, Array3, ArrayView, ArrayView3, Axis, Dim, Dimension, Ix,
    StrideShape,
};
use ndarray_npy::NpzReader;
use rten::Model;
use rten_generate::{
    Generator, GeneratorConfig, GeneratorError, GeneratorUtils, ModelInputsConfig,
};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use std::fmt;
use std::fs::File;
use std::ops::Range;
use tokenizers::Tokenizer;
use utils::Options;

/// Convert an ndarray view into an RTen NdTensorView.
fn as_ndtensor_view<'a, T, const N: usize>(
    view: ArrayView<'a, T, Dim<[Ix; N]>>,
) -> Option<NdTensorView<'a, T, N>>
where
    Dim<[Ix; N]>: Dimension,
{
    view.to_slice().map(|slice| {
        let shape: [usize; N] = view.shape().try_into().unwrap();
        NdTensorView::from_data(shape, slice)
    })
}

/// Convert an owned RTen NdTensor into an ndarray array.
fn into_array<T, const N: usize>(tensor: NdTensor<T, N>) -> Array<T, Dim<[Ix; N]>>
where
    T: Clone,
    Dim<[Ix; N]>: Dimension,
    [usize; N]: Into<StrideShape<Dim<[Ix; N]>>>,
{
    let shape = tensor.shape();
    let data = tensor.into_data();
    Array::from_shape_vec(shape, data).unwrap()
}

pub struct Whisper {
    encoder: Model,
    decoder: Model,
    tokenizer: Tokenizer,
    pos_emb: Array3<f32>,
    mel_filters: Array2<f32>,
    options: Options,
}

impl fmt::Debug for Whisper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Whisper").finish()
    }
}

impl Whisper {
    fn get_audio_features(&self, segments: Vec<Array2<f32>>) -> Array3<f32> {
        let mels: Array3<f32> = segments
            .into_iter()
            .fold(None, |acc, array| {
                Some(match acc {
                    Some(concatenated) => {
                        concatenate![Axis(0), concatenated, array.insert_axis(Axis(0))]
                    }
                    None => array.insert_axis(Axis(0)),
                })
            })
            .unwrap();
        let inputs = as_ndtensor_view(mels.view()).unwrap();
        let encoder_out = self.encoder.run_one(inputs.into(), None).unwrap();
        let result: NdTensor<f32, 3> = encoder_out.try_into().unwrap();
        into_array(result)
    }

    fn get_initial_tokens(&self, prompt: Vec<i32>, language: &str) -> Vec<i32> {
        let lang_token = *self.tokenizer.lang2token.get(language).unwrap();
        let init_tokens: Vec<i32> = vec![50258, lang_token as i32, 50359];

        if prompt.len() > 0 {
            let prev_prompt_len = self.options.n_ctx / 2 - 1;
            let prompt_tokens: Vec<i32>;

            if prompt.len() > prev_prompt_len {
                prompt_tokens = prompt[prompt.len() - prev_prompt_len..].to_vec();
            } else {
                prompt_tokens = prompt;
            }

            let tokens: Vec<i32> = vec![self.options.sot_prev as i32]
                .into_iter()
                .chain(prompt_tokens.into_iter())
                .collect();
            let tokens: Vec<i32> = tokens.into_iter().chain(init_tokens.into_iter()).collect();
            tokens
        } else {
            let tokens = vec![self.options.sot_prev as i32];
            let tokens: Vec<i32> = tokens.into_iter().chain(init_tokens.into_iter()).collect();
            tokens
        }
    }

    fn inference(
        &self,
        audio_features: ArrayView3<f32>,
        prompt: Vec<i32>,
        language: &str,
    ) -> Result<Vec<i32>, GeneratorError> {
        let initial_tokens = self.get_initial_tokens(prompt, language);

        let audio_features_id = self.decoder.node_id("audio_features").unwrap();
        let audio_features_tensor = as_ndtensor_view(audio_features.view()).unwrap();
        let prompt: Vec<_> = initial_tokens.into_iter().map(|id| id as u32).collect();

        let model_inputs = ModelInputsConfig {
            input_ids: "tokens",
            logits: "logits",
            key_cache: ("k", "").into(),
            key_cache_output: ("output_k", "").into(),
            value_cache: ("v", "").into(),
            value_cache_output: ("output_v", "").into(),
            ..Default::default()
        };
        let gen_config = GeneratorConfig { model_inputs };
        let pos_emb_id = self.decoder.node_id("pos_emb").unwrap();
        let pos_embed_slice = |_batch_size: usize, seq_pos: Range<usize>| {
            let pos_emb = self.pos_emb.slice(s![.., seq_pos, ..]);
            as_ndtensor_view(pos_emb).unwrap().into()
        };

        let max_tokens = 224.min(self.options.n_ctx);
        let generator = Generator::from_model_config(&self.decoder, gen_config)?
            .with_prompt(&prompt)
            .with_constant_input(audio_features_id, audio_features_tensor.into())
            .with_varying_input(pos_emb_id, &pos_embed_slice)
            .stop_on_token(self.options.eot_token as u32)
            .take(max_tokens);

        let mut tokens = Vec::with_capacity(max_tokens);
        for token_id in generator {
            let token_id = token_id?;
            tokens.push(token_id as i32);
        }
        Ok(tokens)
    }

    fn encode(&self, mel: Array2<f32>) -> Array3<f32> {
        let num_frames = mel.shape()[1];
        let mut seek = 0;
        let mut segments = vec![];

        while seek < num_frames {
            let segment: Array2<f32>;

            if seek + audio::N_FRAMES < mel.shape()[1] {
                segment = mel.slice(s![.., seek..seek + audio::N_FRAMES]).to_owned();
            } else {
                segment = mel.slice(s![.., seek..]).to_owned();
            }

            segments.push(audio::pad_or_trim(segment, audio::N_FRAMES));
            seek += audio::N_FRAMES;
        }

        self.get_audio_features(segments)
    }

    fn decode_tokens(&self, result: Vec<i32>) -> String {
        self.tokenizer.decode(
            result
                .iter()
                .map(|v| *v as usize)
                .filter(|item| item < &50257)
                .collect(),
        )
    }

    fn get_mel(&self, audio_data: Vec<f32>) -> Array2<f32> {
        audio::log_mel_spectrogram(audio_data, self.mel_filters.clone())
    }

    fn run(&self, mel: Array2<f32>, language: &str) -> String {
        let audio_features = self.encode(mel);

        let mut result: Vec<i32> = vec![];
        for audio_feature in audio_features.axis_iter(Axis(0)) {
            let audio_feature = audio_feature.insert_axis(Axis(0));
            let tokens = self
                .inference(audio_feature, result.clone(), language)
                .expect("generation failed");
            result.extend(tokens.clone());
        }

        self.decode_tokens(result)
    }
}

impl Whisper {
    pub fn new(
        encoder_path: &str,
        decoder_path: &str,
        tokenizer_path: &str,
        pos_emb_path: &str,
        mel_filters_path: &str,
    ) -> Whisper {
        let encoder = Model::load_file(encoder_path).unwrap();
        let decoder = Model::load_file(decoder_path).unwrap();
        let tokenizer = Tokenizer::new(tokenizer_path);
        let pos_emb = {
            let file = File::open(pos_emb_path).expect("Failed to open file");
            let mut npz = NpzReader::new(file).expect("Failed to read NPZ file");
            let pos_emb: Array2<f32> = npz.by_index(0).unwrap();
            pos_emb.insert_axis(Axis(0))
        };
        let mel_filters = get_mel_filteres(mel_filters_path);
        let options = Options::new();

        Whisper {
            encoder,
            decoder,
            tokenizer,
            pos_emb,
            mel_filters,
            options,
        }
    }

    pub fn recognize_from_audio(&self, audio_path: &str, language: &str) -> String {
        let audio_data = read_audio(audio_path).unwrap();
        let mel = self.get_mel(audio_data);
        self.run(mel, language)
    }
}
