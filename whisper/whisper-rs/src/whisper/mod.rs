mod audio;
mod tokenizers;
mod utils;

use audio::{get_mel_filteres, read_audio};
use ndarray::{
    concatenate, s, Array, Array2, Array3, ArrayView, ArrayView2, ArrayView3, Axis, Dim, Dimension,
    Ix, StrideShape,
};
use ndarray_npy::NpzReader;
use rten::{Input, Model, NodeId, Output};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use std::fmt;
use std::fs;
use std::fs::File;
use tokenizers::Tokenizer;
use utils::{KVCache, Options};

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

trait Recognition {
    fn get_encoder(&self) -> &Model;

    fn get_decoder(&self) -> &Model;

    fn get_tokenizer(&self) -> &Tokenizer;

    fn get_pos_emb(&self) -> &Array3<f32>;

    fn get_mel_filters(&self) -> &Array2<f32>;

    fn get_options(&self) -> &Options;

    fn get_default_kvcache(&self) -> KVCache;

    fn inference_logits(
        &self,
        tokens: ArrayView2<i32>,
        decoder_inputs: &[(NodeId, Output)],
        kv_cache: &KVCache,
        initial_token_length: usize,
    ) -> (Array3<f32>, KVCache);

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
        let encoder_out = self.get_encoder().run_one(inputs.into(), None).unwrap();
        let result: NdTensor<f32, 3> = encoder_out.try_into().unwrap();
        into_array(result)
    }

    fn get_initial_tokens(&self, prompt: Vec<i32>, language: &str) -> Vec<i32> {
        let lang_token = *self.get_tokenizer().lang2token.get(language).unwrap();
        let init_tokens: Vec<i32> = vec![50258, lang_token as i32, 50359];

        if prompt.len() > 0 {
            let prev_prompt_len = self.get_options().n_ctx / 2 - 1;
            let prompt_tokens: Vec<i32>;

            if prompt.len() > prev_prompt_len {
                prompt_tokens = prompt[prompt.len() - prev_prompt_len..].to_vec();
            } else {
                prompt_tokens = prompt;
            }

            let tokens: Vec<i32> = vec![self.get_options().sot_prev as i32]
                .into_iter()
                .chain(prompt_tokens.into_iter())
                .collect();
            let tokens: Vec<i32> = tokens.into_iter().chain(init_tokens.into_iter()).collect();
            tokens
        } else {
            let tokens = vec![self.get_options().sot_prev as i32];
            let tokens: Vec<i32> = tokens.into_iter().chain(init_tokens.into_iter()).collect();
            tokens
        }
    }

    fn inference(
        &self,
        audio_features: ArrayView3<f32>,
        prompt: Vec<i32>,
        language: &str,
    ) -> Vec<i32> {
        let initial_tokens = self.get_initial_tokens(prompt, language);
        let initial_token_length = initial_tokens.len();

        let mut tokens: Array2<i32> = Array::from_vec(initial_tokens).insert_axis(Axis(0));
        let mut kv_cache = self.get_default_kvcache();

        // Precompute parts of the decoder graph that only depend on encoder
        // outputs. This makes each decoder step much faster.
        let decoder = self.get_decoder();
        let logits_id = decoder.node_id("logits").unwrap();
        let audio_features_id = decoder.node_id("audio_features").unwrap();
        let audio_features_tensor = as_ndtensor_view(audio_features.view()).unwrap();
        let decoder_inputs = decoder
            .partial_run(
                &[(audio_features_id, audio_features_tensor.into())],
                &[logits_id],
                None,
            )
            .expect("decoder run failed");

        for _ in 0..224 {
            let logits: Array3<f32>;
            (logits, kv_cache) = self.inference_logits(
                tokens.view(),
                &decoder_inputs,
                &kv_cache,
                initial_token_length,
            );
            let next_word = logits
                .slice(s![.., -1, ..])
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as usize)
                .unwrap();

            if next_word == self.get_options().eot_token
                || tokens.shape()[1] > self.get_options().n_ctx
            {
                break;
            }

            let next_word_array = Array::from_elem((1, 1), next_word as i32);
            tokens = concatenate!(Axis(1), tokens, next_word_array);
        }
        tokens = tokens.slice(s![.., initial_token_length..]).to_owned();
        tokens.into_raw_vec()
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
        self.get_tokenizer().decode(
            result
                .iter()
                .map(|v| *v as usize)
                .filter(|item| item < &50257)
                .collect(),
        )
    }

    fn get_mel(&self, audio_data: Vec<f32>) -> Array2<f32> {
        audio::log_mel_spectrogram(audio_data, self.get_mel_filters().clone())
    }

    fn run(&self, mel: Array2<f32>, language: &str) -> String {
        let audio_features = self.encode(mel);

        let mut result: Vec<i32> = vec![];
        for audio_feature in audio_features.axis_iter(Axis(0)) {
            let audio_feature = audio_feature.insert_axis(Axis(0));
            let tokens = self.inference(audio_feature, result.clone(), language);
            result.extend(tokens.clone());
        }

        self.decode_tokens(result)
    }
}

impl Recognition for Whisper {
    fn get_encoder(&self) -> &Model {
        &self.encoder
    }

    fn get_decoder(&self) -> &Model {
        &self.decoder
    }

    fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn get_pos_emb(&self) -> &Array3<f32> {
        &self.pos_emb
    }

    fn get_mel_filters(&self) -> &Array2<f32> {
        &self.mel_filters
    }

    fn get_options(&self) -> &Options {
        &self.options
    }

    fn get_default_kvcache(&self) -> KVCache {
        KVCache::default(512)
    }

    fn inference_logits(
        &self,
        tokens: ArrayView2<i32>,
        decoder_inputs: &[(NodeId, Output)],
        kv_cache: &KVCache,
        initial_token_length: usize,
    ) -> (Array3<f32>, KVCache) {
        let offset = kv_cache.k1.shape()[1];

        let tokens = if tokens.shape()[1] > initial_token_length {
            tokens.slice_move(s![.., -1]).insert_axis(Axis(0))
        } else {
            tokens
        };

        let pos_emb = self
            .pos_emb
            .slice(s![.., offset..offset + tokens.shape()[1], ..]);

        let tokens_id = self.decoder.node_id("tokens").unwrap();
        let pos_emb_id = self.decoder.node_id("pos_emb").unwrap();
        let k1_id = self.decoder.node_id("k1").unwrap();
        let v1_id = self.decoder.node_id("v1").unwrap();
        let k2_id = self.decoder.node_id("k2").unwrap();
        let v2_id = self.decoder.node_id("v2").unwrap();
        let k3_id = self.decoder.node_id("k3").unwrap();
        let v3_id = self.decoder.node_id("v3").unwrap();
        let k4_id = self.decoder.node_id("k4").unwrap();
        let v4_id = self.decoder.node_id("v4").unwrap();
        let k5_id = self.decoder.node_id("k5").unwrap();
        let v5_id = self.decoder.node_id("v5").unwrap();
        let k6_id = self.decoder.node_id("k6").unwrap();
        let v6_id = self.decoder.node_id("v6").unwrap();

        let logits = self.decoder.node_id("logits").unwrap();
        let output_k1_id = self.decoder.node_id("output_k1").unwrap();
        let output_v1_id = self.decoder.node_id("output_v1").unwrap();
        let output_k2_id = self.decoder.node_id("output_k2").unwrap();
        let output_v2_id = self.decoder.node_id("output_v2").unwrap();
        let output_k3_id = self.decoder.node_id("output_k3").unwrap();
        let output_v3_id = self.decoder.node_id("output_v3").unwrap();
        let output_k4_id = self.decoder.node_id("output_k4").unwrap();
        let output_v4_id = self.decoder.node_id("output_v4").unwrap();
        let output_k5_id = self.decoder.node_id("output_k5").unwrap();
        let output_v5_id = self.decoder.node_id("output_v5").unwrap();
        let output_k6_id = self.decoder.node_id("output_k6").unwrap();
        let output_v6_id = self.decoder.node_id("output_v6").unwrap();

        let tokens = as_ndtensor_view(tokens.view()).unwrap();
        let pos_emb = as_ndtensor_view(pos_emb).unwrap();
        let k1 = as_ndtensor_view(kv_cache.k1.view()).unwrap();
        let v1 = as_ndtensor_view(kv_cache.v1.view()).unwrap();
        let k2 = as_ndtensor_view(kv_cache.k2.view()).unwrap();
        let v2 = as_ndtensor_view(kv_cache.v2.view()).unwrap();
        let k3 = as_ndtensor_view(kv_cache.k3.view()).unwrap();
        let v3 = as_ndtensor_view(kv_cache.v3.view()).unwrap();
        let k4 = as_ndtensor_view(kv_cache.k4.view()).unwrap();
        let v4 = as_ndtensor_view(kv_cache.v4.view()).unwrap();
        let k5 = as_ndtensor_view(kv_cache.k5.view()).unwrap();
        let v5 = as_ndtensor_view(kv_cache.v5.view()).unwrap();
        let k6 = as_ndtensor_view(kv_cache.k6.view()).unwrap();
        let v6 = as_ndtensor_view(kv_cache.v6.view()).unwrap();

        // Add the inputs which change on each decoder iteration.
        let mut inputs: Vec<(NodeId, Input)> = vec![
            (tokens_id, tokens.into()),
            (pos_emb_id, pos_emb.into()),
            (k1_id, k1.into()),
            (v1_id, v1.into()),
            (k2_id, k2.into()),
            (v2_id, v2.into()),
            (k3_id, k3.into()),
            (v3_id, v3.into()),
            (k4_id, k4.into()),
            (v4_id, v4.into()),
            (k5_id, k5.into()),
            (v5_id, v5.into()),
            (k6_id, k6.into()),
            (v6_id, v6.into()),
        ];

        // Add the inputs which are constant while decoding a chunk of audio.
        inputs.extend(
            decoder_inputs
                .iter()
                .map(|(node_id, output)| (*node_id, output.into())),
        );

        let [logits, k1, v1, k2, v2, k3, v3, k4, v4, k5, v5, k6, v6] = self
            .decoder
            .run_n(
                &inputs,
                [
                    logits,
                    output_k1_id,
                    output_v1_id,
                    output_k2_id,
                    output_v2_id,
                    output_k3_id,
                    output_v3_id,
                    output_k4_id,
                    output_v4_id,
                    output_k5_id,
                    output_v5_id,
                    output_k6_id,
                    output_v6_id,
                ],
                None,
            )
            .unwrap();

        let logits: NdTensor<f32, 3> = logits.try_into().unwrap();
        let logits = into_array(logits);

        let k1: NdTensor<f32, 3> = k1.try_into().unwrap();
        let k1 = into_array(k1);
        let v1: NdTensor<f32, 3> = v1.try_into().unwrap();
        let v1 = into_array(v1);

        let k2: NdTensor<f32, 3> = k2.try_into().unwrap();
        let k2 = into_array(k2);
        let v2: NdTensor<f32, 3> = v2.try_into().unwrap();
        let v2 = into_array(v2);

        let k3: NdTensor<f32, 3> = k3.try_into().unwrap();
        let k3 = into_array(k3);
        let v3: NdTensor<f32, 3> = v3.try_into().unwrap();
        let v3 = into_array(v3);

        let k4: NdTensor<f32, 3> = k4.try_into().unwrap();
        let k4 = into_array(k4);
        let v4: NdTensor<f32, 3> = v4.try_into().unwrap();
        let v4 = into_array(v4);

        let k5: NdTensor<f32, 3> = k5.try_into().unwrap();
        let k5 = into_array(k5);
        let v5: NdTensor<f32, 3> = v5.try_into().unwrap();
        let v5 = into_array(v5);

        let k6: NdTensor<f32, 3> = k6.try_into().unwrap();
        let k6 = into_array(k6);
        let v6: NdTensor<f32, 3> = v6.try_into().unwrap();
        let v6 = into_array(v6);

        let new_kv_cache = KVCache {
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
        };

        (logits, new_kv_cache)
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
        let encoder = Model::load(&fs::read(encoder_path).unwrap()).unwrap();
        let decoder = Model::load(&fs::read(decoder_path).unwrap()).unwrap();
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
