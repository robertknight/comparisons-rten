import argparse
import time
import typing as t
from collections import namedtuple

import numpy as np
import onnxruntime
import tqdm

from src.audio import (
    HOP_LENGTH,
    N_FRAMES,
    SAMPLE_RATE,  # CHUNK_LENGTH,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
)
from src.decode import (
    BeamSearchDecoder,
    GreedyDecoder,
    MaximumLikelihoodRanker,
    SuppressTokens,
)
from src.tokenizer import get_tokenizer
from src.utils import softmax

WEIGHT_ENC_PATH = "../weights/encoder.onnx"
WEIGHT_DEC_PATH = "../weights/decoder.onnx"

WEIGHT_ENC_QUANT_PATH = "../weights/encoder_quant.onnx"
WEIGHT_DEC_QUANT_PATH = "../weights/decoder_quant.onnx"

POSITIONAL_EMB = np.load("../weights/positional_embedding.npz")["pos_emb"]

AUDIO_PATH = "../data/audio.wav"

ModelDimensions = namedtuple(
    "ModelDimensions",
    [
        "n_mels",
        "n_audio_ctx",
        "n_audio_state",
        "n_audio_head",
        "n_audio_layer",
        "n_vocab",
        "n_text_ctx",
        "n_text_state",
        "n_text_head",
        "n_text_layer",
    ],
)

DecodingResult = namedtuple(
    "DecodingResult",
    [
        "audio_features",
        "language",
        "language_probs",
        "tokens",
        "text",
        "avg_logprob",
        "no_speech_prob",
        "temperature",
    ],
)

dims = ModelDimensions(80, 1500, 512, 8, 6, 51865, 448, 512, 8, 6)

V = True
language = "en"
temperature_increment_on_fallback = 0.2
logprob_threshold = -1.0
no_speech_threshold = 0.6

task = "transcribe"
best_of = None
beam_size = None
patience = 1.0
length_penalty = None
suppress_tokens = "-1"

intermediate = True
dynamic_kv_cache = False

tokenizer = get_tokenizer(True, language=language, task=task)


def get_initial_tokens(tokenizer, options) -> tuple:
    sot_sequence = tokenizer.sot_sequence
    sample_len = options.get("sample_len") or dims.n_text_ctx // 2
    n_ctx = dims.n_text_ctx

    tokens = list(sot_sequence)
    prefix = options.get("prefix", None)
    prompt = options.get("prompt", [])

    if prefix:
        prefix_tokens = (
            tokenizer.encode(" " + prefix.strip())
            if isinstance(prefix, str)
            else prefix
        )
        if sample_len is not None:
            max_prefix_len = n_ctx // 2 - sample_len
            prefix_tokens = prefix_tokens[-max_prefix_len:]
        tokens = tokens + prefix_tokens

    if prompt:
        # prompt_arg_tokens = tokenizer.encode(args.prompt)
        prompt_arg_tokens = []
        prompt_tokens = prompt
        prev_prompt_len = (n_ctx // 2 - 1) - len(prompt_arg_tokens)
        tokens = (
            [tokenizer.sot_prev]
            + prompt_arg_tokens
            + prompt_tokens[-prev_prompt_len:]
            + tokens
        )
    else:
        prompt_tokens = (
            tokenizer.encode(" " + prompt.strip())
            if isinstance(prompt, str)
            else prompt
        )
        tokens = (
            [tokenizer.sot_prev] + prompt_tokens[-(n_ctx // 2 - 1) :] + tokens
        )

    return tuple(tokens)


def get_suppress_tokens(tokenizer, options) -> tuple:
    suppress_tokens = options["suppress_tokens"]

    if isinstance(suppress_tokens, str):
        suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []  # interpret empty string as an empty list
    else:
        assert isinstance(
            suppress_tokens, list
        ), "suppress_tokens must be a list"

    suppress_tokens.extend(
        [tokenizer.sot, tokenizer.sot_prev, tokenizer.sot_lm]
    )
    if tokenizer.no_speech is not None:
        # no-speech probability is collected separately
        suppress_tokens.append(tokenizer.no_speech)

    return tuple(sorted(set(suppress_tokens)))


def get_audio_features(enc_net, mel):
    mel = mel.astype(np.float32)
    output = enc_net.run(None, {"mel": mel})
    audio_features = output[0]

    return audio_features


def new_kv_cache(n_group: int, length: int = 451) -> np.ndarray:
    # model_type = args.model_type
    # if model_type == "tiny.en" or model_type == "tiny":
    #     size = [8, n_group, length, 384]
    # elif model_type == "base.en" or model_type == "base":
    #     size = [12, n_group, length, 512]
    # elif model_type == "small.en" or model_type == "small":
    #     size = [24, n_group, length, 768]
    # elif model_type == "medium.en" or model_type == "medium":
    #     size = [48, n_group, length, 1024]
    # elif model_type == "large":
    #     size = [64, n_group, length, 1280]
    # else:
    #     raise ValueError(f"Unsupported model type: {model_type}")
    size = [12, n_group, length, 512]
    return np.zeros(size, dtype=np.float32, order="C")


def inference_logits(
    dec_net,
    tokens,
    audio_features,
    kv_cache=None,
    initial_token_length=None,
    constant_audio_feature=False,
) -> tuple:
    n_group = tokens.shape[0]
    initial_token_length = (
        initial_token_length if initial_token_length else tokens.shape[-1]
    )
    length = 0
    if kv_cache is None:
        if not dynamic_kv_cache:
            kv_cache = new_kv_cache(n_group)
        else:
            kv_cache = new_kv_cache(n_group, initial_token_length)
        offset = 0
        length = initial_token_length
    else:
        offset = kv_cache.shape[2]
        if not dynamic_kv_cache:
            length = offset + 1
            _kv_cache = new_kv_cache(n_group)
            _kv_cache[:, :, :offset, :] = kv_cache
        else:
            _kv_cache = new_kv_cache(n_group, offset + 1)
            _kv_cache[:, :, :-1, :] = kv_cache
        kv_cache = _kv_cache

    if tokens.shape[-1] > initial_token_length:
        # only need to use the last token except in the first forward pass
        tokens = tokens[:, -1:]

    tokens = tokens.astype(np.int32)
    offset = np.array(offset, dtype=np.int32)
    kv_cache = kv_cache.astype(np.float32)
    pos_emb = POSITIONAL_EMB[offset.item(): offset.item() + tokens.shape[-1]]
    pos_emb = np.expand_dims(pos_emb, axis=0)
    output = dec_net.run(
        None,
        {
            "tokens": tokens,
            "audio_features": audio_features,
            "pos_emb": pos_emb,
            "k1": kv_cache[0][:,:offset.item(),:],
            "v1": kv_cache[1][:,:offset.item(),:],
            "k2": kv_cache[2][:,:offset.item(),:],
            "v2": kv_cache[3][:,:offset.item(),:],
            "k3": kv_cache[4][:,:offset.item(),:],
            "v3": kv_cache[5][:,:offset.item(),:],
            "k4": kv_cache[6][:,:offset.item(),:],
            "v4": kv_cache[7][:,:offset.item(),:],
            "k5": kv_cache[8][:,:offset.item(),:],
            "v5": kv_cache[9][:,:offset.item(),:],
            "k6": kv_cache[10][:,:offset.item(),:],
            "v6": kv_cache[11][:,:offset.item(),:],
        },
    )
    logits, k1, v1, k2, v2, k3, v3, k4, v4, k5, v5, k6, v6 = output
    kv_cache[0, :,: offset.item() + tokens.shape[-1], :] = k1
    kv_cache[1, :,: offset.item() + tokens.shape[-1], :] = v1
    kv_cache[2, :,: offset.item() + tokens.shape[-1], :] = k2
    kv_cache[3, :,: offset.item() + tokens.shape[-1], :] = v2
    kv_cache[4, :,: offset.item() + tokens.shape[-1], :] = k3
    kv_cache[5, :,: offset.item() + tokens.shape[-1], :] = v3
    kv_cache[6, :,: offset.item() + tokens.shape[-1], :] = k4
    kv_cache[7, :,: offset.item() + tokens.shape[-1], :] = v4
    kv_cache[8, :,: offset.item() + tokens.shape[-1], :] = k5
    kv_cache[9, :,: offset.item() + tokens.shape[-1], :] = v5
    kv_cache[10, :,: offset.item() + tokens.shape[-1], :] = k6
    kv_cache[11, :,: offset.item() + tokens.shape[-1], :] = v6

    if not dynamic_kv_cache:
        return logits, kv_cache[:, :, :length, :]
    else:
        return logits, kv_cache


def decode(enc_net, dec_net, mel, options) -> t.List[DecodingResult]:
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    n_group = options.get("beam_size") or options.get("best_of") or 1
    n_ctx = dims.n_text_ctx
    sample_len = options.get("sample_len") or dims.n_text_ctx // 2

    initial_tokens = get_initial_tokens(tokenizer, options)
    sample_begin = len(initial_tokens)
    sot_index = initial_tokens.index(tokenizer.sot)

    logit_filters = []
    logit_filters.append(
        SuppressTokens(get_suppress_tokens(tokenizer, options))
    )
    sequence_ranker = MaximumLikelihoodRanker(options.get("length_penalty"))

    if options.get("beam_size") is not None:
        decoder = BeamSearchDecoder(
            options.get("beam_size"), tokenizer.eot, options.get("patience")
        )
    else:
        decoder = GreedyDecoder(options.get("temperature"), tokenizer.eot)

    decoder.reset()
    n_audio = mel.shape[0]

    audio_features = get_audio_features(enc_net, mel)
    tokens = np.repeat(np.array([initial_tokens]), n_audio, axis=-1)
    languages = [language] * audio_features.shape[0]

    # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
    audio_features = np.repeat(audio_features, n_group, axis=0)
    tokens = np.repeat(tokens, n_group, axis=0)

    n_batch = tokens.shape[0]
    sum_logprobs = np.zeros(n_batch)
    no_speech_probs = [np.nan] * n_batch
    initial_token_length = len(initial_tokens)
    kv_cache = None

    # sampling loop
    for i in range(sample_len):
        constant_audio_feature = i >= 2
        logits, kv_cache = inference_logits(
            dec_net,
            tokens,
            audio_features,
            kv_cache,
            initial_token_length,
            constant_audio_feature,
        )

        if i == 0 and tokenizer.no_speech is not None:  # save no_speech_probs
            probs_at_sot = softmax(logits[:, sot_index], axis=-1)
            no_speech_probs = probs_at_sot[:, tokenizer.no_speech].tolist()

        # now we need to consider the logits at the last token only
        logits = logits[:, -1]

        # apply the logit filters, e.g. for suppressing or applying penalty to
        for logit_filter in logit_filters:
            logit_filter.apply(logits, tokens)

        def rearrange_kv_cache(source_indices):
            if kv_cache is not None:
                kv_cache[...] = kv_cache[:, source_indices]

        # expand the tokens tensor with the selected next tokens
        tokens, completed = decoder.update(
            tokens, logits, sum_logprobs, rearrange_kv_cache
        )

        if completed or tokens.shape[-1] > n_ctx:
            break

        if intermediate:
            texts = [
                tokenizer.decode(t[len(initial_tokens) :]).strip()
                for t in tokens
            ]
            print(texts[0][-32:] + "\n\u001B[2A")

    # reshape the tensors to have (n_audio, n_group) as the first two dimensions
    audio_features = audio_features[::n_group]
    no_speech_probs = no_speech_probs[::n_group]
    assert audio_features.shape[0] == len(no_speech_probs) == n_audio

    tokens = tokens.reshape(n_audio, n_group, -1)
    sum_logprobs = sum_logprobs.reshape(n_audio, n_group)

    # get the final candidates for each group, and slice between the first sampled token and EOT
    tokens, sum_logprobs = decoder.finalize(tokens, sum_logprobs)
    tokens = [
        [t[sample_begin : np.nonzero(t == tokenizer.eot)[0][0]] for t in s]
        for s in tokens
    ]

    # select the top-ranked sample in each group
    selected = sequence_ranker.rank(tokens, sum_logprobs)
    tokens = [t[i].tolist() for i, t in zip(selected, tokens)]
    texts = [tokenizer.decode(t).strip() for t in tokens]

    sum_logprobs = [lp[i] for i, lp in zip(selected, sum_logprobs)]
    avg_logprobs = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

    fields = (
        texts,
        languages,
        tokens,
        audio_features,
        avg_logprobs,
        no_speech_probs,
    )
    if len(set(map(len, fields))) != 1:
        raise RuntimeError(
            f"inconsistent result lengths: {list(map(len, fields))}"
        )

    result = [
        DecodingResult(
            audio_features=features,
            language=language,
            language_probs=None,
            tokens=tokens,
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            temperature=options.get("temperature"),
        )
        for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
            *fields
        )
    ]

    return result


def decode_with_fallback(enc_net, dec_net, segment, decode_options):
    logprob_threshold = decode_options.get("logprob_threshold", -1.0)
    temperature = decode_options.get("temperature", 0)

    temperatures = (
        [temperature] if isinstance(temperature, (int, float)) else temperature
    )

    kwargs = {**decode_options}
    t = temperatures[0]
    if t == 0:
        best_of = kwargs.pop("best_of", None)
    else:
        best_of = kwargs.get("best_of", None)

    options = {**kwargs, "temperature": t}
    results = decode(enc_net, dec_net, segment, options)

    kwargs.pop("beam_size", None)  # no beam search for t > 0
    kwargs.pop("patience", None)  # no patience for t > 0
    kwargs["best_of"] = best_of  # enable best_of for t > 0
    for t in temperatures[1:]:
        needs_fallback = [
            result.avg_logprob < logprob_threshold for result in results
        ]
        if any(needs_fallback):
            options = {**kwargs, "temperature": t}
            retries = decode(
                enc_net, dec_net, segment[needs_fallback], options
            )
            for retry_index, original_index in enumerate(
                np.nonzero(needs_fallback)[0]
            ):
                results[original_index] = retries[retry_index]

    return results


def predict(wav, enc_net, dec_net, immediate=True, microphone=False) -> dict:
    temperature = 0
    if temperature_increment_on_fallback is not None:
        temperature = tuple(
            np.arange(
                temperature, 1.0 + 1e-6, temperature_increment_on_fallback
            )
        )
    else:
        temperature = [temperature]

    decode_options = {
        "task": task,
        "language": language,
        "temperature": temperature,
        "best_of": best_of,
        "beam_size": beam_size,
        "patience": patience,
        "length_penalty": length_penalty,
        "suppress_tokens": suppress_tokens,
        "logprob_threshold": logprob_threshold,
        "prompt": [],
    }

    mel = log_mel_spectrogram(wav)
    mel = np.expand_dims(mel, axis=0)

    seek = 0
    input_stride = (
        N_FRAMES // dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    def add_segment(start, end, text_tokens, result):
        text = tokenizer.decode(
            [token for token in text_tokens if token < tokenizer.eot]
        )
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                # "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )

    num_frames = mel.shape[-1]
    previous_seek_value = seek

    pbar = tqdm.tqdm(
        total=num_frames, unit="frames", disable=immediate is not False
    )

    while seek < num_frames:
        timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
        segment = pad_or_trim(mel[:, :, seek:], N_FRAMES)
        segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

        decode_options["prompt"] = all_tokens[prompt_reset_since:]
        result = decode_with_fallback(
            enc_net, dec_net, segment, decode_options
        )
        result = result[0]
        tokens = np.array(result.tokens)

        # no voice activity check
        should_skip = result.no_speech_prob > no_speech_threshold
        if (
            logprob_threshold is not None
            and result.avg_logprob > logprob_threshold
        ):
            # don't skip if the logprob is high enough, despite the no_speech_prob
            should_skip = False

        if should_skip:
            seek += segment.shape[
                -1
            ]  # fast-forward to the next segment boundary
            continue

        timestamp_tokens = tokens >= tokenizer.timestamp_begin
        consecutive = (
            np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        )

        if len(consecutive) > 0:
            # if the output contains two consecutive timestamp tokens
            last_slice = 0
            for current_slice in consecutive:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = (
                    sliced_tokens[0] - tokenizer.timestamp_begin
                )
                end_timestamp_position = (
                    sliced_tokens[-1] - tokenizer.timestamp_begin
                )
                add_segment(
                    start=timestamp_offset
                    + start_timestamp_position * time_precision,
                    end=timestamp_offset
                    + end_timestamp_position * time_precision,
                    text_tokens=sliced_tokens[1:-1],
                    result=result,
                )
                last_slice = current_slice
            last_timestamp_position = (
                tokens[last_slice - 1] - tokenizer.timestamp_begin
            )
            seek += last_timestamp_position * input_stride
            all_tokens.extend(tokens[: last_slice + 1].tolist())
        else:
            duration = segment_duration
            timestamps = tokens[np.nonzero(timestamp_tokens)[0]]
            if len(timestamps) > 0:
                # no consecutive timestamps but it has a timestamp; use the last one.
                # single timestamp at the end means no speech after the last timestamp.
                last_timestamp_position = (
                    timestamps[-1] - tokenizer.timestamp_begin
                )
                duration = last_timestamp_position * time_precision

            add_segment(
                start=timestamp_offset,
                end=timestamp_offset + duration,
                text_tokens=tokens,
                result=result,
            )
            seek += segment.shape[-1]
            all_tokens.extend(tokens.tolist())

        pbar.update(min(num_frames, seek) - previous_seek_value)
        previous_seek_value = seek

    d = dict(
        text=tokenizer.decode(all_tokens),
        segments=all_segments,
        language=language,
    )
    return d


def recognize_from_audio(
    enc_net: onnxruntime.InferenceSession,
    dec_net: onnxruntime.InferenceSession,
) -> dict:
    immediate = True
    wav = load_audio(AUDIO_PATH)

    start = int(round(time.time() * 1000))
    result = predict(
        wav, enc_net, dec_net, immediate=immediate, microphone=False
    )
    end = int(round(time.time() * 1000))
    estimation_time = end - start
    print(f"\ttotal processing time {estimation_time} ms")

    return result


def main(args: argparse.Namespace) -> None:
    providers = ["CPUExecutionProvider"]
    # providers = ["CUDAExecutionProvider"]

    if args.quant:
        enc_net = onnxruntime.InferenceSession(
            WEIGHT_ENC_QUANT_PATH, providers=providers
        )
        dec_net = onnxruntime.InferenceSession(
            WEIGHT_DEC_QUANT_PATH, providers=providers
        )
    else:
        enc_net = onnxruntime.InferenceSession(
            WEIGHT_ENC_PATH, providers=providers
        )
        dec_net = onnxruntime.InferenceSession(
            WEIGHT_DEC_PATH, providers=providers
        )

    result = recognize_from_audio(enc_net, dec_net)

    print(result["text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant', action='store_true', help='Use quant models')
    args = parser.parse_args()
    main(args)
