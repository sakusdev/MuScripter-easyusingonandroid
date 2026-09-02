# MuScriptor Android model ABI v1

The first Android parity target uses a deliberately static ExecuTorch ABI. It is slower than the final design, but keeps every inference shape fixed and makes desktop-vs-Android numerical debugging much easier.

## Bundle

A model is distributed/imported as a `.msa` zip containing exactly:

```text
manifest.json
conditioner.pte
embedder.pte
decoder.pte
```

Model weights remain subject to the upstream MuScriptor model license. Do not commit generated bundles to this repository.

## Frontend

The Android app must reproduce upstream MuScriptor's 5-second audio frontend exactly:

- mono, 16,000 Hz
- segment length: 80,000 samples
- `n_fft = win_length = 2048`
- periodic Hann window
- hop length 160 (100 Hz)
- centered STFT with reflect padding
- one-sided, unnormalized complex STFT
- magnitude (`power = 1.0`)
- HTK mel scale, no mel normalization
- 512 mel bins from 0 to 8,000 Hz
- `log(mel + 1e-6)`
- a full 5-second segment yields 501 frames

Short final chunks are zero-padded to the 5-second input shape before the frontend. Their actual audio length still has to be respected when decoding events beyond the end of the file.

## conditioner.pte

Inputs:

```text
log_mel          float32 [1, 501, 512]
instrument_token int64   [1, 1]
dataset_token    int64   [1, 1]
```

Output:

```text
prefix float32 [1, 503, D]
```

`D` is 768 for Small, 1024 for Medium, and 1536 for Large.

Class inputs are the tokenized representation used by upstream `ClassConditioner`: 0 is the null condition. The v1 app defaults both class conditions to 0.

The prefix order is intentionally:

```text
self_wav (501), dataset_name (1), instrument_group (1)
```

Upstream creates the dictionary in instrument/dataset/wav order but prepends each tensor, so the actual transformer order is reversed. Matching this is parity-critical.

## embedder.pte

Input:

```text
token_id int64 [1, 1]
```

Output:

```text
embedding float32 [1, 1, D]
```

The first token of every chunk is MuScriptor's initial token (`card`). Subsequent calls use generated MT3 token IDs or teacher-forced tie-prelude IDs.

## decoder.pte

Inputs:

```text
embedding float32 [1, 1, D]
input_pos int64   [1]
```

Output:

```text
logits float32 [1, card]
```

The decoder owns K/V caches as mutable registered buffers. `input_pos` is the absolute transformer position inside the current chunk. Starting again at position 0 logically resets the cache because all slots above `input_pos` are masked until overwritten.

Model cardinality is 1393 for Small and 1395 for current Medium/Large checkpoints. Regardless of model cardinality, only generated token IDs 0..1392 are valid; logits for reserved IDs are masked before greedy selection.

## Bring-up generation loop

For each 5-second chunk:

1. Compute log-mel and run `conditioner.pte`.
2. Set `input_pos = 0`.
3. Feed the 503 prefix embeddings into `decoder.pte` **one position at a time**. Ignore the logits from those calls.
4. Run `embedder.pte` for the initial token and feed that embedding to the decoder at position 503.
5. Greedily choose the highest valid logit.
6. Feed the chosen token to the Kotlin MT3 decoder.
7. If EOS was not reached, embed that token and call the decoder at the next position.
8. Continue until EOS or the generation limit.
9. The next chunk starts again at decoder position 0.

For chunks after the first, the Kotlin `Mt3StreamDecoder.tieSectionTokenIds()` sequence is teacher-forced before free generation, matching upstream prelude forcing.

This v1 loop performs 503 decoder calls just to prefill conditions. That is intentionally temporary. Once parity is established, a multi-token prefill method can replace steps 2–3 while leaving the checkpoint weights, MT3 logic, and decode ABI unchanged.

## Context length

Upstream allows up to 2,000 generation iterations per chunk. With 503 conditioning positions, the default exported cache length is 2,503 positions. This is memory-heavy, especially in fp32, and will be profiled/optimized after Small reaches end-to-end parity.

## Export

From a Python environment containing ExecuTorch 1.4:

```bash
pip install -r tools/requirements-export.txt
python tools/export_muscriptor.py /path/to/model.safetensors \
  --config /path/to/config.json \
  -o build/muscriptor-small-android
```

This creates both the unpacked directory and `build/muscriptor-small-android.msa`.
