# MuScriptor Android port plan

Goal: run the real MuScriptor model entirely on Android with **no server and no network permission**.

## Why this needs an adapter

Upstream MuScriptor is not a single `audio -> MIDI` tensor graph. It has:

1. audio decode / resample to mono 16 kHz
2. 2048-point STFT, hop 160
3. 512-bin HTK mel filterbank + `log(mel + 1e-6)`
4. learned conditioning projection
5. an autoregressive Transformer decoder
6. per-layer KV caches
7. MT3 token decoding / cross-chunk tie handling
8. MIDI serialization

Trying to export the Python orchestration directly would make the Android graph fragile and slow. The port keeps the neural math in ExecuTorch and moves deterministic orchestration to Kotlin/native code.

## Runtime split

### A. Android DSP

Input: arbitrary audio/video selected through Storage Access Framework.

Output per 5-second chunk:

```text
log_mel: float32 [1, ~501, 512]
```

Must numerically match upstream:

- sample rate: 16000
- n_fft: 2048
- hop: 160
- Hann window
- centered STFT with reflect padding
- HTK mel scale
- 512 mel bins
- power = 1.0
- log epsilon = 1e-6

### B. `frontend.pte` (stateless)

Contains only learned conditioning layers from the checkpoint:

- mel `output_proj`
- instrument-group embedding
- dataset embedding

Input:

```text
log_mel          float32 [1, T, 512]
instrument_group int64   [1, N]
dataset_name     int64   [1, 1]
```

Output:

```text
condition_prefix float32 [1, P, D]
```

The output order must match upstream `ConditioningProvider` exactly.

### C. token embedding table

The token embedding is tiny compared with the Transformer and can be exported as a compact model asset/table.

```text
small:  (card+1) x 768
medium: (card+1) x 1024
```

Android performs a direct lookup to create `[1, 1, D]` token embeddings.

This lets the decoder accept **embeddings** rather than sometimes token IDs and sometimes audio soft-prefix tokens.

### D. `decoder.pte` (stateful)

Contains:

- positional embedding math
- all Transformer blocks
- model-owned K/V cache buffers
- output norm
- output linear head

Forward ABI:

```text
forward(
    x:         float [1, T, D],
    input_pos: int64 [1]
) -> logits float [1, card]
```

First call for a 5-second chunk:

```text
x = concat(condition_prefix, initial_token_embedding)
input_pos = 0
```

Later calls:

```text
x = embedding(next_token)   # T = 1
input_pos += previous_T
```

The same method therefore handles prefill and decode without a Python/data-dependent branch. `T` is dynamic: large on prefill, 1 during decode.

## KV cache

Do **not** send the full KV cache through the Java API on every token.

Refactor MuScriptor attention to use export-friendly, model-owned buffers similar to ExecuTorch's LLM `KVCache`:

```text
k_cache[layer]: [1, heads, max_context, head_dim]
v_cache[layer]: [1, heads, max_context, head_dim]
```

Update by tensor `input_pos`, not Python integer counters.

For MuScriptor this is especially useful because one 5-second jazz chunk can require hundreds of generated MT3 tokens.

## Android generation loop

Pseudocode:

```kotlin
val prefix = frontend.encode(logMel, instrumentGroup, dataset)
var x = concat(prefix, tokenEmbedding(INITIAL_TOKEN))
var pos = 0L

repeat(MAX_GEN_LEN) {
    val logits = decoder.forward(x, pos)
    val token = argmax(maskReservedTokens(logits))
    if (token == EOS) return@repeat

    mt3.feed(token)
    pos += x.sequenceLength
    x = tokenEmbedding(token)
}
```

Default inference must match upstream:

- greedy decoding (`use_sampling = false`)
- cfg coefficient 1.0
- mask token IDs >= 1393
- 5-second chunks
- prelude/tie forcing between chunks

## Cross-chunk tie forcing

After chunk N, Android's MT3 state machine knows every still-open `(program, pitch)` pair.

Chunk N+1 starts with the exact tie prologue produced by upstream `tie_section_token_ids()` before normal generation. This is necessary for instrument identity and sustained-note quality at 5-second boundaries.

## Models

Initial targets:

| Variant | Dim | Layers | Heads | Priority |
| --- | ---: | ---: | ---: | --- |
| Small | 768 | 14 | 12 | First end-to-end target |
| Medium | 1024 | 24 | 16 | Main quality target |
| Large | 1536 | 48 | 24 | Later experiment |

Start with float32/XNNPACK for correctness. Then benchmark:

1. FP32 XNNPACK reference
2. dynamic activation + INT8 weight
3. 8da4w / low-bit weight path
4. Vulkan PT2E path

Never quantize before comparing Android logits/notes against desktop PyTorch on the same 5-second fixtures.

## Validation fixture

Use the existing Ferris Wheel chorus fixture because it is polyphonic and already has desktop MuScriptor results.

Required parity checks:

- frontend prefix max/mean absolute error
- first-step logits top-k
- first 50 greedy tokens exact equality
- complete 5-second token stream
- decoded note onset/pitch/instrument equality
- MIDI note list equality

A useful first target is the 85–90 s chunk that desktop Medium decoded to 377 tokens before EOS.

## Milestones

### M0 — Android shell

- [x] Compose project
- [x] no INTERNET permission
- [x] import `.pte` into app-private storage
- [x] ExecuTorch runtime load boundary
- [ ] CI green

### M1 — Small decoder parity

- [ ] export-friendly attention/KV cache adapter
- [ ] `frontend.pte`
- [ ] `decoder.pte`
- [ ] token embedding asset
- [ ] Python parity test

### M2 — Android DSP + 1 chunk

- [ ] PCM decode
- [ ] exact 16 kHz resample
- [ ] STFT + mel implementation
- [ ] one 5-second Small transcription on device
- [ ] compare against desktop token fixture

### M3 — full audio

- [ ] 5-second chunk scheduler
- [ ] MT3 decoder in Kotlin
- [ ] tie/prelude forcing
- [ ] MIDI writer
- [ ] progress + piano roll UI

### M4 — Medium

- [ ] export Medium
- [ ] memory benchmark
- [ ] INT8 / low-bit benchmark
- [ ] Pixel-class device profiling

### M5 — acceleration

- [ ] Vulkan build flavor
- [ ] optional Qualcomm QNN backend
- [ ] choose backend automatically from device capability

## References

- Upstream MuScriptor: https://github.com/muscriptor/muscriptor
- ExecuTorch Android: https://docs.pytorch.org/executorch/stable/using-executorch-android.html
- ExecuTorch custom autoregressive/KV-cache guidance: https://github.com/pytorch/executorch/blob/main/docs/source/llm/export-custom-llm.md
