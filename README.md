# MuScriptor Easy for Android

A **fully local / offline** Android port of [MuScriptor](https://github.com/muscriptor/muscriptor).

The goal is simple:

> Pick audio on Android → transcribe locally on the phone → save MIDI.

No inference server. No cloud upload. The Android manifest intentionally does **not** request the `INTERNET` permission.

## Current status

The Android app now has the complete local control path wired:

- Jetpack Compose Android UI
- Storage Access Framework audio/video picker
- Android `MediaExtractor` / `MediaCodec` audio decoding
- mono conversion + MuScriptor-style sinc resampling to 16 kHz
- pure-Kotlin 2048-point FFT / periodic Hann / reflect padding
- exact 512-bin HTK log-mel frontend target (`501 x 512` per 5 s chunk)
- `.msa` model-bundle importer into app-private storage
- SHA-256 verification for every exported `.pte` module
- automatic reuse of the most recently imported model after app restart
- ExecuTorch Android 1.4 runtime
- conditioner / token-embedder / stateful decoder modules
- model-owned KV cache with chunk-local masking
- dynamic block prefill (`503` conditions + initial token = `504` rows in one call)
- legacy single-step decoder fallback for older bundles
- greedy autoregressive generation
- MT3 1393-token vocabulary decoder
- cross-chunk tie / prelude forcing
- MuScriptor instrument-group naming
- pure-Kotlin type-1 MIDI writer
- Android `Save MIDI` flow
- unit tests for MT3, frontend/resampling reference fixtures, and MIDI bytes
- ExecuTorch Python-runtime ABI tests for stateful KV cache and dynamic block prefill
- GitHub Actions producing an installable debug APK artifact

The remaining high-value milestone is **real-checkpoint parity on an Android device**: export the official Small/Medium weights into `.msa`, run known audio, and compare generated tokens against desktop MuScriptor.

See [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md).

## Runtime architecture

```text
Audio / video URI
    ↓
MediaExtractor + MediaCodec
    ↓
mono float PCM
    ↓
MuScriptor/Julius-style sinc resampler
    ↓
16 kHz mono
    ↓
5-second chunks
    ↓
2048 FFT + 512-bin HTK log-mel (Kotlin)
    ↓
conditioner.pte
    ↓
503 conditioning embeddings
    + initial model token
    ↓
one 504-row block prefill
    ↓
optional forced tie prelude (1 token / call)
    ↓
decoder.pte + model-owned KV cache
    ↓
greedy MT3 tokens (1 token / call)
    ↓
Kotlin MT3 state machine
    ↓
note events
    ↓
type-1 MIDI
```

The first end-to-end performance target is **MuScriptor Small** on XNNPACK/CPU. Medium is the main quality target after parity and memory profiling are solid.

## Android stack

- Android Gradle Plugin 9.3
- compileSdk / targetSdk 36
- Jetpack Compose API-36-compatible stack
- ExecuTorch Android 1.4
- Java 17
- minimum Android API 26

## Building the app

Android Studio with Android API 36 installed can open the project directly.

From a command line with Gradle 9.5 installed:

```bash
gradle :app:testDebugUnitTest :app:assembleDebug
```

GitHub Actions runs the unit tests, builds the debug APK, and uploads it as the `muscriptor-easy-debug` artifact on every push to `main`.

## Exporting a MuScriptor model

Model weights are **not** committed to this repository. Convert a local MuScriptor `model.safetensors` checkpoint into the Android bundle format on a desktop machine.

Create a Python environment and install the pinned exporter dependencies:

```bash
python -m venv .venv-export
source .venv-export/bin/activate   # Windows: .venv-export\Scripts\activate
python -m pip install -U pip
python -m pip install -r tools/requirements-export.txt
```

Use the optimized dynamic exporter:

```bash
PYTHONPATH=tools python tools/export_muscriptor_dynamic.py /path/to/model.safetensors \
  -o build/muscriptor-small
```

On Windows PowerShell, set `PYTHONPATH` first or run from the `tools` directory:

```powershell
$env:PYTHONPATH = "tools"
python tools/export_muscriptor_dynamic.py C:\path\to\model.safetensors -o build\muscriptor-small
```

If `config.json` is next to the checkpoint it is picked up automatically. Otherwise the official Small/Medium/Large architecture is inferred from the tensors where possible.

The command creates:

```text
build/
├── muscriptor-small/
│   ├── manifest.json
│   ├── conditioner.pte
│   ├── embedder.pte
│   └── decoder.pte
└── muscriptor-small.msa
```

Import the single `.msa` file from the Android app. The bundle manifest records the source-weight SHA-256 plus SHA-256 hashes for all three `.pte` files, and the app verifies the module hashes before loading them.

For a correctness-first backend fallback, pass `--portable` to skip XNNPACK partitioning:

```bash
PYTHONPATH=tools python tools/export_muscriptor_dynamic.py model.safetensors \
  -o build/muscriptor-small-portable \
  --portable
```

`tools/export_muscriptor.py` remains the original single-step ABI exporter. The Android app can still load its bundles, but they feed the 503-row condition prefix one position at a time and are intended mainly as a compatibility/parity fallback.

## `.msa` bundle ABI v1

A bundle is a ZIP with exactly four root entries:

```text
manifest.json
conditioner.pte
embedder.pte
decoder.pte
```

The app rejects extra paths, duplicate entries, missing files, invalid ABI metadata, oversized extraction, and SHA-256 mismatches before any ExecuTorch module is mmap-loaded.

Optimized bundles set:

```json
{
  "decoder_sequence_mode": "dynamic_block_v1",
  "max_prefill_sequence": 504
}
```

in the manifest's `runtime` object. Their decoder uses **one stateful `forward` method** with a dynamic sequence dimension. This is deliberate: the ExecuTorch Android 1.4 `Module` wrapper does not opt into cross-method shared memory arenas, so a single method is the safest way to guarantee the same model-owned KV buffers are used by block prefill and subsequent one-token decoding.

Position `0` starts a logically fresh chunk. Cache slots above each query's absolute position are masked, so stale bytes left by a previous 5-second chunk cannot be attended to.

Older manifests without `decoder_sequence_mode` are interpreted as `single_step` and continue to work.

## MIDI output

Without local beat-grid detection yet, the Android writer matches MuScriptor's fallback MIDI behavior:

- Standard MIDI File type 1
- 480 PPQ
- 120 BPM placeholder tempo
- one track per detected program/instrument group
- melodic channels 0–8 then 10–15
- drums on channel 9
- velocity 100
- human-readable MuScriptor instrument-group track names where available

Tempo/downbeat detection and notation export are separate later milestones; they are not required for raw transcription parity.

## Parity target

The most useful next regression test is a known desktop MuScriptor token stream. For each 5-second chunk we want to compare:

1. Kotlin log-mel vs PyTorch log-mel.
2. conditioning prefix output.
3. block-prefill final logits vs upstream first-step logits.
4. greedy token IDs, including EOS.
5. cross-chunk forced tie prelude.
6. decoded note events / final MIDI.

A previously measured reference is the Ferris Wheel 85–90 s Medium chunk, which emitted 377 tokens before EOS on the desktop path.

## Model storage

Imported bundles live only in app-private storage under:

```text
files/models/
```

The newest imported bundle is loaded automatically on the next app launch. The expensive SHA-256 pass is done at import time; startup revalidates the bundle structure and mmap-loads the modules without rehashing gigabytes of model data.

The APK stays small; model data remains local after import.

## Licensing

MuScriptor's source code is MIT-licensed. The published MuScriptor model weights are CC BY-NC 4.0; respect the upstream model license when distributing model bundles or applications that use them.

ExecuTorch is distributed under its upstream license.
