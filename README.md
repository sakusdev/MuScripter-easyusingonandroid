# MuScriptor Easy for Android

A **fully local / offline** Android port of [MuScriptor](https://github.com/muscriptor/muscriptor).

The goal is simple:

> Pick audio on Android → transcribe locally on the phone → export MIDI.

No inference server. No cloud upload. The Android manifest intentionally does **not** request the `INTERNET` permission.

## Current status

The Android bootstrap builds successfully in GitHub Actions.

Already in the repo:

- Jetpack Compose Android shell
- Storage Access Framework audio/video picker
- local `.pte` model importer into app-private storage
- ExecuTorch Android 1.4 runtime + model load boundary
- Kotlin MT3 token vocabulary / streaming note decoder
- cross-chunk tie/prelude state handling
- MT3 unit tests
- Android CI producing a debug APK artifact
- detailed MuScriptor → ExecuTorch port architecture

Still to wire for real transcription:

- MuScriptor `.safetensors` → Android-friendly `.pte` exporter
- Android PCM decode/resample + exact STFT/log-mel frontend
- stateful ExecuTorch autoregressive decoder with KV cache
- generation loop and model/decoder integration
- MIDI writer / piano-roll result UI

See [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md).

## Target architecture

```text
Audio / video
    ↓
Android PCM decoder + resampler
    ↓
16 kHz mono
    ↓
STFT + 512-bin log-mel
    ↓
frontend.pte
    ↓
conditioning prefix
    ↓
decoder.pte + model-owned KV cache
    ↓
greedy MT3 tokens
    ↓
Kotlin MT3 state machine
    ↓
MIDI / piano roll
```

The first end-to-end target is **MuScriptor Small** on XNNPACK/CPU. Medium is the main quality target after parity and memory profiling are solid.

## Android stack

- Android Gradle Plugin 9.3
- compileSdk / targetSdk 36
- Jetpack Compose API-36-compatible stack
- ExecuTorch Android 1.4
- Java 17
- minimum Android API 26

## Building

Android Studio with Android API 36 installed can open the project directly.

From a command line with Gradle 9.5 installed:

```bash
gradle :app:testDebugUnitTest :app:assembleDebug
```

GitHub Actions runs the unit tests, builds the debug APK, and uploads it as the `muscriptor-easy-debug` artifact on every push to `main`.

## Model files

Model weights and generated `.pte` bundles are intentionally gitignored.

The app imports a user-selected `.pte` into app-private storage:

```text
files/models/
```

This avoids bundling hundreds of megabytes of model data into the APK and keeps inference local after installation.

## MT3 parity work

The Android/Kotlin decoder already uses MuScriptor's generated-token vocabulary layout (1393 generated token IDs), 100 Hz timing, note retrigger handling, drums, and the cross-chunk tie prologue used by upstream MuScriptor. The remaining parity-critical part is exporting the neural frontend/decoder so its greedy token stream matches desktop PyTorch.

The planned regression fixture is the existing Ferris Wheel 85–90 s Medium transcription, which emits 377 tokens before EOS on the desktop reference path.

## Licensing

MuScriptor's source code is MIT-licensed. The published MuScriptor model weights are CC BY-NC 4.0; respect the upstream model license when distributing model bundles or applications that use them.

ExecuTorch is distributed under its upstream license.
