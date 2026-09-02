# MuScriptor Easy for Android

A **fully local / offline** Android port of [MuScriptor](https://github.com/muscriptor/muscriptor).

The goal is simple:

> Pick audio on Android → transcribe locally on the phone → export MIDI.

No inference server. No cloud upload. The Android manifest intentionally does **not** request the `INTERNET` permission.

## Current status

Early porting work.

Already in the repo:

- Jetpack Compose Android shell
- Storage Access Framework audio/video picker
- local `.pte` model importer
- ExecuTorch Android runtime dependency
- ExecuTorch model load boundary
- Android CI
- detailed MuScriptor → ExecuTorch port architecture

Not wired yet:

- MuScriptor `.safetensors` → Android `.pte` exporter
- Android STFT/mel frontend
- autoregressive decoder loop
- MT3 decoder / MIDI export

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
- compileSdk 37
- Jetpack Compose 1.12 / BOM 2026.08
- ExecuTorch Android 1.4
- Java 17
- minimum Android API 26

## Building

Android Studio with a current API 37 SDK can open the project directly.

From a command line with Gradle 9.5 installed:

```bash
gradle :app:assembleDebug
```

GitHub Actions builds the debug APK on every push to `main`.

## Model files

Model weights and generated `.pte` bundles are intentionally gitignored.

The app imports a user-selected `.pte` into app-private storage:

```text
files/models/
```

This avoids bundling hundreds of megabytes of model data into the APK and keeps inference local after installation.

## Licensing

MuScriptor's source code is MIT-licensed. The published MuScriptor model weights are CC BY-NC 4.0; respect the upstream model license when distributing model bundles or applications that use them.

ExecuTorch is distributed under its upstream license.
