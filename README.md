# Classroom Behavior Monitoring

[Tiếng Việt](README.vi.md)

Classroom Behavior Monitoring is a near-real-time inference project for long-duration student behavior detection in classroom scenes.

The current production-oriented runtime focuses on two behaviors:

- `sleeping`
- `using_phone`

The system is designed for stable per-track monitoring and alerting, not for isolated single-frame action labels.

## Table of Contents

- [Overview](#overview)
- [Repository Scope](#repository-scope)
- [System Pipeline](#system-pipeline)
- [Repository Layout](#repository-layout)
- [Verified Environment](#verified-environment)
- [Installation](#installation)
- [Restore Runtime Artifacts](#restore-runtime-artifacts)
- [Quick Start](#quick-start)
- [Run the Web UI](#run-the-web-ui)
- [Run the Deploy Bundle](#run-the-deploy-bundle)
- [Run the Runtime Script Directly](#run-the-runtime-script-directly)
- [Telegram Alerts](#telegram-alerts)
- [Key Configuration Files](#key-configuration-files)
- [Performance Notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)

## Overview

This repository packages the runtime, launchers, configuration, and support tooling needed to run classroom behavior monitoring on:

- video files
- laptop webcams
- RTSP streams

The project now includes two operator paths:

1. a browser-based Web UI under [`app/`](app)
2. Windows-friendly CLI launchers under [`deploy_bundle/`](deploy_bundle)

Use the Web UI when you want a simpler operator workflow with browser preview. Use the deploy bundle when you want a direct script-based demo flow.

## Repository Scope

This repository includes:

- inference runtime code
- deployment launchers
- Web UI
- benchmark and plotting utilities
- public documentation

This repository does not keep large runtime assets in Git:

- inference model weights
- sample videos
- generated outputs
- local scratch files

Restore those assets using [docs/ARTIFACTS.md](docs/ARTIFACTS.md).

## System Pipeline

Current runtime pipeline:

`PicoDet -> OC_SORT -> PPLCNet_x1_0 -> Temporal Filter -> Behavior State Machine -> Telegram Alert`

At a high level:

- `PicoDet` detects students
- `OC_SORT` tracks students across frames
- `PPLCNet_x1_0` classifies behavior crops
- the temporal filter smooths noisy frame-level predictions
- the behavior state machine decides when to emit warning/alert states
- Telegram delivery is optional

## Repository Layout

- [`app/`](app): browser-based Web UI built with FastAPI
- [`deploy/`](deploy): runtime code plus vendored PaddleDetection / PP-Tracking modules
- [`deploy_bundle/`](deploy_bundle): clean deployment launchers, config, environment notes, secret templates
- [`docs/`](docs): support documentation such as artifact recovery
- [`tools/`](tools): benchmark and analysis utilities
- `output_inference/`: restored inference artifacts, not tracked in Git
- `test_data/`: restored demo videos, not tracked in Git
- `output/`: generated outputs, not tracked in Git
- `tmp/`: scratch space, not tracked in Git

## Verified Environment

The current verified public setup was checked on:

- Windows
- Python `3.9.25`
- PaddlePaddle GPU `3.2.2`
- NumPy `1.23.5`
- OpenCV `4.5.5`

The verified Conda environment file is:

- [deploy_bundle/environment/environment.demo.yml](deploy_bundle/environment/environment.demo.yml)

The verified package snapshot is:

- [deploy_bundle/environment/verified_versions.txt](deploy_bundle/environment/verified_versions.txt)

## Installation

There are two supported setup paths:

1. verified Conda environment
2. plain Python environment with manual Paddle installation

`requirements.txt` intentionally excludes PaddlePaddle because CPU and GPU installs differ by machine.

### Option A: Verified Conda Environment

Use this path if you want the closest match to the validated environment:

```powershell
conda env create -f deploy_bundle/environment/environment.demo.yml
conda activate paddle_det
```

### Option B: Plain `venv` + `pip`

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install exactly one PaddlePaddle package first.

Verified Windows GPU path:

```powershell
python -m pip install paddlepaddle-gpu==3.2.2
```

CPU-only alternative:

```powershell
python -m pip install paddlepaddle==3.2.2
```

Then install the shared dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Smoke Check

Run these checks before restoring large artifacts:

```powershell
python -c "import cv2, numpy, paddle, yaml, scipy, PIL, numba; print('Imports OK')"
python -c "import paddle; paddle.utils.run_check()"
```

## Restore Runtime Artifacts

The repository is not runnable until the external runtime assets are restored.

See:

- [docs/ARTIFACTS.md](docs/ARTIFACTS.md)

Minimum required paths:

- `output_inference/picodet_m_416_classroom`
- `output_inference/pplcnet_behavior`
- optionally `test_data/data.mp4` for the default demo flow

`output_inference/` is the canonical runtime artifact directory used by the current deployment path.

## Quick Start

If you want the shortest working path:

1. Create and activate `paddle_det`
2. Restore the required model artifacts
3. Install Web UI dependencies:

```powershell
pip install -r app/requirements.txt
```

4. Start the browser UI:

```bat
app\run_web.cmd
```

5. Open:

- `http://127.0.0.1:8000`

If you prefer script-based execution instead of the web app, use the deploy bundle described below.

## Run the Web UI

The Web UI is the recommended operator entrypoint when you want:

- browser-based preview
- one-click source selection
- session logs and alerts in one place
- no OpenCV preview window on the desktop

Detailed app notes live in:

- [app/README.md](app/README.md)

### What the Web UI Supports

- video file inference
- webcam inference
- RTSP inference

### Install Web UI Dependencies

Install these into the same environment that already runs the inference pipeline:

```powershell
pip install -r app/requirements.txt
```

### Start the Web UI

From `cmd.exe`:

```bat
app\run_web.cmd
```

Or directly with Python:

```powershell
python -m app
```

Default address:

- `http://127.0.0.1:8000`

Optional environment variables:

- `APP_HOST`
- `APP_PORT`
- `MEDIAMTX_BIN`
- `MEDIAMTX_HOST`
- `MEDIAMTX_RTSP_PORT`
- `MEDIAMTX_WEBRTC_PORT`

Example:

```powershell
$env:APP_HOST="0.0.0.0"
$env:APP_PORT="8000"
$env:MEDIAMTX_BIN="C:\tools\mediamtx.exe"
python -m app
```

Important operational notes:

- the Web UI runs a single active inference session at a time
- browser preview uses MediaMTX WebRTC
- `preview_local=False` is forced, so the app does not open a separate OpenCV window
- the app is intentionally single-worker because session state is kept in process memory

## Run the Deploy Bundle

The deploy bundle is the script-based operator path for Windows.

Primary files:

- [deploy_bundle/run_demo.ps1](deploy_bundle/run_demo.ps1)
- [deploy_bundle/run_demo.cmd](deploy_bundle/run_demo.cmd)
- [deploy_bundle/menu.cmd](deploy_bundle/menu.cmd)

### Default Video Demo

```powershell
.\deploy_bundle\run_demo.ps1
```

### Native CMD Launcher

```bat
deploy_bundle\run_demo.cmd video
```

### Interactive CMD Menu

```bat
deploy_bundle\menu.cmd
```

### Webcam

```powershell
.\deploy_bundle\run_demo.ps1 -SourceType camera -CameraId 0
```

### RTSP

```powershell
.\deploy_bundle\run_demo.ps1 -SourceType rtsp -RtspUrl "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101"
```

### Dry Run

```powershell
.\deploy_bundle\run_demo.ps1 -DryRun
```

## Run the Runtime Script Directly

Use the runtime script directly when you want the lowest-level execution path.

Main runtime:

- [deploy/pipeline/pipeline_product.py](deploy/pipeline/pipeline_product.py)

### Video File

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --video_file test_data/data.mp4 `
  --device gpu
```

### RTSP

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu
```

### RTSP With Local Preview and No Saved Video

When calling `pipeline_product.py` directly, preview and save toggles must be passed through `-o/--opt`.

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu `
  -o preview_local=True save_visual_output=False
```

### RTSP Republish Through mediaMTX

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu `
  --pushurl "rtsp://127.0.0.1:8554/output"
```

More live-stream notes are documented in:

- [RTSP_DEMO_GUIDE.md](RTSP_DEMO_GUIDE.md)

## Telegram Alerts

Telegram alerts are disabled by default.

To enable them:

1. Copy the example env file:

```powershell
Copy-Item deploy_bundle\secrets\telegram.env.example deploy_bundle\secrets\telegram.env
```

2. Fill these keys in `deploy_bundle/secrets/telegram.env`:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

3. Run with Telegram enabled:

```powershell
.\deploy_bundle\run_demo.ps1 -UseTelegram
```

Or with CMD:

```bat
deploy_bundle\run_demo.cmd video --use-telegram
```

## Key Configuration Files

Primary runtime config:

- [deploy/pipeline/config/infer_cfg_pphuman.yml](deploy/pipeline/config/infer_cfg_pphuman.yml)

Deploy-bundle config:

- [deploy_bundle/config/demo_final.yml](deploy_bundle/config/demo_final.yml)

Important runtime defaults:

- `MOT.skip_frame_num = 2`
- `ID_BASED_CLSACTION.skip_frame_num = 2`
- `ID_BASED_CLSACTION.crop_mode = full`
- `preview_local = auto`
- `save_visual_output = auto`
- `sleep_warn_seconds = 5.0`
- `sleep_alert_seconds = 12.0`
- `phone_warn_seconds = 6.0`
- `phone_alert_seconds = 15.0`
- `TELEGRAM_ALERT.enable = False`

## Performance Notes

For lower latency and a smoother operator experience:

- prefer the Web UI or local preview over RTSP republish if you are operating on the same machine
- keep `save_visual_output=False` for live runs unless you explicitly need recorded output
- use the camera sub-stream instead of the main stream when available
- watch `Frame Age` in the Web UI instead of assuming `Processing FPS` equals what the viewer feels
- keep `pushurl` disabled unless you really need external stream redistribution

Practical trade-off:

- MediaMTX WebRTC gives a better browser preview path than in-app MJPEG
- RTSP republish still introduces more buffering and encoding stages than a pure local preview window

## Troubleshooting

### `Config not found` or missing models

Make sure the required config file exists and the external model artifacts were restored to `output_inference/`.

### `--preview_local` or `--no-save-video` is rejected by `pipeline_product.py`

This is expected when calling the runtime script directly. Those switches are launcher-level conveniences. Use:

```powershell
-o preview_local=True save_visual_output=False
```

### Web UI starts but browser preview does not

Check:

- you installed `app/requirements.txt` into the active inference environment
- `mediamtx.exe` is installed and reachable through `PATH` or `MEDIAMTX_BIN`
- the selected source path or RTSP URL is valid
- the model directories are present
- the session log panel for runtime errors

### RTSP is far behind real time

This is usually caused by stream buffering, decode cost, re-encode cost, or downstream player buffering. Start by:

- using a lower-resolution RTSP sub-stream
- disabling output saving
- keeping MediaMTX on the same machine as the app
- avoiding unnecessary republish paths

## Security Notes

- Do not commit `deploy_bundle/secrets/telegram.env`
- Do not store production secrets in tracked YAML files
- If a Telegram token or chat ID was ever exposed, rotate it before deployment
- Keep the Web UI on `127.0.0.1` unless you intentionally place it behind network controls or a reverse proxy

## Acknowledgements

This project builds on components and runtime ideas from:

- PaddleDetection
- PP-Tracking
- OpenCV
- FastAPI
