# Classroom Behavior Monitoring

[Tiếng Việt](README.vi.md)

Near-real-time classroom monitoring for long-duration student behaviors:

- `sleeping`
- `using_phone`

Current runtime pipeline:

`PicoDet -> OC_SORT -> PPLCNet_x1_0 -> Temporal Buffer -> Behavior State Machine -> Telegram Alert`

The project is optimized for stable per-track alerting over time, not for isolated frame-by-frame action labels.

## Repository Scope

This repository keeps:

- runtime code
- deploy scripts
- benchmark and plotting utilities
- public documentation

This repository does not store large runtime assets in Git:

- Paddle inference models
- sample videos
- generated outputs
- local scratch files

Download those assets from [docs/ARTIFACTS.md](docs/ARTIFACTS.md).

## Tested Setup

The current public setup has been checked on:

- Windows
- Python `3.9.25`
- PaddlePaddle GPU `3.2.2`
- NumPy `1.23.5`
- OpenCV `4.5.5`

`paddle.utils.run_check()` passed in the verified `paddle_det` environment on the author's machine.

## Installation

The repository provides two installation paths:

1. a verified Conda environment for Windows + GPU
2. a plain `venv` + `pip` flow for users who want a lighter setup

`requirements.txt` intentionally excludes PaddlePaddle because CPU and GPU installations differ by machine.

### Option A: Verified Conda Environment

Use this path if you want the closest match to the author's working environment.

```powershell
conda env create -f deploy_bundle/environment/environment.demo.yml
conda activate paddle_det
```

The verified package list is tracked in [deploy_bundle/environment/verified_versions.txt](deploy_bundle/environment/verified_versions.txt).

### Option B: Plain `venv` + `pip`

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install exactly one PaddlePaddle package first:

Windows GPU path verified in this repository:

```powershell
python -m pip install paddlepaddle-gpu==3.2.2
```

CPU-only alternative:

```powershell
python -m pip install paddlepaddle==3.2.2
```

Then install the shared Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

If your CUDA or platform setup differs, select the matching PaddlePaddle wheel from the official install guide before continuing.

### Smoke Test

Run these checks before restoring models:

```powershell
python -c "import cv2, numpy, paddle, yaml, scipy, PIL, numba; print('Imports OK')"
python -c "import paddle; paddle.utils.run_check()"
```

## Restore Runtime Artifacts

This repository is not runnable until you restore the external assets listed in [docs/ARTIFACTS.md](docs/ARTIFACTS.md).

Minimum required paths:

- `output_inference/picodet_m_416_classroom`
- `output_inference/pplcnet_behavior`
- optionally `test_data/data.mp4` for the default demo

`output_inference/` is the canonical runtime artifact directory for this repository.

## Quick Start

### 1. Run the Deploy Bundle

Default video demo:

```powershell
.\deploy_bundle\run_demo.ps1
```

Webcam:

```powershell
.\deploy_bundle\run_demo.ps1 -SourceType camera -CameraId 0
```

RTSP:

```powershell
.\deploy_bundle\run_demo.ps1 -SourceType rtsp -RtspUrl "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101"
```

Dry-run without starting inference:

```powershell
.\deploy_bundle\run_demo.ps1 -DryRun
```

### 2. Run the Runtime Script Directly

Video file:

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --video_file test_data/data.mp4 `
  --device gpu
```

RTSP local preview:

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu
```

RTSP republish through mediaMTX:

```powershell
python deploy/pipeline/pipeline_product.py `
  --config deploy/pipeline/config/infer_cfg_pphuman.yml `
  --rtsp "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101" `
  --device gpu `
  --pushurl "rtsp://127.0.0.1:8554/output"
```

More live-stream notes are in [RTSP_DEMO_GUIDE.md](RTSP_DEMO_GUIDE.md).

## Telegram Setup

Telegram alerts are disabled by default.

To enable them:

```powershell
Copy-Item deploy_bundle\secrets\telegram.env.example deploy_bundle\secrets\telegram.env
```

Fill these keys in `deploy_bundle/secrets/telegram.env`:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Then run:

```powershell
.\deploy_bundle\run_demo.ps1 -UseTelegram
```

Do not commit `deploy_bundle/secrets/telegram.env`.

## Main Entry Points

- runtime script: [deploy/pipeline/pipeline_product.py](deploy/pipeline/pipeline_product.py)
- public launcher: [deploy_bundle/run_demo.ps1](deploy_bundle/run_demo.ps1)
- main runtime config: [deploy/pipeline/config/infer_cfg_pphuman.yml](deploy/pipeline/config/infer_cfg_pphuman.yml)
- deploy bundle config: [deploy_bundle/config/demo_final.yml](deploy_bundle/config/demo_final.yml)

Important defaults:

- `MOT.skip_frame_num = 2`
- `ID_BASED_CLSACTION.skip_frame_num = 2`
- `ID_BASED_CLSACTION.crop_mode = full`
- `sleep_warn_seconds = 5.0`
- `sleep_alert_seconds = 12.0`
- `phone_warn_seconds = 6.0`
- `phone_alert_seconds = 15.0`
- `TELEGRAM_ALERT.enable = False`

## Repository Layout

- `deploy/`: runtime code plus vendored PaddleDetection and PP-Tracking modules
- `deploy_bundle/`: clean demo launcher and environment notes
- `docs/`: public support documents such as artifact download instructions
- `tools/`: benchmarking and plotting utilities
- `output_inference/`: downloaded runtime artifacts, excluded from Git
- `test_data/`: downloaded demo videos, excluded from Git
- `output/`: generated outputs, excluded from Git
- `tmp/`: scratch space, excluded from Git

## Notes

- The public repository is centered on the classroom runtime, not on training code.
- Legacy local mirrors may still exist under `models/`, but `output_inference/` is the source of truth for deployment.
- The PowerShell launcher is Windows-oriented. On other platforms, run `pipeline_product.py` directly.
