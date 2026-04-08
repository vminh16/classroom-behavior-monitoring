# Deploy Bundle

This folder is the clean public-facing deploy entrypoint for the classroom-monitoring demo.

It keeps only:

- launcher scripts
- deploy config
- environment notes
- secret templates

Large runtime artifacts are restored separately from the Drive bundle described in [../docs/ARTIFACTS.md](../docs/ARTIFACTS.md).

## What This Bundle Runs

- runtime: `deploy/pipeline/pipeline_product.py`
- detector + tracker: PicoDet + OCSORT
- classifier: PPLCNet_x1_0 for `normal`, `using_phone`, `sleeping`
- alert logic: temporal filter + behavior state machine
- Telegram: optional and loaded from a local env file, not from YAML

## Included Files

- `config/demo_final.yml`: final secret-free config for demo
- `config/tracker_config.demo.yml`: final tracker config
- `run_demo.ps1`: main one-command launcher
- `run_demo.bat`: batch wrapper for the PowerShell launcher
- `environment/environment.demo.yml`: minimal environment file
- `environment/verified_versions.txt`: versions verified on the author's machine
- `secrets/telegram.env.example`: template for Telegram secrets

## Required External Artifacts

Before running this bundle, restore:

- `output_inference/picodet_m_416_classroom`
- `output_inference/pplcnet_behavior`
- optionally `test_data/data.mp4` if you want the default demo input

See [../docs/ARTIFACTS.md](../docs/ARTIFACTS.md).

## Inputs

The launcher supports:

- video file
- webcam / laptop camera
- RTSP stream

If no input is passed, it defaults to:

- `test_data/data.mp4`

## Output

If you do not pass `-OutputDir`, output is written to:

- `output/demo_YYYYMMDD_HHMMSS`

That folder is local-only and is excluded from Git.

## Quick Start

### 1. Use the verified conda environment

Verified environment name:

- `paddle_det`

Versions:

- `environment/verified_versions.txt`

### 2. Run the default video demo

```powershell
.\deploy_bundle\run_demo.ps1
```

Or:

```bat
deploy_bundle\run_demo.bat
```

### 3. Run on webcam

```powershell
.\deploy_bundle\run_demo.ps1 -SourceType camera -CameraId 0
```

### 4. Run on RTSP

```powershell
.\deploy_bundle\run_demo.ps1 -SourceType rtsp -RtspUrl "rtsp://<user>:<password>@<camera-ip>/Streaming/Channels/101"
```

### 5. Dry-run only

```powershell
.\deploy_bundle\run_demo.ps1 -DryRun
```

This prints the resolved command without starting inference.

## Telegram Secrets

Telegram is disabled by default.

If you want Telegram alerts:

1. Copy `secrets/telegram.env.example` to `secrets/telegram.env`
2. Fill:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Run:

```powershell
.\deploy_bundle\run_demo.ps1 -UseTelegram
```

## Notes

- This bundle is intentionally separate from research artifacts and generated outputs.
- The deploy bundle now reuses artifacts from `output_inference/` so the public repo only needs one canonical model download location.
