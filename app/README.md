# App Web UI

This folder contains a self-contained web UI for the classroom monitoring demo.

The app is intentionally simple:

- FastAPI control plane
- Jinja2 single-page dashboard
- WebRTC preview delivered through MediaMTX
- one active inference session at a time
- no OpenCV preview window
- preview stays inside the same web page

## What It Controls

- video file inference
- laptop camera inference
- RTSP inference

The web UI reuses the existing runtime in `deploy/pipeline/pipeline_product.py`.
It patches the predictor output hook at runtime so the annotated frame is pushed to a local MediaMTX instance and then read back through WebRTC in the browser.

## Why This Design

Two practical browser delivery options were considered:

1. Browser preview through MediaMTX/WebRTC
   - lower end-to-end latency
   - H.264/WebRTC path is much cheaper for the browser than per-frame MJPEG
   - better fit for near-realtime operator preview

2. Browser preview directly from the app with MJPEG
   - simpler deployment
   - but noticeably heavier on CPU and browser repaint
   - higher latency once inference and overlay are enabled

This app now uses option 1 because low-latency preview matters more than eliminating the media service.

## Performance Notes

The app keeps preview overhead bounded by:

- no `cv2.imshow()` desktop window
- RTSP push queue size kept small in the pipeline
- MediaMTX WebRTC viewer embedded in the same UI
- `preview_local=False` forced in runtime so no `cv2.imshow()` window is opened

For the lowest latency:

- keep `save_video` off during live runs
- use a smaller RTSP sub-stream when available
- prefer camera or RTSP sub-streams over large main streams
- keep MediaMTX on the same machine as the app and pipeline
- focus on `Frame Age` in the UI, not only `Processing FPS`

The UI also hardens a few operational details:

- only full pipeline configs are shown in the config dropdown
- RTSP credentials are masked in the displayed source label
- log and alert rendering avoids direct HTML injection
- browser preview is isolated from the Python process

## Install

Install the extra web dependencies into the same environment that already runs the pipeline:

```powershell
pip install -r app/requirements.txt
```

Install MediaMTX and either:

- put `mediamtx.exe` on `PATH`, or
- set `MEDIAMTX_BIN` to the full path of `mediamtx.exe`

## Run

From the repository root:

```powershell
python -m app
```

Or from `cmd.exe`:

```bat
app\run_web.cmd
```

Default address:

- `http://127.0.0.1:8000`

The default bind address stays on loopback for safety. If you expose the app on
`0.0.0.0`, put it behind your own network controls or reverse proxy auth.

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

## Main Files

- `app/main.py`: FastAPI routes and page rendering
- `app/mediamtx.py`: MediaMTX process management and preview URL generation
- `app/runtime.py`: inference manager, runtime bridge, and session metrics
- `app/schemas.py`: request models
- `app/templates/index.html`: dashboard UI
- `app/static/app.css`: styling
- `app/static/app.js`: client-side actions and polling

## Limits

- one active session at a time
- browser preview depends on a local MediaMTX process
- RTSP preview latency still depends on source buffering, decode cost, and model throughput
- the app is intentionally single-worker because session state lives in memory inside one process
