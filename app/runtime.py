from __future__ import annotations

import copy
import importlib
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import yaml

from .mediamtx import MediaMTXManager
from .schemas import StartSessionRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "deploy" / "pipeline"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
PIPELINE_CONFIG_DIR = PIPELINE_DIR / "config"
DEPLOY_BUNDLE_CONFIG_DIR = PROJECT_ROOT / "deploy_bundle" / "config"
DEFAULT_CONFIG_PATH = PIPELINE_CONFIG_DIR / "infer_cfg_pphuman.yml"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

_RUNTIME_IMPORT_LOCK = threading.Lock()
_RUNTIME_MODULES: Optional[Tuple[Any, Any]] = None


def _ensure_pipeline_import_path() -> None:
    pipeline_path = str(PIPELINE_DIR)
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)


def _load_runtime_modules() -> Tuple[Any, Any]:
    global _RUNTIME_MODULES
    with _RUNTIME_IMPORT_LOCK:
        if _RUNTIME_MODULES is not None:
            return _RUNTIME_MODULES

        _ensure_pipeline_import_path()
        cfg_utils = importlib.import_module("cfg_utils")
        pipeline_product = importlib.import_module("pipeline_product")
        _patch_pipeline_runtime(pipeline_product)
        _RUNTIME_MODULES = (cfg_utils, pipeline_product)
        return _RUNTIME_MODULES


def _is_supported_pipeline_config(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
    except Exception:
        return False
    if not isinstance(cfg, dict):
        return False
    return "visual" in cfg and "warmup_frame" in cfg and "MOT" in cfg


def _mask_rtsp_url(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    try:
        parsed = urlsplit(value)
    except Exception:
        return value
    if not parsed.scheme or "@" not in parsed.netloc:
        return value

    hostname = parsed.hostname or ""
    port_part = ""
    if parsed.port is not None:
        port_part = ":{}".format(parsed.port)
    username = parsed.username or "user"
    masked_netloc = "{}:***@{}{}".format(username, hostname, port_part)
    return urlunsplit(
        (parsed.scheme, masked_netloc, parsed.path, parsed.query,
         parsed.fragment))


def _safe_source_name(source: Any) -> str:
    if isinstance(source, int):
        return "camera_{}".format(source)

    if source is None:
        return "camera"

    text = str(source).strip()
    if not text:
        return "camera"

    scheme = urlsplit(text).scheme.lower()
    if scheme in {"rtsp", "rtsps", "rtmp", "rtmps"}:
        parsed = urlsplit(text)
        host = parsed.hostname or parsed.netloc or parsed.scheme or "stream"
        path = parsed.path.strip("/").replace("/", "_")
        pieces = [parsed.scheme or "stream", host]
        if path:
            pieces.append(path)
        base = "_".join(pieces)
    else:
        base = os.path.splitext(os.path.basename(text))[0]

    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
    return base[:96] if base else "camera"


def _patch_pipeline_runtime(pipeline_product: Any) -> None:
    if getattr(pipeline_product, "_web_ui_runtime_patched", False):
        return

    original_write = pipeline_product.PipePredictor._write_visual_output
    original_safe_print = pipeline_product._safe_print
    original_is_finished = pipeline_product.FrameSourceReader.is_finished

    def patched_write(self, image_bgr, writer, pushstream):
        callback = getattr(self, "_web_frame_callback", None)
        if callback is not None:
            should_continue = callback(self, image_bgr)
            if should_continue is False:
                return False
        return original_write(self, image_bgr, writer, pushstream)

    def patched_safe_print(*args, **kwargs):
        original_safe_print(*args, **kwargs)
        callback = getattr(pipeline_product, "_web_log_callback", None)
        if callback is not None:
            try:
                callback(" ".join(str(arg) for arg in args))
            except Exception:
                pass

    def patched_is_finished(self):
        callback = getattr(pipeline_product, "_web_stop_callback", None)
        if callback is not None:
            try:
                if callback():
                    return True
            except Exception:
                pass
        return original_is_finished(self)

    pipeline_product.PipePredictor._write_visual_output = patched_write
    pipeline_product._safe_print = patched_safe_print
    pipeline_product.FrameSourceReader.is_finished = patched_is_finished
    pipeline_product._web_ui_runtime_patched = True


class FrameBroker:
    def __init__(self):
        self._cond = threading.Condition()
        self._logs = deque(maxlen=300)
        self._alerts = deque(maxlen=100)
        self._alert_keys = deque(maxlen=100)
        self._video_seq = 0
        self._last_publish_wall_time = 0.0
        self._publish_times = deque(maxlen=240)
        self._stream_max_fps = 10.0
        self._stream_max_width = 960
        self._jpeg_quality = 80
        self._stop_requested = False
        self._state = {
            "session_state": "idle",
            "source_type": None,
            "source_label": None,
            "device": None,
            "config_path": None,
            "run_mode": None,
            "fps": 0.0,
            "inference_fps": 0.0,
            "preview_fps": 0.0,
            "preview_target_fps": 10.0,
            "avg_latency_ms": 0.0,
            "server_frame_age_ms": 0.0,
            "track_count": 0,
            "frame_width": None,
            "frame_height": None,
            "frame_index": 0,
            "preview_transport": "webrtc",
            "preview_stream_path": None,
            "preview_viewer_url": None,
            "preview_whep_url": None,
            "alerts_total": 0,
            "logs_total": 0,
            "error": None,
            "started_at": None,
            "finished_at": None,
            "updated_at": time.time(),
        }
        self._log_counter = 0
        self._alert_counter = 0

    def configure_stream(self, max_fps: int, max_width: int,
                         jpeg_quality: int) -> None:
        with self._cond:
            self._stream_max_fps = max(1.0, float(max_fps))
            self._stream_max_width = max(320, int(max_width))
            self._jpeg_quality = max(40, min(95, int(jpeg_quality)))
            self._state["preview_target_fps"] = round(self._stream_max_fps, 2)

    def reset_session(self, request: StartSessionRequest, source_label: str,
                      config_path: str) -> None:
        with self._cond:
            self._logs.clear()
            self._alerts.clear()
            self._alert_keys.clear()
            self._video_seq = 0
            self._stop_requested = False
            self._last_publish_wall_time = 0.0
            self._publish_times.clear()
            now = time.time()
            self._state.update({
                "session_state": "starting",
                "source_type": request.source_type,
                "source_label": source_label,
                "device": request.device,
                "config_path": config_path,
                "run_mode": request.run_mode,
                "fps": 0.0,
                "inference_fps": 0.0,
                "preview_fps": 0.0,
                "preview_target_fps": round(self._stream_max_fps, 2),
                "avg_latency_ms": 0.0,
                "server_frame_age_ms": 0.0,
                "track_count": 0,
                "frame_width": None,
                "frame_height": None,
                "frame_index": 0,
                "preview_transport": "webrtc",
                "preview_stream_path": None,
                "preview_viewer_url": None,
                "preview_whep_url": None,
                "alerts_total": 0,
                "logs_total": 0,
                "error": None,
                "started_at": now,
                "finished_at": None,
                "updated_at": now,
            })
            self._cond.notify_all()

    def request_stop(self) -> None:
        with self._cond:
            self._stop_requested = True
            self._state["session_state"] = "stopping"
            self._state["updated_at"] = time.time()
            self._cond.notify_all()

    def stop_requested(self) -> bool:
        with self._cond:
            return self._stop_requested

    def set_session_state(self, state: str, error: Optional[str] = None) -> None:
        with self._cond:
            self._state["session_state"] = state
            self._state["error"] = error
            if state in {"completed", "stopped", "error"}:
                self._state["finished_at"] = time.time()
            self._state["updated_at"] = time.time()
            self._cond.notify_all()

    def update_metrics(self, **kwargs: Any) -> None:
        with self._cond:
            self._state.update(kwargs)
            self._state["updated_at"] = time.time()

    def publish_frame(self, _image_bgr: np.ndarray,
                      metadata: Dict[str, Any]) -> bool:
        now = time.monotonic()
        with self._cond:
            self._video_seq += 1
            preview_fps = 0.0
            self._publish_times.append(now)
            while len(self._publish_times) > 1 and (
                    now - self._publish_times[0]) > 1.5:
                self._publish_times.popleft()
            if len(self._publish_times) > 1:
                elapsed = max(now - self._publish_times[0], 1e-3)
                preview_fps = (len(self._publish_times) - 1) / elapsed
            self._last_publish_wall_time = now
            self._state.update(metadata)
            self._state["preview_fps"] = round(preview_fps, 2)
            self._state["preview_target_fps"] = round(self._stream_max_fps, 2)
            self._cond.notify_all()
            return not self._stop_requested

    def close(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def log(self, message: str, level: str = "info") -> None:
        text = str(message or "").strip()
        if not text:
            return
        with self._cond:
            self._log_counter += 1
            self._logs.append({
                "id": self._log_counter,
                "ts": time.time(),
                "level": level,
                "message": text,
            })
            self._state["logs_total"] = self._log_counter
            self._state["updated_at"] = time.time()
            self._cond.notify_all()

    def add_alert(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        key = (
            event.get("track_id"),
            event.get("event_type"),
            event.get("alert_level"),
            event.get("event_kind"),
            round(float(event.get("trigger_time", 0.0)), 1),
        )
        with self._cond:
            if key in self._alert_keys:
                return
            self._alert_keys.append(key)
            self._alert_counter += 1
            self._alerts.appendleft({
                "id": self._alert_counter,
                "ts": time.time(),
                "track_id": event.get("track_id"),
                "event_type": event.get("event_type"),
                "alert_level": event.get("alert_level"),
                "event_kind": event.get("event_kind"),
                "duration_at_trigger": event.get("duration_at_trigger"),
                "display_text": event.get("display_text"),
                "state": event.get("state"),
            })
            self._state["alerts_total"] = self._alert_counter
            self._state["updated_at"] = time.time()
            self._cond.notify_all()

    def snapshot(self) -> Dict[str, Any]:
        with self._cond:
            return {
                "state": copy.deepcopy(self._state),
                "logs": list(self._logs),
                "alerts": list(self._alerts),
                "frame_seq": self._video_seq,
            }

@dataclass
class ActiveSession:
    request: StartSessionRequest
    thread: threading.Thread
    source_label: str


class InferenceManager:
    def __init__(self, media_server: Optional[MediaMTXManager] = None):
        self._lock = threading.Lock()
        self._broker = FrameBroker()
        self._session: Optional[ActiveSession] = None
        self._media_server = media_server or MediaMTXManager(PROJECT_ROOT /
                                                             "app")

    @property
    def broker(self) -> FrameBroker:
        return self._broker

    def get_options(self) -> Dict[str, Any]:
        return {
            "video_files": self._discover_video_files(),
            "config_files": self._discover_config_files(),
            "default_config": str(DEFAULT_CONFIG_PATH),
            "default_video": self._default_video_path(),
            "device_options": ["GPU", "CPU"],
            "run_mode_options": ["paddle", "trt_fp32", "trt_fp16", "trt_int8"],
            "media_server": self._media_server.status(),
        }

    def start(self, request: StartSessionRequest) -> Dict[str, Any]:
        source_label = self._resolve_source_label(request)
        config_path = self._resolve_config_path(request.config_path)
        preview_stream_path = self._preview_stream_path(request)
        self._validate_request(request, config_path)
        self._media_server.ensure_running()

        with self._lock:
            if self._session and not self._session.thread.is_alive():
                self._session = None
            if self._session and self._session.thread.is_alive():
                raise RuntimeError("Another inference session is already running.")

            self._broker.configure_stream(request.web_stream_fps,
                                          request.web_stream_width,
                                          request.jpeg_quality)
            self._broker.reset_session(request, source_label, config_path)
            self._broker.update_metrics(
                preview_transport="webrtc",
                preview_stream_path=preview_stream_path,
                preview_viewer_url=self._media_server.viewer_url(
                    preview_stream_path),
                preview_whep_url=self._media_server.whep_url(
                    preview_stream_path),
            )
            self._broker.log(
                "Starting session: {} ({})".format(source_label,
                                                   request.source_type))
            self._broker.log(
                "Preview transport: WebRTC via {}".format(
                    self._media_server.viewer_url(preview_stream_path)))

            thread = threading.Thread(
                target=self._run_session,
                args=(request, config_path, source_label),
                name="web-ui-inference",
                daemon=True)
            self._session = ActiveSession(
                request=request, thread=thread, source_label=source_label)
            thread.start()

        return self.snapshot()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            session = self._session
        if session is None:
            return self.snapshot()

        self._broker.log("Stop requested by user.", level="warning")
        self._broker.request_stop()
        session.thread.join(timeout=5.0)
        if session.thread.is_alive():
            self._broker.log(
                "Pipeline is still shutting down; waiting for the next loop boundary.",
                level="warning")
        else:
            with self._lock:
                if self._session and self._session.thread is session.thread:
                    self._session = None
        return self.snapshot()

    def shutdown(self) -> None:
        self.stop()
        self._broker.close()
        self._media_server.shutdown()

    def snapshot(self) -> Dict[str, Any]:
        snapshot = self._broker.snapshot()
        snapshot["media"] = self._media_server.status()
        return snapshot

    def _run_session(self, request: StartSessionRequest, config_path: str,
                     source_label: str) -> None:
        cfg_utils = None
        pipeline_product = None
        predictor = None
        try:
            cfg_utils, pipeline_product = _load_runtime_modules()
            pipeline_product._web_log_callback = self._broker.log
            pipeline_product._web_stop_callback = self._broker.stop_requested

            argv = self._build_pipeline_argv(request, config_path)
            parser = cfg_utils.argsparser()
            args = parser.parse_args(argv)
            cfg = cfg_utils.merge_cfg(args)

            pipeline = pipeline_product.Pipeline(args, cfg)
            if getattr(pipeline, "multi_camera", False):
                raise RuntimeError(
                    "The web UI currently supports exactly one source at a time."
                )

            predictor = pipeline.predictor
            predictor._web_frame_callback = self._make_frame_callback(
                request, source_label, config_path)

            self._broker.set_session_state("running")
            self._broker.log("Pipeline initialized successfully.")

            pipeline.run_multithreads()

            if self._broker.stop_requested():
                self._broker.set_session_state("stopped")
                self._broker.log("Session stopped.")
            else:
                self._broker.set_session_state("completed")
                self._broker.log("Session completed.")
        except Exception as exc:
            self._broker.log(str(exc), level="error")
            self._broker.log(traceback.format_exc(), level="error")
            self._broker.set_session_state("error", error=str(exc))
        finally:
            if predictor is not None:
                predictor._web_frame_callback = None
            if pipeline_product is not None:
                pipeline_product._web_log_callback = None
                pipeline_product._web_stop_callback = None
            current_thread = threading.current_thread()
            with self._lock:
                if self._session and self._session.thread is current_thread:
                    self._session = None

    def _make_frame_callback(self, request: StartSessionRequest,
                             source_label: str,
                             config_path: str) -> Callable[[Any, np.ndarray],
                                                           bool]:

        def callback(predictor: Any, image_bgr: np.ndarray) -> bool:
            mot_res = predictor.pipeline_res.get("mot")
            track_count = 0
            if isinstance(mot_res, dict):
                boxes = mot_res.get("boxes")
                if boxes is not None:
                    track_count = len(boxes)

            _, average_latency, qps = predictor.pipe_timer.get_total_time()
            frame_index = int(getattr(predictor.pipe_timer, "img_num", 0))
            capture_time = getattr(predictor, "_web_last_capture_time", None)
            frame_age_ms = 0.0
            if capture_time is not None:
                frame_age_ms = max(
                    0.0, (time.monotonic() - float(capture_time)) * 1000.0)

            self._broker.update_metrics(
                session_state="running",
                source_type=request.source_type,
                source_label=source_label,
                device=request.device,
                config_path=config_path,
                run_mode=request.run_mode,
                fps=round(float(qps), 2),
                inference_fps=round(float(qps), 2),
                avg_latency_ms=round(float(average_latency) * 1000.0, 2),
                server_frame_age_ms=round(frame_age_ms, 2),
                track_count=track_count,
                frame_width=int(image_bgr.shape[1]),
                frame_height=int(image_bgr.shape[0]),
                frame_index=frame_index,
            )

            latest_events = getattr(predictor, "latest_behavior_events", [])
            for event in latest_events:
                self._broker.add_alert(event)

            return self._broker.publish_frame(image_bgr, {})

        return callback

    def _build_pipeline_argv(self, request: StartSessionRequest,
                             config_path: str) -> List[str]:
        argv = [
            "--config",
            config_path,
            "--device",
            request.device,
            "--run_mode",
            request.run_mode,
            "--output_dir",
            self._resolve_output_dir(request),
            "--pushurl",
            self._media_server.rtsp_base_url,
        ]

        if request.source_type == "video":
            argv.extend(["--video_file", self._resolve_video_path(request)])
        elif request.source_type == "camera":
            argv.extend(["--camera_id", str(request.camera_id)])
        else:
            argv.extend(["--rtsp", str(request.rtsp_url).strip()])

        opt_args = [
            "visual=True",
            "visual_style=minimal",
            "preview_local=False",
            "save_visual_output={}".format("True"
                                           if request.save_video else "False"),
            "TELEGRAM_ALERT.enable={}".format("True"
                                              if request.enable_telegram else
                                              "False"),
        ]
        if request.mot_skip_frame_num is not None:
            opt_args.append("MOT.skip_frame_num={}".format(
                int(request.mot_skip_frame_num)))
        if request.cls_skip_frame_num is not None:
            opt_args.append("ID_BASED_CLSACTION.skip_frame_num={}".format(
                int(request.cls_skip_frame_num)))
        if request.mot_threshold is not None:
            opt_args.append("MOT.threshold={:.4f}".format(
                float(request.mot_threshold)))
        if request.cls_threshold is not None:
            opt_args.append("ID_BASED_CLSACTION.threshold={:.4f}".format(
                float(request.cls_threshold)))

        argv.append("--opt")
        argv.extend(opt_args)
        return argv

    def _preview_stream_path(self, request: StartSessionRequest) -> str:
        if request.source_type == "video":
            source = self._resolve_video_path(request)
        elif request.source_type == "camera":
            source = int(request.camera_id)
        else:
            source = str(request.rtsp_url or "").strip()
        return "{}/{}".format(self._media_server.path_prefix,
                               _safe_source_name(source))

    def _validate_request(self, request: StartSessionRequest,
                          config_path: str) -> None:
        if not Path(config_path).exists():
            raise FileNotFoundError("Config not found: {}".format(config_path))
        if not _is_supported_pipeline_config(Path(config_path)):
            raise ValueError(
                "Unsupported config for the web UI: {}. Use a full pipeline config like infer_cfg_pphuman.yml or demo_final.yml."
                .format(config_path))

        if request.source_type == "video":
            video_path = Path(self._resolve_video_path(request))
            if not video_path.exists():
                raise FileNotFoundError("Video file not found: {}".format(
                    video_path))
        elif request.source_type == "rtsp":
            if not request.rtsp_url or not str(request.rtsp_url).strip():
                raise ValueError("RTSP URL is required for RTSP inference.")

    def _resolve_output_dir(self, request: StartSessionRequest) -> str:
        if request.output_dir:
            return str(Path(request.output_dir).expanduser().resolve())
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return str((PROJECT_ROOT / "output" / "webui" / stamp).resolve())

    def _resolve_source_label(self, request: StartSessionRequest) -> str:
        if request.source_type == "video":
            return Path(self._resolve_video_path(request)).name
        if request.source_type == "camera":
            return "camera_{}".format(int(request.camera_id))
        return _mask_rtsp_url(str(request.rtsp_url or "rtsp"))

    def _resolve_video_path(self, request: StartSessionRequest) -> str:
        text = str(request.video_path or "").strip()
        if text:
            return str(Path(text).expanduser().resolve())
        default_video = self._default_video_path()
        if default_video:
            return default_video
        raise FileNotFoundError("No video file provided and no default demo video found.")

    def _resolve_config_path(self, config_path: Optional[str]) -> str:
        text = str(config_path or "").strip()
        if text:
            return str(Path(text).expanduser().resolve())
        return str(DEFAULT_CONFIG_PATH.resolve())

    def _discover_video_files(self) -> List[str]:
        if not TEST_DATA_DIR.exists():
            return []
        files = []
        for path in sorted(TEST_DATA_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(str(path.resolve()))
        return files

    def _default_video_path(self) -> Optional[str]:
        for candidate in self._discover_video_files():
            if Path(candidate).name.lower() == "data.mp4":
                return candidate
        files = self._discover_video_files()
        return files[0] if files else None

    def _discover_config_files(self) -> List[str]:
        results = []
        for folder in (PIPELINE_CONFIG_DIR, DEPLOY_BUNDLE_CONFIG_DIR):
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.yml")):
                if _is_supported_pipeline_config(path):
                    results.append(str(path.resolve()))
        return results
