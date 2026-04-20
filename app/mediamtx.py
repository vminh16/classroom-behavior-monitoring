from __future__ import annotations

import os
import shutil
import socket
import subprocess as sp
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import yaml


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


class MediaMTXManager:
    def __init__(self, app_root: Path):
        self._app_root = Path(app_root).resolve()
        self._runtime_dir = self._app_root / ".runtime" / "mediamtx"
        self._config_path = self._runtime_dir / "mediamtx.yml"
        self._host = os.getenv("MEDIAMTX_HOST", "127.0.0.1").strip() or "127.0.0.1"
        self._rtsp_port = _env_int("MEDIAMTX_RTSP_PORT", 8554)
        self._webrtc_port = _env_int("MEDIAMTX_WEBRTC_PORT", 8889)
        self._path_prefix = (
            os.getenv("MEDIAMTX_PATH_PREFIX", "webui").strip().strip("/") or
            "webui")
        self._binary_path = self._discover_binary()
        self._proc: Optional[sp.Popen] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._log_tail = []
        self._log_thread: Optional[threading.Thread] = None

    @property
    def path_prefix(self) -> str:
        return self._path_prefix

    @property
    def rtsp_base_url(self) -> str:
        return "rtsp://{}:{}/{}".format(self._host, self._rtsp_port,
                                         self._path_prefix)

    def whep_url(self, stream_path: str) -> str:
        return "http://{}:{}/{}/whep".format(
            self._host, self._webrtc_port, stream_path.strip("/"))

    def viewer_url(self, stream_path: str) -> str:
        return "http://{}:{}/{}".format(self._host, self._webrtc_port,
                                         stream_path.strip("/"))

    def status(self) -> Dict[str, object]:
        with self._lock:
            proc = self._proc
            binary_path = self._binary_path
            return {
                "available": binary_path is not None,
                "running": bool(proc is not None and proc.poll() is None),
                "binary_path": binary_path,
                "host": self._host,
                "rtsp_port": self._rtsp_port,
                "webrtc_port": self._webrtc_port,
                "path_prefix": self._path_prefix,
                "viewer_base_url": "http://{}:{}/".format(self._host,
                                                           self._webrtc_port),
                "rtsp_base_url": self.rtsp_base_url,
                "error": self._last_error,
                "logs": list(self._log_tail[-20:]),
            }

    def ensure_running(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            if self._binary_path is None:
                raise RuntimeError(
                    "MediaMTX was not found. Install mediamtx and add it to PATH, or set MEDIAMTX_BIN to mediamtx.exe."
                )
            self._runtime_dir.mkdir(parents=True, exist_ok=True)
            self._write_config()
            creationflags = getattr(sp, "CREATE_NO_WINDOW", 0)
            self._last_error = None
            self._log_tail = []
            self._proc = sp.Popen(
                [self._binary_path],
                cwd=str(self._runtime_dir),
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                stdin=sp.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
            self._log_thread = threading.Thread(
                target=self._drain_logs,
                name="mediamtx-log-drain",
                daemon=True)
            self._log_thread.start()

        if not self._wait_for_port(self._host, self._rtsp_port, timeout=6.0):
            self._terminate()
            raise RuntimeError(
                "MediaMTX did not open RTSP on {}:{}.".format(
                    self._host, self._rtsp_port))
        if not self._wait_for_port(self._host, self._webrtc_port, timeout=6.0):
            self._terminate()
            raise RuntimeError(
                "MediaMTX did not open WebRTC HTTP on {}:{}.".format(
                    self._host, self._webrtc_port))

    def shutdown(self) -> None:
        self._terminate()

    def _discover_binary(self) -> Optional[str]:
        env_path = os.getenv("MEDIAMTX_BIN")
        if env_path:
            candidate = Path(env_path).expanduser()
            if candidate.exists():
                return str(candidate.resolve())

        local_candidates = [
            self._app_root / "bin" / "mediamtx.exe",
            self._app_root / "bin" / "mediamtx" / "mediamtx.exe",
            self._app_root / "tools" / "mediamtx.exe",
        ]
        for candidate in local_candidates:
            if candidate.exists():
                return str(candidate.resolve())

        for name in ("mediamtx.exe", "mediamtx"):
            discovered = shutil.which(name)
            if discovered:
                return discovered
        return None

    def _write_config(self) -> None:
        config = {
            "logLevel": "warn",
            "logDestinations": ["stdout"],
            "api": False,
            "metrics": False,
            "pprof": False,
            "playback": False,
            "rtsp": True,
            "rtspTransports": ["tcp"],
            "rtspAddress": "{}:{}".format(self._host, self._rtsp_port),
            "hls": False,
            "rtmp": False,
            "srt": False,
            "webrtc": True,
            "webrtcAddress": "{}:{}".format(self._host, self._webrtc_port),
            "webrtcEncryption": False,
            "webrtcAllowOrigins": ["*"],
            "webrtcLocalUDPAddress": ":8189",
            "webrtcLocalTCPAddress": "",
            "webrtcIPsFromInterfaces": True,
            "webrtcAdditionalHosts": [self._host, "127.0.0.1", "localhost"],
            "paths": {
                "all_others": {}
            },
        }
        with open(self._config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    def _drain_logs(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw_line in proc.stdout:
            line = str(raw_line or "").strip()
            if not line:
                continue
            with self._lock:
                self._log_tail.append(line)
                if len(self._log_tail) > 120:
                    self._log_tail = self._log_tail[-120:]

    def _terminate(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except Exception:
                    pass

    def _wait_for_port(self, host: str, port: int,
                       timeout: float) -> bool:
        deadline = time.time() + max(0.1, float(timeout))
        while time.time() < deadline:
            with self._lock:
                if self._proc is not None and self._proc.poll() is not None:
                    self._last_error = (
                        "MediaMTX exited early with code {}.".format(
                            self._proc.returncode))
                    return False
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.3)
            try:
                if sock.connect_ex((host, port)) == 0:
                    return True
            except Exception:
                pass
            finally:
                sock.close()
            time.sleep(0.1)
        self._last_error = "Timed out waiting for {}:{}.".format(host, port)
        return False
