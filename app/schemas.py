from typing import Literal, Optional

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    source_type: Literal["video", "camera", "rtsp"] = "video"
    video_path: Optional[str] = None
    camera_id: int = 0
    rtsp_url: Optional[str] = None
    config_path: Optional[str] = None
    device: Literal["CPU", "GPU"] = "GPU"
    run_mode: str = "paddle"
    enable_telegram: bool = False
    save_video: bool = False
    output_dir: Optional[str] = None
    mot_skip_frame_num: Optional[int] = Field(default=None, ge=0, le=16)
    cls_skip_frame_num: Optional[int] = Field(default=None, ge=0, le=16)
    mot_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cls_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    web_stream_fps: int = Field(default=15, ge=1, le=30)
    web_stream_width: int = Field(default=720, ge=320, le=1920)
    jpeg_quality: int = Field(default=70, ge=40, le=95)


class StopSessionResponse(BaseModel):
    ok: bool
    state: str
