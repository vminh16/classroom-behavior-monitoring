# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import yaml
import cv2
import numpy as np
import math
import paddle
import sys
import copy
import threading
import queue
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from urllib.parse import urlparse
from datacollector import DataCollector, Result

# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from cfg_utils import argsparser, print_arguments, merge_cfg
from pipe_utils import PipeTimer
from pipe_utils import get_test_images, crop_image_with_det, crop_image_with_mot, parse_mot_res, parse_mot_keypoint, parse_mot_keypoint_relative, normal_crop
from pipe_utils import PushStream
from minimal_visualize import draw_minimal_video_result

from python.infer import Detector
from python.keypoint_infer import KeyPointDetector
from python.keypoint_postprocess import translate_to_ori_images
from python.keypoint_preprocess import expand_crop
from python.preprocess import decode_image, ShortSizeScale
from python.visualize import visualize_box_mask, visualize_attr, visualize_pose, visualize_action, localize_label
from behavior_filter import BehaviorFilter
from behavior_state_machine import BehaviorStateMachine
from telegram_alert import TelegramAlertSink
from async_cls_action import AsyncClsActionWorker, ClsActionPostProcessor

from pptracking.python.mot_sde_infer import SDE_Detector
from pptracking.python.mot.visualize import plot_tracking_dict
from pptracking.python.mot.utils import flow_statistic

from pphuman.attr_infer import AttrDetector
from pphuman.video_action_infer import VideoActionRecognizer
from pphuman.action_infer import SkeletonActionRecognizer, DetActionRecognizer, ClsActionRecognizer
from pphuman.action_utils import KeyPointBuff, ActionVisualHelper
try:
    from pphuman.reid import ReID
    from pphuman.mtmct import mtmct_process
    _PPHUMAN_OPTIONAL_IMPORT_ERROR = None
except Exception as exc:
    ReID = None
    mtmct_process = None
    _PPHUMAN_OPTIONAL_IMPORT_ERROR = exc

from download import auto_download_model


def _require_dependency(feature_name, *deps, import_error=None):
    if all(dep is not None for dep in deps):
        return
    msg = "{} is enabled, but its runtime dependency is unavailable.".format(
        feature_name)
    if import_error is not None:
        msg = "{} Original import error: {}".format(msg, import_error)
    raise ImportError(msg)


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except (OSError, ValueError):
        pass


def _raise_if_vehicle_pipeline_enabled(args, cfg):
    removed_sections = {
        'VEHICLE_PLATE': True,  # legacy plate configs may omit `enable`
        'VEHICLE_ATTR': False,
        'VEHICLE_PRESSING': False,
        'VEHICLE_RETROGRADE': False,
    }
    enabled_sections = []
    for key, enable_default in removed_sections.items():
        section = cfg.get(key)
        if isinstance(section, dict) and section.get('enable', enable_default):
            enabled_sections.append(key)

    if enabled_sections:
        raise ValueError(
            "Vehicle pipeline has been removed from this project. Disable these sections in the config: {}"
            .format(", ".join(enabled_sections)))

    if getattr(args, 'illegal_parking_time', -1) != -1:
        raise ValueError(
            "--illegal_parking_time is no longer supported because the vehicle pipeline has been removed."
        )


def _source_scheme(source):
    if not isinstance(source, str):
        return ""
    try:
        return urlparse(source).scheme.strip().lower()
    except Exception:
        return ""


def _is_live_stream_source(source):
    if isinstance(source, int):
        return True
    return _source_scheme(source) in {"rtsp", "rtsps", "rtmp", "rtmps"}


def _is_reconnectable_live_source(source):
    return isinstance(source, str) and _is_live_stream_source(source)


def _safe_source_name(source):
    if isinstance(source, int):
        return "camera_{}".format(source)

    if source is None:
        return "camera"

    text = str(source).strip()
    if not text:
        return "camera"

    if _is_live_stream_source(text):
        parsed = urlparse(text)
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


class _NullCollector(object):
    def append(self, frameid, result):
        return

    def get_res(self):
        return {}


@dataclass
class FramePacket:
    frame_rgb: np.ndarray
    frame_index: int
    capture_time: float


class FrameSourceReader(object):
    def __init__(self,
                 source,
                 capture,
                 open_capture_fn,
                 is_live_source=False,
                 is_reconnectable_source=False,
                 file_queue_size=30,
                 live_queue_size=1,
                 startup_timeout=5.0,
                 live_empty_reads_before_retry=30,
                 live_retry_delay_seconds=0.1):
        self.source = source
        self.capture = capture
        self.open_capture_fn = open_capture_fn
        self.is_live_source = bool(is_live_source)
        self.is_reconnectable_source = bool(is_reconnectable_source)
        self.startup_timeout = float(startup_timeout)
        self.live_empty_reads_before_retry = max(
            1, int(live_empty_reads_before_retry))
        self.live_retry_delay_seconds = float(live_retry_delay_seconds)
        self.queue = queue.Queue(
            maxsize=live_queue_size if self.is_live_source else file_queue_size)
        self.stop_event = threading.Event()
        self.first_frame_event = threading.Event()
        self.done_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.frame_index = 0
        self.error = None
        self.reconnect_count = 0

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self._release_capture()

    def join(self, timeout=None):
        self.thread.join(timeout=timeout)

    def get(self, timeout=0.1):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def empty(self):
        return self.queue.empty()

    def is_finished(self):
        return self.done_event.is_set()

    def wait_for_first_frame(self):
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.first_frame_event.is_set():
                return True
            if self.done_event.is_set() and self.queue.empty():
                return False
            time.sleep(0.05)
        return self.first_frame_event.is_set()

    def _run(self):
        empty_reads = 0
        try:
            while not self.stop_event.is_set():
                if self.capture is None or not self.capture.isOpened():
                    if self.is_live_source and self.is_reconnectable_source:
                        if not self._reopen_capture():
                            time.sleep(self.live_retry_delay_seconds)
                            continue
                        empty_reads = 0
                        continue
                    self.error = RuntimeError(
                        "Unable to open input source: {}".format(self.source))
                    break

                ret, frame = self.capture.read()
                capture_time = time.monotonic()
                if ret and frame is not None:
                    empty_reads = 0
                    packet = FramePacket(
                        frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        frame_index=self.frame_index,
                        capture_time=capture_time)
                    self.frame_index += 1
                    self._enqueue(packet)
                    self.first_frame_event.set()
                    continue

                if not self.is_live_source:
                    break

                empty_reads += 1
                if self.is_reconnectable_source and empty_reads >= self.live_empty_reads_before_retry:
                    empty_reads = 0
                    self._release_capture()
                    time.sleep(self.live_retry_delay_seconds)
                    continue

                if empty_reads >= self.live_empty_reads_before_retry:
                    empty_reads = 0
                time.sleep(self.live_retry_delay_seconds)
        finally:
            self.done_event.set()
            self._release_capture()

    def _enqueue(self, packet):
        if self.is_live_source:
            while not self.stop_event.is_set():
                try:
                    self.queue.put(packet, timeout=0.05)
                    return
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
            return

        while not self.stop_event.is_set():
            try:
                self.queue.put(packet, timeout=0.1)
                return
            except queue.Full:
                continue

    def _reopen_capture(self):
        if self.stop_event.is_set():
            return False
        self._release_capture()
        capture = self.open_capture_fn(self.source)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            return False
        self.capture = capture
        self.reconnect_count += 1
        _safe_print(
            "Reconnected input source {} (count={})".format(
                self.source, self.reconnect_count))
        return True

    def _release_capture(self):
        if self.capture is None:
            return
        try:
            self.capture.release()
        except Exception:
            pass
        self.capture = None


class Pipeline(object):
    """
    Pipeline

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
    """

    def __init__(self, args, cfg):
        _raise_if_vehicle_pipeline_enabled(args, cfg)
        self.multi_camera = False
        reid_cfg = cfg.get('REID', False)
        self.enable_mtmct = reid_cfg['enable'] if reid_cfg else False
        self.is_video = False
        self.output_dir = args.output_dir
        self.vis_result = cfg['visual']
        self.input = self._parse_input(args.image_file, args.image_dir,
                                       args.video_file, args.video_dir,
                                       args.camera_id, args.rtsp)
        if self.multi_camera:
            self.predictor = []
            for name in self.input:
                predictor_item = PipePredictor(
                    args, cfg, is_video=True, multi_camera=True)
                predictor_item.set_file_name(name)
                self.predictor.append(predictor_item)

        else:
            self.predictor = PipePredictor(args, cfg, self.is_video)
            if self.is_video:
                self.predictor.set_file_name(self.input)

    def _parse_input(self, image_file, image_dir, video_file, video_dir,
                     camera_id, rtsp):

        # parse input as is_video and multi_camera

        if image_file is not None or image_dir is not None:
            input = get_test_images(image_dir, image_file)
            self.is_video = False
            self.multi_camera = False

        elif video_file and len(video_file) > 0:
            assert os.path.exists(
                video_file
            ) or _is_live_stream_source(
                video_file
            ), "video_file not exists and is not a supported live stream url."
            self.multi_camera = False
            input = video_file
            self.is_video = True

        elif video_dir is not None:
            videof = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
            if len(videof) > 1:
                self.multi_camera = True
                videof.sort()
                input = videof
            else:
                input = videof[0]
            self.is_video = True

        elif rtsp is not None:
            rtsp = [
                rtsp_item for rtsp_item in rtsp
                if _is_live_stream_source(rtsp_item)
            ]
            if len(rtsp) == 0:
                raise ValueError(
                    "--rtsp only supports rtsp://, rtsps://, rtmp:// or rtmps:// inputs."
                )
            self.multi_camera = len(rtsp) > 1
            input = rtsp if self.multi_camera else rtsp[0]
            self.is_video = True

        elif camera_id != -1:
            self.multi_camera = False
            input = camera_id
            self.is_video = True

        else:
            raise ValueError(
                "Illegal Input, please set one of ['video_file', 'camera_id', 'image_file', 'image_dir']"
            )

        return input

    def run_multithreads(self):
        if self.multi_camera:
            multi_res = []
            threads = []
            for idx, (predictor,
                      input) in enumerate(zip(self.predictor, self.input)):
                thread = threading.Thread(
                    name=str(idx).zfill(3),
                    target=predictor.run,
                    args=(input, idx))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for predictor, thread in zip(self.predictor, threads):
                thread.join()
                collector_data = predictor.get_result()
                multi_res.append(collector_data)

            if self.enable_mtmct:
                _require_dependency(
                    "MTMCT", mtmct_process, import_error=_PPHUMAN_OPTIONAL_IMPORT_ERROR)
                mtmct_process(
                    multi_res,
                    self.input,
                    mtmct_vis=self.vis_result,
                    output_dir=self.output_dir)

        else:
            self.predictor.run(self.input)

    def run(self):
        if self.multi_camera:
            multi_res = []
            for predictor, input in zip(self.predictor, self.input):
                predictor.run(input)
                collector_data = predictor.get_result()
                multi_res.append(collector_data)
            if self.enable_mtmct:
                _require_dependency(
                    "MTMCT", mtmct_process, import_error=_PPHUMAN_OPTIONAL_IMPORT_ERROR)
                mtmct_process(
                    multi_res,
                    self.input,
                    mtmct_vis=self.vis_result,
                    output_dir=self.output_dir)

        else:
            self.predictor.run(self.input)


def get_model_dir(cfg):
    """ 
        Auto download inference model if the model_path is a url link. 
        Otherwise it will use the model_path directly.
    """
    for key in cfg.keys():
        if type(cfg[key]) ==  dict and \
            ("enable" in cfg[key].keys() and cfg[key]['enable']
                or "enable" not in cfg[key].keys()):

            if "model_dir" in cfg[key].keys():
                model_dir = cfg[key]["model_dir"]
                downloaded_model_dir = auto_download_model(model_dir)
                if downloaded_model_dir:
                    model_dir = downloaded_model_dir
                    cfg[key]["model_dir"] = model_dir
                print(key, " model dir: ", model_dir)

        elif key == "MOT":  # for idbased and skeletonbased actions
            model_dir = cfg[key]["model_dir"]
            downloaded_model_dir = auto_download_model(model_dir)
            if downloaded_model_dir:
                model_dir = downloaded_model_dir
                cfg[key]["model_dir"] = model_dir
            print("mot_model_dir model_dir: ", model_dir)


class PipePredictor(object):
    """
    Predictor in single camera
    
    The pipeline for image input: 

        1. Detection
        2. Detection -> Attribute

    The pipeline for video input: 

        1. Tracking
        2. Tracking -> Attribute
        3. Tracking -> KeyPoint -> SkeletonAction Recognition
        4. VideoAction Recognition

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
        is_video (bool): whether the input is video, default as False
        multi_camera (bool): whether to use multi camera in pipeline, 
            default as False
    """

    def __init__(self, args, cfg, is_video=True, multi_camera=False):
        _raise_if_vehicle_pipeline_enabled(args, cfg)
        # general module for pphuman
        self.with_mot = cfg.get('MOT', False)['enable'] if cfg.get(
            'MOT', False) else False
        self.with_human_attr = cfg.get('ATTR', False)['enable'] if cfg.get(
            'ATTR', False) else False
        if self.with_mot:
            print('Multi-Object Tracking enabled')
        if self.with_human_attr:
            print('Human Attribute Recognition enabled')

        # only for pphuman
        self.with_skeleton_action = cfg.get(
            'SKELETON_ACTION', False)['enable'] if cfg.get('SKELETON_ACTION',
                                                           False) else False
        self.with_video_action = cfg.get(
            'VIDEO_ACTION', False)['enable'] if cfg.get('VIDEO_ACTION',
                                                        False) else False
        self.with_idbased_detaction = cfg.get(
            'ID_BASED_DETACTION', False)['enable'] if cfg.get(
                'ID_BASED_DETACTION', False) else False
        self.with_idbased_clsaction = cfg.get(
            'ID_BASED_CLSACTION', False)['enable'] if cfg.get(
                'ID_BASED_CLSACTION', False) else False
        self.with_mtmct = cfg.get('REID', False)['enable'] if cfg.get(
            'REID', False) else False
        
        # Keypoint only mode (without STGCN)
        self.with_kpt_only = cfg.get('KPT', False).get('enable', False) if cfg.get('KPT', False) else False

        if self.with_skeleton_action:
            print('SkeletonAction Recognition enabled')
        if self.with_kpt_only and not self.with_skeleton_action:
            print('Keypoint Detection enabled (without action recognition)')
        if self.with_video_action:
            print('VideoAction Recognition enabled')
        if self.with_idbased_detaction:
            print('IDBASED Detection Action Recognition enabled')
        if self.with_idbased_clsaction:
            print('IDBASED Classification Action Recognition enabled')
        if self.with_mtmct:
            print("MTMCT enabled")

        self.modebase = {
            "framebased": False,
            "videobased": False,
            "idbased": False,
            "skeletonbased": False
        }

        self.basemode = {
            "MOT": "idbased",
            "ATTR": "idbased",
            "VIDEO_ACTION": "videobased",
            "SKELETON_ACTION": "skeletonbased",
            "ID_BASED_DETACTION": "idbased",
            "ID_BASED_CLSACTION": "idbased",
            "REID": "idbased",
        }

        self.is_video = is_video
        self.multi_camera = multi_camera
        self.cfg = cfg

        self.output_dir = args.output_dir
        self.draw_center_traj = args.draw_center_traj
        self.secs_interval = args.secs_interval
        self.do_entrance_counting = args.do_entrance_counting
        self.do_break_in_counting = args.do_break_in_counting
        self.region_type = args.region_type
        self.region_polygon = args.region_polygon

        self.warmup_frame = self.cfg['warmup_frame']
        self.pipeline_res = Result()
        self.pipe_timer = PipeTimer()
        self.file_name = None
        self.collector = DataCollector() if self.with_mtmct else _NullCollector(
        )
        self.latest_behavior_events = []
        self.behavior_event_history = deque(maxlen=512)
        self.telegram_alert = None
        self.cls_action_async_worker = None
        self.cls_action_postprocessor = None
        self._async_cls_cache = {}
        self._async_cls_last_frame_id = -1
        self._async_cls_frame_buffer = OrderedDict()
        self._async_cls_frame_buffer_size = 0
        self._show_local_preview = False
        self._save_visual_output = bool(self.cfg.get('visual', False))
        self.preview_window_name = str(
            self.cfg.get('preview_window_name', 'Paddle-Pipeline'))
        self.preview_max_fps = max(
            0.0, float(self.cfg.get('preview_max_fps', 0.0) or 0.0))
        self.preview_max_width = max(
            0, int(self.cfg.get('preview_max_width', 0) or 0))
        self._last_preview_wall_time = 0.0

        self.pushurl = args.pushurl

        # auto download inference model
        get_model_dir(self.cfg)

        if self.with_human_attr:
            attr_cfg = self.cfg['ATTR']
            basemode = self.basemode['ATTR']
            self.modebase[basemode] = True
            self.attr_predictor = AttrDetector.init_with_cfg(args, attr_cfg)

        if not is_video:

            det_cfg = self.cfg['DET']
            model_dir = det_cfg['model_dir']
            batch_size = det_cfg['batch_size']
            self.det_predictor = Detector(
                model_dir, args.device, args.run_mode, batch_size,
                args.trt_min_shape, args.trt_max_shape, args.trt_opt_shape,
                args.trt_calib_mode, args.cpu_threads, args.enable_mkldnn)
        else:
            if self.with_idbased_detaction:
                idbased_detaction_cfg = self.cfg['ID_BASED_DETACTION']
                basemode = self.basemode['ID_BASED_DETACTION']
                self.modebase[basemode] = True

                self.det_action_predictor = DetActionRecognizer.init_with_cfg(
                    args, idbased_detaction_cfg)
                self.det_action_visual_helper = ActionVisualHelper(1)

            if self.with_idbased_clsaction:
                idbased_clsaction_cfg = self.cfg['ID_BASED_CLSACTION']
                basemode = self.basemode['ID_BASED_CLSACTION']
                self.modebase[basemode] = True

                self.cls_action_visual_helper = ActionVisualHelper(1)
                if idbased_clsaction_cfg.get('async_worker', {}).get(
                        'enable', False):
                    self.cls_action_async_worker = AsyncClsActionWorker.init_with_cfg(
                        args, idbased_clsaction_cfg)
                    self.cls_action_predictor = self.cls_action_async_worker.predictor
                    self.clsaction_labels = self._get_cls_action_labels()
                    self._async_cls_frame_buffer_size = max(
                        4,
                        int(
                            idbased_clsaction_cfg.get('async_worker',
                                                      {}).get(
                                                          'frame_buffer_size',
                                                          32) or 32))
                    print(
                        'Async cls_action worker enabled on {} ({})'.format(
                            self.cls_action_async_worker.worker_device,
                            self.cls_action_async_worker.worker_run_mode))
                else:
                    self.cls_action_predictor = ClsActionRecognizer.init_with_cfg(
                        args, idbased_clsaction_cfg)
                    self.clsaction_labels = self._get_cls_action_labels()
                    self.cls_action_postprocessor = self._build_cls_action_postprocessor(
                        idbased_clsaction_cfg)

                telegram_cfg = self.cfg.get('TELEGRAM_ALERT', {})
                if telegram_cfg.get('enable', False):
                    self.telegram_alert = TelegramAlertSink.from_cfg(
                        telegram_cfg)
                    print('Telegram alert enabled')

            if self.with_skeleton_action:
                skeleton_action_cfg = self.cfg['SKELETON_ACTION']
                display_frames = skeleton_action_cfg['display_frames']
                self.coord_size = skeleton_action_cfg['coord_size']
                basemode = self.basemode['SKELETON_ACTION']
                self.modebase[basemode] = True
                skeleton_action_frames = skeleton_action_cfg['max_frames']

                self.skeleton_action_predictor = SkeletonActionRecognizer.init_with_cfg(
                    args, skeleton_action_cfg)
                self.skeleton_action_visual_helper = ActionVisualHelper(
                    display_frames)

                kpt_cfg = self.cfg['KPT']
                kpt_model_dir = kpt_cfg['model_dir']
                kpt_batch_size = kpt_cfg['batch_size']
                self.kpt_predictor = KeyPointDetector(
                    kpt_model_dir,
                    args.device,
                    args.run_mode,
                    kpt_batch_size,
                    args.trt_min_shape,
                    args.trt_max_shape,
                    args.trt_opt_shape,
                    args.trt_calib_mode,
                    args.cpu_threads,
                    args.enable_mkldnn,
                    use_dark=False)
                self.kpt_buff = KeyPointBuff(skeleton_action_frames)
            
            # Keypoint only mode (without STGCN)
            elif self.with_kpt_only:
                kpt_cfg = self.cfg['KPT']
                kpt_model_dir = kpt_cfg['model_dir']
                kpt_batch_size = kpt_cfg['batch_size']
                self.kpt_predictor = KeyPointDetector(
                    kpt_model_dir,
                    args.device,
                    args.run_mode,
                    kpt_batch_size,
                    args.trt_min_shape,
                    args.trt_max_shape,
                    args.trt_opt_shape,
                    args.trt_calib_mode,
                    args.cpu_threads,
                    args.enable_mkldnn,
                    use_dark=False)

            if self.with_mtmct:
                _require_dependency(
                    "REID", ReID, import_error=_PPHUMAN_OPTIONAL_IMPORT_ERROR)
                reid_cfg = self.cfg['REID']
                basemode = self.basemode['REID']
                self.modebase[basemode] = True
                self.reid_predictor = ReID.init_with_cfg(args, reid_cfg)

            if self.with_mot or self.modebase["idbased"] or self.modebase[
                    "skeletonbased"]:
                mot_cfg = self.cfg['MOT']
                model_dir = mot_cfg['model_dir']
                tracker_config = mot_cfg['tracker_config']
                batch_size = mot_cfg['batch_size']
                skip_frame_num = mot_cfg.get('skip_frame_num', -1)
                basemode = self.basemode['MOT']
                self.modebase[basemode] = True
                self.mot_predictor = SDE_Detector(
                    model_dir,
                    tracker_config,
                    args.device,
                    args.run_mode,
                    batch_size,
                    args.trt_min_shape,
                    args.trt_max_shape,
                    args.trt_opt_shape,
                    args.trt_calib_mode,
                    args.cpu_threads,
                    args.enable_mkldnn,
                    skip_frame_num=skip_frame_num,
                    draw_center_traj=self.draw_center_traj,
                    secs_interval=self.secs_interval,
                    threshold=mot_cfg.get('threshold', 0.5),
                    do_entrance_counting=self.do_entrance_counting,
                    do_break_in_counting=self.do_break_in_counting,
                    region_type=self.region_type,
                    region_polygon=self.region_polygon)

            if self.with_video_action:
                video_action_cfg = self.cfg['VIDEO_ACTION']
                basemode = self.basemode['VIDEO_ACTION']
                self.modebase[basemode] = True
                self.video_action_predictor = VideoActionRecognizer.init_with_cfg(
                    args, video_action_cfg)

    def set_file_name(self, path):
        self._show_local_preview = self._resolve_preview_local(path)
        self._save_visual_output = self._resolve_save_visual_output(path)
        self.file_name = _safe_source_name(path)

    def _compute_behavior_timestamp(self,
                                    frame_id,
                                    source_fps,
                                    is_live_source=False,
                                    capture_time=None,
                                    first_capture_time=None):
        if is_live_source:
            if capture_time is None:
                return time.monotonic()
            if first_capture_time is None:
                return 0.0
            return max(0.0, float(capture_time) - float(first_capture_time))
        return frame_id / max(float(source_fps), 1.0)

    @staticmethod
    def _is_truthy(value):
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ('1', 'true', 'yes', 'y', 'on'):
                return True
            if text in ('0', 'false', 'no', 'n', 'off'):
                return False
        return bool(value)

    def _resolve_preview_local(self, path):
        setting = self.cfg.get('preview_local', 'auto')
        is_live_source = isinstance(path, int) or _is_live_stream_source(path)
        if isinstance(setting, str):
            text = setting.strip().lower()
            if text in ('auto', 'live'):
                return is_live_source
            if text in ('file', 'video'):
                return not is_live_source
        return self._is_truthy(setting)

    def _resolve_save_visual_output(self, path):
        setting = self.cfg.get('save_visual_output', True)
        is_live_source = isinstance(path, int) or _is_live_stream_source(path)
        if isinstance(setting, str):
            text = setting.strip().lower()
            if text == 'auto':
                return not is_live_source
            if text in ('live', ):
                return is_live_source
            if text in ('file', 'video'):
                return not is_live_source
        return self._is_truthy(setting)

    def _preview_frame(self, image_bgr):
        if not self._show_local_preview:
            return True

        now = time.monotonic()
        if self.preview_max_fps > 0.0:
            min_interval = 1.0 / self.preview_max_fps
            if (now - self._last_preview_wall_time) < min_interval:
                try:
                    return (cv2.waitKey(1) & 0xFF) != ord('q')
                except Exception:
                    return True
            self._last_preview_wall_time = now

        preview_image = image_bgr
        if self.preview_max_width > 0:
            height, width = preview_image.shape[:2]
            if width > self.preview_max_width:
                scale = float(self.preview_max_width) / float(width)
                target_height = max(1, int(round(height * scale)))
                preview_image = cv2.resize(
                    preview_image, (self.preview_max_width, target_height),
                    interpolation=cv2.INTER_AREA)
        try:
            cv2.imshow(self.preview_window_name, preview_image)
            return (cv2.waitKey(1) & 0xFF) != ord('q')
        except Exception:
            return True

    def _close_pushstream(self, pushstream):
        if pushstream is None:
            return
        if hasattr(pushstream, 'close'):
            try:
                pushstream.close(timeout=2.0)
            finally:
                if hasattr(pushstream, 'stats'):
                    _safe_print("[PushStream] stats: {}".format(
                        pushstream.stats()))
            return
        pipe = getattr(pushstream, 'pipe', None)
        if pipe is None:
            return
        try:
            if pipe.stdin:
                pipe.stdin.close()
        except Exception:
            pass
        try:
            pipe.wait(timeout=2.0)
        except Exception:
            try:
                pipe.terminate()
            except Exception:
                pass

    def _reset_action_predictor_cache(self, predictor):
        if predictor is None:
            return
        if predictor is getattr(self, 'cls_action_predictor', None) and getattr(
                self, 'cls_action_async_worker', None) is not None:
            self.cls_action_async_worker.reset()
            return
        if hasattr(predictor, 'result_history'):
            predictor.result_history.clear()
        if hasattr(predictor, 'id_in_last_frame'):
            predictor.id_in_last_frame = []
        if hasattr(predictor, 'skip_frame_cnt'):
            predictor.skip_frame_cnt = 0

    def _drop_track_cache(self, track_ids):
        stale_ids = {int(tid) for tid in track_ids}
        if not stale_ids:
            return

        if getattr(self, 'cls_action_async_worker', None) is not None:
            for tid in stale_ids:
                self._async_cls_cache.pop(int(tid), None)

        if hasattr(self, 'behavior_filter'):
            for tid in stale_ids:
                self.behavior_filter.history.pop(tid, None)
                self.behavior_filter.last_state.pop(tid, None)

        predictor_names = ['det_action_predictor']
        if getattr(self, 'cls_action_async_worker', None) is None:
            predictor_names.insert(0, 'cls_action_predictor')
        for predictor_name in predictor_names:
            predictor = getattr(self, predictor_name, None)
            if predictor is None:
                continue
            if hasattr(predictor, 'result_history'):
                for tid in stale_ids:
                    predictor.result_history.pop(tid, None)
            if hasattr(predictor, 'id_in_last_frame'):
                predictor.id_in_last_frame = [
                    tid for tid in predictor.id_in_last_frame
                    if int(tid) not in stale_ids
                ]

    def _record_behavior_events(self, events):
        if not events:
            return
        self.latest_behavior_events.extend(events)
        if hasattr(self, 'behavior_event_history'):
            self.behavior_event_history.extend(events)
        for evt in events:
            track_id = evt.get('track_id')
            evt_type = evt.get('event_type')
            level = evt.get('alert_level')
            duration = evt.get('duration_at_trigger', 0.0)
            _safe_print(
                f"[BehaviorEvent] track={track_id} type={evt_type} level={level} duration={duration:.1f}s"
            )

    def _cleanup_behavior_state(self, now_ts):
        if getattr(self, 'cls_action_async_worker', None) is not None:
            return []
        if not hasattr(self, 'behavior_state_machine'):
            return []

        track_ids_before = set(self.behavior_state_machine.tracks.keys())
        stale_events = self.behavior_state_machine.cleanup_stale(now_ts)
        stale_ids = track_ids_before - set(
            self.behavior_state_machine.tracks.keys())
        self._drop_track_cache(stale_ids)
        self._record_behavior_events(stale_events)
        return stale_events

    def _reset_runtime_state_for_reconnect(self):
        self.pipeline_res = Result()
        self.latest_behavior_events = []
        self._clear_async_cls_cache()

        if hasattr(self, 'behavior_state_machine'):
            self.behavior_state_machine.tracks.clear()
        if hasattr(self, 'behavior_filter'):
            self.behavior_filter.history.clear()
            self.behavior_filter.last_state.clear()
        if getattr(self, 'cls_action_async_worker', None) is not None:
            self.cls_action_async_worker.reset()

        self._reset_action_predictor_cache(
            getattr(self, 'cls_action_predictor', None))
        self._reset_action_predictor_cache(
            getattr(self, 'det_action_predictor', None))

        if hasattr(self, 'kpt_buff'):
            self.kpt_buff.keypoint_saver.clear()
            self.kpt_buff.id_to_pop.clear()
            self.kpt_buff.flag_to_pop = False

        for helper_name in ('skeleton_action_visual_helper',
                            'det_action_visual_helper',
                            'cls_action_visual_helper'):
            helper = getattr(self, helper_name, None)
            if helper is not None and hasattr(helper, 'action_history'):
                helper.action_history.clear()

        if self.with_mtmct:
            self.collector = DataCollector()

    def _crop_image_with_mot_aligned(self, image_rgb, mot_res, expand=True):
        boxes = mot_res.get('boxes')
        if boxes is None or len(boxes) == 0:
            return mot_res, [], [], []

        valid_boxes = []
        crop_res = []
        new_bboxes = []
        ori_bboxes = []
        for box in boxes:
            if expand:
                crop_image, new_bbox, ori_bbox = expand_crop(image_rgb,
                                                             box[1:])
            else:
                crop_image, new_bbox, ori_bbox = normal_crop(image_rgb,
                                                             box[1:])
            if crop_image is None:
                continue
            valid_boxes.append(box)
            crop_res.append(crop_image)
            new_bboxes.append(new_bbox)
            ori_bboxes.append(ori_bbox)

        filtered_mot_res = copy.deepcopy(mot_res)
        if len(valid_boxes) == 0:
            box_width = boxes.shape[1] if len(boxes.shape) > 1 else 7
            filtered_mot_res['boxes'] = np.zeros(
                (0, box_width), dtype=boxes.dtype)
        else:
            filtered_mot_res['boxes'] = np.stack(valid_boxes).astype(
                boxes.dtype, copy=False)
        return filtered_mot_res, crop_res, new_bboxes, ori_bboxes

    def _init_flow_state(self, frame_width, frame_height):
        entrance = None
        records = []
        center_traj = [{}] if self.draw_center_traj else None
        id_set = set()
        interval_id_set = set()
        in_id_list = []
        out_id_list = []
        prev_center = {}

        if self.do_entrance_counting or self.do_break_in_counting:
            if self.region_type == 'horizontal':
                entrance = [0, frame_height / 2., frame_width,
                            frame_height / 2.]
            elif self.region_type == 'vertical':
                entrance = [frame_width / 2, 0., frame_width / 2, frame_height]
            elif self.region_type == 'custom':
                entrance = []
                assert len(
                    self.region_polygon
                ) % 2 == 0, "region_polygon should be pairs of coords points when do break_in counting."
                assert len(
                    self.region_polygon
                ) > 6, 'region_type is custom, region_polygon should be at least 3 pairs of point coords.'
                for i in range(0, len(self.region_polygon), 2):
                    entrance.append(
                        [self.region_polygon[i], self.region_polygon[i + 1]])
                entrance.append([frame_width, frame_height])
            else:
                raise ValueError("region_type:{} unsupported.".format(
                    self.region_type))

        return {
            'entrance': entrance,
            'records': records,
            'center_traj': center_traj,
            'id_set': id_set,
            'interval_id_set': interval_id_set,
            'in_id_list': in_id_list,
            'out_id_list': out_id_list,
            'prev_center': prev_center,
        }

    def _ensure_output_target(self, writer, pushstream, out_path, thread_idx,
                              frame_width, frame_height, source_fps):
        if writer is not None or pushstream is not None:
            return writer, pushstream, out_path
        if len(self.pushurl) == 0 and not self._save_visual_output:
            return writer, pushstream, out_path

        video_out_name = self.file_name or 'output'
        if self.multi_camera:
            video_out_name = "{}_t{}".format(video_out_name,
                                             str(thread_idx).zfill(2))

        if len(self.pushurl) > 0:
            pushurl = self.pushurl.rstrip('/') + '/' + video_out_name
            print("the result will push stream to url:{}".format(pushurl))
            push_cfg = self.cfg.get('PUSH_STREAM', {})
            pushstream = PushStream(
                pushurl,
                queue_size=push_cfg.get('queue_size', 1),
                muxdelay=push_cfg.get('muxdelay', 0.1))
            pushstream.initcmd(source_fps, frame_width, frame_height)
            if not self._save_visual_output:
                return writer, pushstream, out_path

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_out_name + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, source_fps,
                                 (frame_width, frame_height))
        return writer, pushstream, out_path

    def _write_visual_output(self, image_bgr, writer, pushstream):
        if pushstream is not None:
            try:
                ok = pushstream.enqueue_frame(image_bgr)
                if not ok:
                    _safe_print("ERROR: Push stream queue is full or closed")
                    return False
            except Exception as exc:
                _safe_print("ERROR: Unable to write to push stream: {}".format(
                    exc))
                return False

        if writer is not None:
            writer.write(image_bgr)
        return self._preview_frame(image_bgr)

    def _build_cls_action_postprocessor(self, cls_cfg):
        normal_id, phone_id, sleeping_id = self._resolve_behavior_class_ids(
            self.clsaction_labels)
        state_cfg = cls_cfg.get('state_machine', {})
        self.behavior_filter = BehaviorFilter(window_size=60, majority_ratio=0.6)
        self.behavior_state_machine = BehaviorStateMachine(
            normal_id=normal_id,
            phone_id=phone_id,
            sleeping_id=sleeping_id,
            sleep_warn_seconds=state_cfg.get('sleep_warn_seconds', 5.0),
            sleep_alert_seconds=state_cfg.get('sleep_alert_seconds', 12.0),
            phone_warn_seconds=state_cfg.get('phone_warn_seconds', 6.0),
            phone_alert_seconds=state_cfg.get('phone_alert_seconds', 15.0),
            warn_cooldown_seconds=state_cfg.get('warn_cooldown_seconds', 8.0),
            alert_cooldown_seconds=state_cfg.get('alert_cooldown_seconds',
                                                  15.0),
            stale_track_seconds=state_cfg.get('stale_track_seconds', 2.0),
            min_confidence=state_cfg.get('min_confidence', None),
            emit_end_event=state_cfg.get('emit_end_event', True))
        return ClsActionPostProcessor(self.behavior_filter,
                                      self.behavior_state_machine)

    def _clear_async_cls_cache(self):
        self._async_cls_cache.clear()
        self._async_cls_last_frame_id = -1
        self._async_cls_frame_buffer.clear()

    def _buffer_async_cls_frame(self, frame_rgb, frame_id):
        if self.cls_action_async_worker is None or self._async_cls_frame_buffer_size <= 0:
            return
        self._async_cls_frame_buffer[int(frame_id)] = frame_rgb.copy()
        while len(self._async_cls_frame_buffer) > self._async_cls_frame_buffer_size:
            self._async_cls_frame_buffer.popitem(last=False)

    def _trim_async_frame_buffer(self, upto_frame_id):
        stale_keys = [
            key for key in self._async_cls_frame_buffer.keys()
            if int(key) <= int(upto_frame_id)
        ]
        for key in stale_keys:
            self._async_cls_frame_buffer.pop(key, None)

    def _drain_async_cls_packets(self):
        worker = self.cls_action_async_worker
        if worker is None:
            return
        if worker.last_error is not None:
            raise RuntimeError(
                "Async cls_action worker failed: {}".format(worker.last_error))

        for packet in worker.poll_packets():
            if packet.frame_id <= self._async_cls_last_frame_id:
                continue
            self._async_cls_last_frame_id = packet.frame_id
            for stale_id in packet.stale_track_ids:
                self._async_cls_cache.pop(int(stale_id), None)
            for tid, res in packet.results.items():
                self._async_cls_cache[int(tid)] = res
            if packet.results:
                self.cls_action_visual_helper.update(list(packet.results.items()))

            self.latest_behavior_events = []
            if packet.events:
                self._record_behavior_events(packet.events)
                frame_rgb = self._async_cls_frame_buffer.get(packet.frame_id)
                if frame_rgb is not None:
                    result_snapshot = {
                        'mot': packet.mot_res,
                        'cls_action': packet.results,
                    }
                    self._notify_behavior_events(
                        frame_rgb,
                        packet.frame_id,
                        result_snapshot=result_snapshot)
            self._trim_async_frame_buffer(packet.frame_id)

    def _build_async_visible_cls_results(self, mot_res):
        if mot_res is None:
            return {}
        boxes = mot_res.get('boxes')
        if boxes is None or len(boxes) == 0:
            return {}
        visible = {}
        for mot_box in boxes:
            track_id = int(mot_box[0])
            res = self._async_cls_cache.get(track_id)
            if res is not None:
                visible[track_id] = dict(res) if isinstance(res, dict) else res
        return visible

    def _resolve_behavior_class_ids(self, labels):
        normal_id, phone_id, sleeping_id = 0, 1, 2
        if not labels:
            return normal_id, phone_id, sleeping_id
        for idx, raw_label in enumerate(labels):
            label = str(raw_label).strip().lower()
            if 'normal' in label:
                normal_id = idx
            elif 'phone' in label:
                phone_id = idx
            elif 'sleep' in label:
                sleeping_id = idx
        return normal_id, phone_id, sleeping_id

    def _get_cls_action_labels(self):
        cls_labels = getattr(self, 'clsaction_labels', None)
        if cls_labels:
            return cls_labels
        cls_labels = getattr(getattr(self, 'cls_action_predictor', None),
                             'pred_config', None)
        cls_labels = getattr(cls_labels, 'labels',
                             ['normal', 'using_phone', 'sleeping'])
        if not cls_labels:
            cls_labels = ['normal', 'using_phone', 'sleeping']
        return cls_labels

    def _build_behavior_alert_frame(self,
                                    frame_rgb,
                                    frame_id,
                                    event,
                                    result_snapshot=None):
        image = draw_minimal_video_result(frame_rgb, result_snapshot
                                          if result_snapshot is not None else
                                          self.pipeline_res,
                                          self._get_cls_action_labels())
        height, width = image.shape[:2]
        level = str(event.get('alert_level', '')).strip().lower()
        color = (0, 0, 255) if level == 'alert' else (0, 215, 255)
        title = "{} {} | ID {}".format(
            str(event.get('event_type', 'unknown')).upper(),
            str(event.get('alert_level', 'unknown')).upper(),
            event.get('track_id', 'n/a'))
        detail = "frame {} | {:.1f}s | {}".format(
            int(frame_id), float(event.get('duration_at_trigger', 0.0)),
            str(event.get('event_kind', 'unknown')).upper())

        left = 12
        top = 12
        banner_width = min(width - 12, 520)
        banner_height = min(height - 12, 86)
        cv2.rectangle(image, (left, top), (banner_width, banner_height),
                      (18, 18, 18), -1)
        cv2.rectangle(image, (left, top), (banner_width, banner_height), color,
                      2)
        cv2.putText(image, title, (left + 12, top + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(image, detail, (left + 12, top + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return image

    def _notify_behavior_events(self, frame_rgb, frame_id, result_snapshot=None):
        sink = getattr(self, 'telegram_alert', None)
        if sink is None or not self.latest_behavior_events:
            return

        source_name = self.file_name if self.file_name else 'camera'
        for event in self.latest_behavior_events:
            if not sink.accepts(event):
                continue
            image = self._build_behavior_alert_frame(
                frame_rgb, frame_id, event, result_snapshot=result_snapshot)
            ok = sink.enqueue(event, image, source_name, frame_id)
            if not ok:
                _safe_print(
                    "[TelegramAlert] queue is full or image encoding failed")

    def get_result(self):
        return self.collector.get_res()

    def run(self, input, thread_idx=0):
        if self.is_video:
            self.predict_video(input, thread_idx=thread_idx)
        else:
            self.predict_image(input)
        self.pipe_timer.info()
        if hasattr(self, 'mot_predictor'):
            self.mot_predictor.det_times.tracking_info(average=True)

    def predict_image(self, input):
        # det
        # det -> attr
        batch_loop_cnt = math.ceil(
            float(len(input)) / self.det_predictor.batch_size)
        self.warmup_frame = min(10, len(input) // 2) - 1
        for i in range(batch_loop_cnt):
            start_index = i * self.det_predictor.batch_size
            end_index = min((i + 1) * self.det_predictor.batch_size, len(input))
            batch_file = input[start_index:end_index]
            batch_input = [decode_image(f, {})[0] for f in batch_file]

            if i > self.warmup_frame:
                self.pipe_timer.total_time.start()
                self.pipe_timer.module_time['det'].start()
            # det output format: class, score, xmin, ymin, xmax, ymax
            det_res = self.det_predictor.predict_image(
                batch_input, visual=False)
            det_res = self.det_predictor.filter_box(det_res,
                                                    self.cfg['crop_thresh'])
            if i > self.warmup_frame:
                self.pipe_timer.module_time['det'].end()
                self.pipe_timer.track_num += len(det_res['boxes'])
            self.pipeline_res.update(det_res, 'det')

            if self.with_human_attr:
                crop_inputs = crop_image_with_det(batch_input, det_res)
                attr_res_list = []

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].start()

                for crop_input in crop_inputs:
                    attr_res = self.attr_predictor.predict_image(
                        crop_input, visual=False)
                    attr_res_list.extend(attr_res['output'])

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].end()

                attr_res = {'output': attr_res_list}
                self.pipeline_res.update(attr_res, 'attr')

            self.pipe_timer.img_num += len(batch_input)
            if i > self.warmup_frame:
                self.pipe_timer.total_time.end()

            if self.cfg['visual']:
                self.visualize_image(batch_file, batch_input, self.pipeline_res)

    def _open_capture(self, source):
        if isinstance(source, int):
            # On Windows webcams are more stable with DirectShow than MSMF.
            backend_candidates = []
            if os.name == 'nt':
                backend_candidates = [
                    ("DSHOW", cv2.CAP_DSHOW),
                    ("MSMF", cv2.CAP_MSMF),
                    ("ANY", cv2.CAP_ANY),
                ]
            else:
                backend_candidates = [("ANY", cv2.CAP_ANY)]

            for backend_name, backend in backend_candidates:
                capture = cv2.VideoCapture(source, backend)
                if capture.isOpened():
                    try:
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    print(
                        "camera {} opened with backend {}".format(
                            source, backend_name))
                    return capture
                capture.release()
            return cv2.VideoCapture(source)

        if _is_live_stream_source(source):
            backend_candidates = []
            ffmpeg_backend = getattr(cv2, 'CAP_FFMPEG', None)
            if ffmpeg_backend is not None:
                backend_candidates.append(("FFMPEG", ffmpeg_backend))
            backend_candidates.append(("ANY", cv2.CAP_ANY))
            for backend_name, backend in backend_candidates:
                capture = cv2.VideoCapture(source, backend)
                if capture.isOpened():
                    try:
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    print(
                        "stream {} opened with backend {}".format(
                            source, backend_name))
                    return capture
                capture.release()

        capture = cv2.VideoCapture(source)
        if _is_live_stream_source(source):
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return capture

    def capturevideo(self, capture, queue, is_live_source=False):
        frame_id = 0
        empty_reads = 0
        max_empty_reads = 30 if is_live_source else 1
        while (1):
            ret, frame = capture.read()
            if not ret or frame is None:
                if is_live_source and empty_reads < max_empty_reads:
                    empty_reads += 1
                    time.sleep(0.05)
                    continue
                return
            empty_reads = 0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use blocking put to ensure no frames are dropped for video files
            # This ensures all frames are processed
            queue.put(frame_rgb)  # Will block if queue is full

    def _predict_video_legacy(self, video_file, thread_idx=0):
        # mot
        # mot -> attr
        # mot -> pose -> action
        is_live_source = isinstance(video_file, int)
        capture = self._open_capture(video_file)
        if not capture.isOpened():
            print("ERROR: Unable to open input source: {}".format(video_file))
            return

        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 30
            print("video source reported invalid fps, fallback to 30")
        _safe_print("video fps: %d, frame_count: %d" % (fps, frame_count))

        if len(self.pushurl) > 0:
            video_out_name = 'output' if self.file_name is None else self.file_name
            pushurl = self.pushurl.rstrip('/') + '/' + video_out_name
            print("the result will push stream to url:{}".format(pushurl))
            pushstream = PushStream(pushurl)
            pushstream.initcmd(fps, width, height)
        elif self.cfg['visual']:
            video_out_name = 'output' if (
                self.file_name is None or
                type(self.file_name) == int) else self.file_name
            if type(video_file) == str and "rtsp" in video_file:
                video_out_name = video_out_name + "_t" + str(thread_idx).zfill(
                    2) + "_rtsp"
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, video_out_name + ".mp4")
            fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_id = 0

        entrance, records, center_traj = None, None, None
        if self.draw_center_traj:
            center_traj = [{}]
        id_set = set()
        interval_id_set = set()
        in_id_list = list()
        out_id_list = list()
        prev_center = dict()
        records = list()
        if self.do_entrance_counting or self.do_break_in_counting:
            if self.region_type == 'horizontal':
                entrance = [0, height / 2., width, height / 2.]
            elif self.region_type == 'vertical':
                entrance = [width / 2, 0., width / 2, height]
            elif self.region_type == 'custom':
                entrance = []
                assert len(
                    self.region_polygon
                ) % 2 == 0, "region_polygon should be pairs of coords points when do break_in counting."
                assert len(
                    self.region_polygon
                ) > 6, 'region_type is custom, region_polygon should be at least 3 pairs of point coords.'

                for i in range(0, len(self.region_polygon), 2):
                    entrance.append(
                        [self.region_polygon[i], self.region_polygon[i + 1]])
                entrance.append([width, height])
            else:
                raise ValueError("region_type:{} unsupported.".format(
                    self.region_type))

        video_fps = fps

        video_action_imgs = []

        if self.with_video_action:
            short_size = self.cfg["VIDEO_ACTION"]["short_size"]
            scale = ShortSizeScale(short_size)

        framequeue = queue.Queue(30)  # Increased buffer for video file processing

        thread = threading.Thread(
            target=self.capturevideo,
            args=(capture, framequeue, is_live_source),
            daemon=True)
        thread.start()
        time.sleep(1.0 if is_live_source else 0.5)
        if not thread.is_alive():	
            _safe_print("ERROR: Video capture thread failed to start!")
            capture.release()
            return
        while True:
            if framequeue.empty():
                # Check if capture thread is still running
                if not thread.is_alive():
                    # Thread stopped - video ended or error
                    # Double-check queue is really empty
                    if framequeue.empty():
                        _safe_print("Video processing completed. Total frames: {}".format(frame_id))
                        break
                
                # Queue is empty but thread still running - wait for more frames
                time.sleep(0.01) 
                continue
            if frame_id % 10 == 0:
                _safe_print('Thread: {}; frame id: {}'.format(thread_idx, frame_id))

            frame_rgb = framequeue.get()
            if frame_id > self.warmup_frame:
                self.pipe_timer.total_time.start()

            if self.modebase["idbased"] or self.modebase["skeletonbased"]:
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].start()

                mot_skip_frame_num = self.mot_predictor.skip_frame_num
                reuse_det_result = False
                if mot_skip_frame_num > 1 and frame_id > 0 and frame_id % mot_skip_frame_num > 0:
                    reuse_det_result = True
                res = self.mot_predictor.predict_image(
                    [copy.deepcopy(frame_rgb)],
                    visual=False,
                    reuse_det_result=reuse_det_result,
                    frame_count=frame_id)

                # mot output format: id, class, score, xmin, ymin, xmax, ymax
                mot_res = parse_mot_res(res)
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].end()
                    self.pipe_timer.track_num += len(mot_res['boxes'])

                if frame_id % 10 == 0:
                    _safe_print("Thread: {}; trackid number: {}".format(
                        thread_idx, len(mot_res['boxes'])))

                # flow_statistic only support single class MOT
                boxes, scores, ids = res[0]  # batch size = 1 in MOT
                if isinstance(boxes, dict):
                    # multi-class: fallback to class 0 for statistics
                    boxes0 = boxes.get(0, np.zeros((0, 4), dtype=np.float32))
                    scores0 = scores.get(0, np.zeros((0,), dtype=np.float32))
                    ids0 = ids.get(0, np.zeros((0,), dtype=np.float32))
                else:
                    boxes0 = boxes[0]
                    scores0 = scores[0]
                    ids0 = ids[0]

                boxes0 = np.array(boxes0)
                scores0 = np.array(scores0)
                ids0 = np.array(ids0)
                if boxes0.ndim == 1 and boxes0.size == 4:
                    boxes0 = boxes0.reshape(1, 4)
                if scores0.ndim == 0:
                    scores0 = scores0.reshape(1)
                if ids0.ndim == 0:
                    ids0 = ids0.reshape(1)

                mot_result = (frame_id + 1, boxes0, scores0,
                              ids0)  # single class
                statistic = flow_statistic(
                    mot_result,
                    self.secs_interval,
                    self.do_entrance_counting,
                    self.do_break_in_counting,
                    self.region_type,
                    video_fps,
                    entrance,
                    id_set,
                    interval_id_set,
                    in_id_list,
                    out_id_list,
                    prev_center,
                    records,
                    ids2names=self.mot_predictor.pred_config.labels)
                records = statistic['records']

                # nothing detected
                if len(mot_res['boxes']) == 0:
                    frame_id += 1
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.img_num += 1
                        self.pipe_timer.total_time.end()
                    if self.cfg['visual']:
                        _, _, fps = self.pipe_timer.get_total_time()
                        im = self.visualize_video(
                            frame_rgb, mot_res, self.collector, frame_id, fps,
                            entrance, records, center_traj)  # visualize
                        if len(self.pushurl) > 0:
                            pushstream.pipe.stdin.write(im.tobytes())
                        else:
                            writer.write(im)
                            if self.file_name is None:  # use camera_id
                                cv2.imshow('Paddle-Pipeline', im)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                    continue

                self.pipeline_res.update(mot_res, 'mot')
                crop_input, new_bboxes, ori_bboxes = crop_image_with_mot(
                    frame_rgb, mot_res)

                if self.with_human_attr:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['attr'].start()
                    attr_res = self.attr_predictor.predict_image(
                        crop_input, visual=False)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['attr'].end()
                    self.pipeline_res.update(attr_res, 'attr')

                if self.with_idbased_detaction:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['det_action'].start()
                    det_action_res = self.det_action_predictor.predict(
                        crop_input, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['det_action'].end()
                    self.pipeline_res.update(det_action_res, 'det_action')

                    if self.cfg['visual']:
                        self.det_action_visual_helper.update(det_action_res)

                if self.with_idbased_clsaction:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['cls_action'].start()
                    frame_timestamp = frame_id / max(float(video_fps), 1.0)
                    cls_action_res = self.cls_action_predictor.predict_with_mot(
                        crop_input, mot_res)
                    cls_action_res = self.apply_behavior_filter(
                        cls_action_res, now=frame_timestamp)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['cls_action'].end()
                    self.pipeline_res.update(cls_action_res, 'cls_action')

                    if self.cfg['visual']:
                        self.cls_action_visual_helper.update(cls_action_res)
                    self._notify_behavior_events(frame_rgb, frame_id)

                if self.with_skeleton_action:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].start()
                    kpt_pred = self.kpt_predictor.predict_image(
                        crop_input, visual=False)
                    keypoint_vector, score_vector = translate_to_ori_images(
                        kpt_pred, np.array(new_bboxes))
                    kpt_res = {}
                    kpt_res['keypoint'] = [
                        keypoint_vector.tolist(), score_vector.tolist()
                    ] if len(keypoint_vector) > 0 else [[], []]
                    kpt_res['bbox'] = ori_bboxes
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].end()

                    self.pipeline_res.update(kpt_res, 'kpt')

                    self.kpt_buff.update(kpt_res, mot_res)  # collect kpt output
                    state = self.kpt_buff.get_state(
                    )  # whether frame num is enough or lost tracker

                    skeleton_action_res = {}
                    if state:
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time[
                                'skeleton_action'].start()
                        collected_keypoint = self.kpt_buff.get_collected_keypoint(
                        )  # reoragnize kpt output with ID
                        # Use relative keypoints for custom STGCN (normalized coordinates)
                        # Original: parse_mot_keypoint(collected_keypoint, self.coord_size)
                        skeleton_action_input = parse_mot_keypoint_relative(
                            collected_keypoint)
                        skeleton_action_res = self.skeleton_action_predictor.predict_skeleton_with_mot(
                            skeleton_action_input)
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time['skeleton_action'].end()
                        self.pipeline_res.update(skeleton_action_res,
                                                 'skeleton_action')

                    if self.cfg['visual']:
                        self.skeleton_action_visual_helper.update(
                            skeleton_action_res)
                
                # Keypoint only mode (without STGCN)
                elif self.with_kpt_only:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].start()
                    kpt_pred = self.kpt_predictor.predict_image(
                        crop_input, visual=False)
                    keypoint_vector, score_vector = translate_to_ori_images(
                        kpt_pred, np.array(new_bboxes))
                    kpt_res = {}
                    kpt_res['keypoint'] = [
                        keypoint_vector.tolist(), score_vector.tolist()
                    ] if len(keypoint_vector) > 0 else [[], []]
                    kpt_res['bbox'] = ori_bboxes
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].end()

                    self.pipeline_res.update(kpt_res, 'kpt')

                if self.with_mtmct and frame_id % 10 == 0:
                    crop_input, img_qualities, rects = self.reid_predictor.crop_image_with_mot(
                        frame_rgb, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['reid'].start()
                    reid_res = self.reid_predictor.predict_batch(crop_input)

                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['reid'].end()

                    reid_res_dict = {
                        'features': reid_res,
                        "qualities": img_qualities,
                        "rects": rects
                    }
                    self.pipeline_res.update(reid_res_dict, 'reid')
                else:
                    self.pipeline_res.clear('reid')

            if self.with_video_action:
                # get the params
                frame_len = self.cfg["VIDEO_ACTION"]["frame_len"]
                sample_freq = self.cfg["VIDEO_ACTION"]["sample_freq"]

                if sample_freq * frame_len > frame_count:  # video is too short
                    sample_freq = int(frame_count / frame_len)

                # filter the warmup frames
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['video_action'].start()

                # collect frames
                if frame_id % sample_freq == 0:
                    # Scale image
                    scaled_img = scale(frame_rgb)
                    video_action_imgs.append(scaled_img)

                # the number of collected frames is enough to predict video action
                if len(video_action_imgs) == frame_len:
                    classes, scores = self.video_action_predictor.predict(
                        video_action_imgs)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['video_action'].end()

                    video_action_res = {"class": classes[0], "score": scores[0]}
                    self.pipeline_res.update(video_action_res, 'video_action')

                    print("video_action_res:", video_action_res)

                    video_action_imgs.clear()  # next clip

            self.collector.append(frame_id, self.pipeline_res)

            if frame_id > self.warmup_frame:
                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()
            frame_id += 1

            if self.cfg['visual']:
                _, _, fps = self.pipe_timer.get_total_time()

                im = self.visualize_video(frame_rgb, self.pipeline_res,
                                          self.collector, frame_id, fps,
                                          entrance, records, center_traj)
                if len(self.pushurl) > 0:
                    pushstream.pipe.stdin.write(im.tobytes())
                else:
                    writer.write(im)
                    if self.file_name is None:  # use camera_id
                        cv2.imshow('Paddle-Pipeline', im)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        if self.cfg['visual'] and len(self.pushurl) == 0:
            writer.release()
            _safe_print('save result to {}'.format(out_path))

        sink = getattr(self, 'telegram_alert', None)
        if sink is not None:
            sink.flush()
            _safe_print("[TelegramAlert] stats: {}".format(sink.stats()))

    def predict_video(self, video_file, thread_idx=0):
        is_live_source = _is_live_stream_source(video_file)
        is_reconnectable_source = _is_reconnectable_live_source(video_file)

        capture = None
        open_attempts = 3 if is_live_source else 1
        for attempt in range(open_attempts):
            capture = self._open_capture(video_file)
            if capture is not None and capture.isOpened():
                break
            if capture is not None:
                capture.release()
            capture = None
            if attempt + 1 < open_attempts:
                time.sleep(0.5)

        if capture is None or not capture.isOpened():
            print("ERROR: Unable to open input source: {}".format(video_file))
            return

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0:
            source_fps = 30.0
            print("video source reported invalid fps, fallback to 30")
        _safe_print("video fps: %d, frame_count: %d" %
                    (int(round(source_fps)), frame_count))

        reader = FrameSourceReader(
            source=video_file,
            capture=capture,
            open_capture_fn=self._open_capture,
            is_live_source=is_live_source,
            is_reconnectable_source=is_reconnectable_source,
            startup_timeout=5.0 if is_live_source else 1.0,
            live_empty_reads_before_retry=30,
            live_retry_delay_seconds=0.1)
        reader.start()

        if not reader.wait_for_first_frame():
            if reader.is_finished() and reader.empty():
                _safe_print("Video processing completed. Total frames: 0")
            else:
                _safe_print(
                    "ERROR: Timed out waiting for first frame from {}".format(
                        video_file))
            reader.stop()
            reader.join(timeout=2.0)
            return

        frame_id = 0
        first_capture_time = None
        last_reconnect_count = reader.reconnect_count
        writer = None
        pushstream = None
        out_path = None
        flow_state = None
        video_action_imgs = []

        if self.with_video_action:
            short_size = self.cfg["VIDEO_ACTION"]["short_size"]
            scale = ShortSizeScale(short_size)
        if self.cls_action_async_worker is not None:
            self.cls_action_async_worker.start()
            self._clear_async_cls_cache()

        try:
            while True:
                packet = reader.get(timeout=0.5 if is_live_source else 0.1)
                reconnect_changed = reader.reconnect_count != last_reconnect_count

                if packet is None:
                    if self.cls_action_async_worker is not None:
                        self._drain_async_cls_packets()
                    if reconnect_changed:
                        last_reconnect_count = reader.reconnect_count
                        self._reset_runtime_state_for_reconnect()
                        video_action_imgs = []
                        if flow_state is not None:
                            flow_state = self._init_flow_state(
                                frame_width, frame_height)
                    if reader.is_finished() and reader.empty():
                        _safe_print(
                            "Video processing completed. Total frames: {}".
                            format(frame_id))
                        break
                    continue

                frame_rgb = packet.frame_rgb
                frame_height, frame_width = frame_rgb.shape[:2]
                self._web_last_capture_time = packet.capture_time
                self._web_last_input_frame_index = packet.frame_index
                if flow_state is None:
                    flow_state = self._init_flow_state(frame_width,
                                                       frame_height)

                if reconnect_changed:
                    last_reconnect_count = reader.reconnect_count
                    self._reset_runtime_state_for_reconnect()
                    video_action_imgs = []
                    flow_state = self._init_flow_state(frame_width,
                                                       frame_height)

                writer, pushstream, out_path = self._ensure_output_target(
                    writer, pushstream, out_path, thread_idx, frame_width,
                    frame_height, source_fps)

                if first_capture_time is None:
                    first_capture_time = packet.capture_time
                now_ts = self._compute_behavior_timestamp(
                    frame_id=frame_id,
                    source_fps=source_fps,
                    is_live_source=is_live_source,
                    capture_time=packet.capture_time,
                    first_capture_time=first_capture_time)

                if frame_id % 10 == 0:
                    _safe_print('Thread: {}; frame id: {}'.format(
                        thread_idx, frame_id))

                self.pipeline_res = Result()
                self.latest_behavior_events = []
                if self.cls_action_async_worker is not None:
                    self._drain_async_cls_packets()

                if frame_id > self.warmup_frame:
                    self.pipe_timer.total_time.start()

                if self.modebase["idbased"] or self.modebase["skeletonbased"]:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['mot'].start()

                    mot_skip_frame_num = self.mot_predictor.skip_frame_num
                    reuse_det_result = False
                    if mot_skip_frame_num > 1 and frame_id > 0 and frame_id % mot_skip_frame_num > 0:
                        reuse_det_result = True

                    mot_raw = self.mot_predictor.predict_image(
                        [copy.deepcopy(frame_rgb)],
                        visual=False,
                        reuse_det_result=reuse_det_result,
                        frame_count=frame_id)
                    mot_res = parse_mot_res(mot_raw)

                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['mot'].end()
                        self.pipe_timer.track_num += len(mot_res['boxes'])

                    if frame_id % 10 == 0:
                        _safe_print("Thread: {}; trackid number: {}".format(
                            thread_idx, len(mot_res['boxes'])))

                    boxes, scores, ids = mot_raw[0]
                    if isinstance(boxes, dict):
                        boxes0 = boxes.get(0, np.zeros((0, 4), dtype=np.float32))
                        scores0 = scores.get(0,
                                             np.zeros((0,), dtype=np.float32))
                        ids0 = ids.get(0, np.zeros((0,), dtype=np.float32))
                    else:
                        boxes0 = boxes[0]
                        scores0 = scores[0]
                        ids0 = ids[0]

                    boxes0 = np.array(boxes0)
                    scores0 = np.array(scores0)
                    ids0 = np.array(ids0)
                    if boxes0.ndim == 1 and boxes0.size == 4:
                        boxes0 = boxes0.reshape(1, 4)
                    if scores0.ndim == 0:
                        scores0 = scores0.reshape(1)
                    if ids0.ndim == 0:
                        ids0 = ids0.reshape(1)

                    statistic = flow_statistic(
                        (frame_id + 1, boxes0, scores0, ids0),
                        self.secs_interval,
                        self.do_entrance_counting,
                        self.do_break_in_counting,
                        self.region_type,
                        source_fps,
                        flow_state['entrance'],
                        flow_state['id_set'],
                        flow_state['interval_id_set'],
                        flow_state['in_id_list'],
                        flow_state['out_id_list'],
                        flow_state['prev_center'],
                        flow_state['records'],
                        ids2names=self.mot_predictor.pred_config.labels)
                    flow_state['records'] = statistic['records']

                    self.pipeline_res.update(mot_res, 'mot')
                    mot_res, crop_input, new_bboxes, ori_bboxes = self._crop_image_with_mot_aligned(
                        frame_rgb, mot_res)
                    self.pipeline_res.update(mot_res, 'mot')

                    if len(mot_res['boxes']) == 0:
                        if self.cls_action_async_worker is not None:
                            self._buffer_async_cls_frame(frame_rgb, frame_id)
                            self.cls_action_async_worker.submit_cleanup(
                                frame_id, now_ts)
                            self._drain_async_cls_packets()
                        else:
                            self._cleanup_behavior_state(now_ts)
                    else:
                        if self.with_human_attr:
                            if frame_id > self.warmup_frame:
                                self.pipe_timer.module_time['attr'].start()
                            attr_res = self.attr_predictor.predict_image(
                                crop_input, visual=False)
                            if frame_id > self.warmup_frame:
                                self.pipe_timer.module_time['attr'].end()
                            self.pipeline_res.update(attr_res, 'attr')

                        if self.with_idbased_detaction:
                            if frame_id > self.warmup_frame:
                                self.pipe_timer.module_time[
                                    'det_action'].start()
                            det_action_res = self.det_action_predictor.predict(
                                crop_input, mot_res)
                            if frame_id > self.warmup_frame:
                                self.pipe_timer.module_time['det_action'].end()
                            self.pipeline_res.update(det_action_res,
                                                     'det_action')
                            if self.cfg['visual']:
                                self.det_action_visual_helper.update(
                                    det_action_res)

                        if self.with_idbased_clsaction:
                            if self.cls_action_async_worker is not None:
                                self._buffer_async_cls_frame(frame_rgb, frame_id)
                                self.cls_action_async_worker.submit_infer(
                                    frame_id, now_ts, crop_input, mot_res)
                                self._drain_async_cls_packets()
                                cls_action_res = self._build_async_visible_cls_results(
                                    mot_res)
                            else:
                                if frame_id > self.warmup_frame:
                                    self.pipe_timer.module_time[
                                        'cls_action'].start()
                                cls_action_res = self.cls_action_predictor.predict_with_mot(
                                    crop_input, mot_res)
                                cls_action_res = self.apply_behavior_filter(
                                    cls_action_res, now=now_ts)
                                if frame_id > self.warmup_frame:
                                    self.pipe_timer.module_time[
                                        'cls_action'].end()
                            self.pipeline_res.update(cls_action_res,
                                                     'cls_action')
                            if self.cfg['visual']:
                                if self.cls_action_async_worker is None:
                                    self.cls_action_visual_helper.update(
                                        cls_action_res)
                            if self.cls_action_async_worker is None:
                                self._notify_behavior_events(frame_rgb,
                                                             frame_id)

                        if self.with_skeleton_action:
                            if frame_id > self.warmup_frame:
                                self.pipe_timer.module_time['kpt'].start()
                            kpt_pred = self.kpt_predictor.predict_image(
                                crop_input, visual=False)
                            keypoint_vector, score_vector = translate_to_ori_images(
                                kpt_pred, np.array(new_bboxes))
                            kpt_res = {
                                'keypoint':
                                [keypoint_vector.tolist(), score_vector.tolist()]
                                if len(keypoint_vector) > 0 else [[], []],
                                'bbox':
                                ori_bboxes
                            }
                            if frame_id > self.warmup_frame:
                                self.pipe_timer.module_time['kpt'].end()
                            self.pipeline_res.update(kpt_res, 'kpt')

                            self.kpt_buff.update(kpt_res, mot_res)
                            state = self.kpt_buff.get_state()
                            skeleton_action_res = {}
                            if state:
                                if frame_id > self.warmup_frame:
                                    self.pipe_timer.module_time[
                                        'skeleton_action'].start()
                                collected_keypoint = self.kpt_buff.get_collected_keypoint(
                                )
                                skeleton_action_input = parse_mot_keypoint_relative(
                                    collected_keypoint)
                                skeleton_action_res = self.skeleton_action_predictor.predict_skeleton_with_mot(
                                    skeleton_action_input)
                                if frame_id > self.warmup_frame:
                                    self.pipe_timer.module_time[
                                        'skeleton_action'].end()
                                self.pipeline_res.update(skeleton_action_res,
                                                         'skeleton_action')
                            if self.cfg['visual']:
                                self.skeleton_action_visual_helper.update(
                                    skeleton_action_res)

                        elif self.with_kpt_only:
                            if frame_id > self.warmup_frame:
                                self.pipe_timer.module_time['kpt'].start()
                            kpt_pred = self.kpt_predictor.predict_image(
                                crop_input, visual=False)
                            keypoint_vector, score_vector = translate_to_ori_images(
                                kpt_pred, np.array(new_bboxes))
                            kpt_res = {
                                'keypoint':
                                [keypoint_vector.tolist(), score_vector.tolist()]
                                if len(keypoint_vector) > 0 else [[], []],
                                'bbox':
                                ori_bboxes
                            }
                            if frame_id > self.warmup_frame:
                                self.pipe_timer.module_time['kpt'].end()
                            self.pipeline_res.update(kpt_res, 'kpt')

                        if self.with_mtmct and frame_id % 10 == 0:
                            crop_input, img_qualities, rects = self.reid_predictor.crop_image_with_mot(
                                frame_rgb, mot_res)
                            if frame_id > self.warmup_frame:
                                self.pipe_timer.module_time['reid'].start()
                            reid_res = self.reid_predictor.predict_batch(
                                crop_input)
                            if frame_id > self.warmup_frame:
                                self.pipe_timer.module_time['reid'].end()
                            reid_res_dict = {
                                'features': reid_res,
                                'qualities': img_qualities,
                                'rects': rects
                            }
                            self.pipeline_res.update(reid_res_dict, 'reid')

                if self.with_video_action:
                    frame_len = self.cfg["VIDEO_ACTION"]["frame_len"]
                    sample_freq = max(1,
                                      int(self.cfg["VIDEO_ACTION"][
                                          "sample_freq"]))
                    if frame_count > 0 and sample_freq * frame_len > frame_count:
                        sample_freq = max(1, int(frame_count / frame_len))

                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['video_action'].start()

                    if frame_id % sample_freq == 0:
                        video_action_imgs.append(scale(frame_rgb))

                    if len(video_action_imgs) == frame_len:
                        classes, scores = self.video_action_predictor.predict(
                            video_action_imgs)
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time['video_action'].end()
                        video_action_res = {
                            "class": classes[0],
                            "score": scores[0]
                        }
                        self.pipeline_res.update(video_action_res,
                                                 'video_action')
                        print("video_action_res:", video_action_res)
                        video_action_imgs.clear()

                self.collector.append(frame_id, self.pipeline_res)

                if frame_id > self.warmup_frame:
                    self.pipe_timer.img_num += 1
                    self.pipe_timer.total_time.end()

                visual_frame_id = frame_id + 1
                frame_id += 1

                if self.cfg['visual']:
                    _, _, display_fps = self.pipe_timer.get_total_time()
                    im = self.visualize_video(frame_rgb, self.pipeline_res,
                                              self.collector, visual_frame_id,
                                              display_fps,
                                              flow_state['entrance'],
                                              flow_state['records'],
                                              flow_state['center_traj'])
                    if not self._write_visual_output(im, writer, pushstream):
                        break
        finally:
            reader.stop()
            reader.join(timeout=2.0)
            if self.cls_action_async_worker is not None:
                self.cls_action_async_worker.stop(timeout=2.0)

            if writer is not None:
                writer.release()
                if out_path is not None:
                    _safe_print('save result to {}'.format(out_path))

            self._close_pushstream(pushstream)

            if self._show_local_preview:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            sink = getattr(self, 'telegram_alert', None)
            if sink is not None:
                sink.flush()
                _safe_print("[TelegramAlert] stats: {}".format(sink.stats()))

    def visualize_video(self,
                        image_rgb,
                        result,
                        collector,
                        frame_id,
                        fps,
                        entrance=None,
                        records=None,
                        center_traj=None):
        if self.cfg.get('visual_style', 'classic') == 'minimal':
            cls_labels = getattr(getattr(self, 'cls_action_predictor', None),
                                 'pred_config', None)
            cls_labels = getattr(cls_labels, 'labels',
                                 ['normal', 'using_phone', 'sleeping'])
            return draw_minimal_video_result(image_rgb, result, cls_labels)

        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        mot_res = copy.deepcopy(result.get('mot'))

        if mot_res is not None:
            ids = mot_res['boxes'][:, 0]
            scores = mot_res['boxes'][:, 2]
            boxes = mot_res['boxes'][:, 3:]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        else:
            boxes = np.zeros([0, 4])
            ids = np.zeros([0])
            scores = np.zeros([0])

        # single class, still need to be defaultdict type for ploting
        num_classes = 1
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        online_tlwhs[0] = boxes
        online_scores[0] = scores
        online_ids[0] = ids

        if mot_res is not None:
            image = plot_tracking_dict(
                image,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                ids2names=self.mot_predictor.pred_config.labels,
                do_entrance_counting=self.do_entrance_counting,
                do_break_in_counting=self.do_break_in_counting,
                entrance=entrance,
                records=records,
                center_traj=center_traj)

        human_attr_res = result.get('attr')
        if human_attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            human_attr_res = human_attr_res['output']
            image = visualize_attr(image, human_attr_res, boxes)
            image = np.array(image)

        kpt_res = result.get('kpt')
        if kpt_res is not None:
            image = visualize_pose(
                image,
                kpt_res,
                visual_thresh=self.cfg['kpt_thresh'],
                returnimg=True)

        video_action_res = result.get('video_action')
        if video_action_res is not None:
            video_action_score = None
            if video_action_res and video_action_res["class"] == 1:
                video_action_score = video_action_res["score"]
            mot_boxes = mot_res['boxes'] if mot_res else None
            image = visualize_action(
                image,
                mot_boxes,
                action_visual_collector=None,
                action_text="SkeletonAction",
                video_action_score=video_action_score,
                video_action_text="Fight")

        visual_helper_for_display = []
        action_to_display = []

        skeleton_action_res = result.get('skeleton_action')
        if skeleton_action_res is not None:
            visual_helper_for_display.append(self.skeleton_action_visual_helper)
            # action_to_display.append("Falling")  # Original for fall detection
            action_to_display.append("Ngu gat")  # Changed for sleeping detection (class 1)

        det_action_res = result.get('det_action')
        if det_action_res is not None:
            visual_helper_for_display.append(self.det_action_visual_helper)
            action_to_display.append("Smoking")

        cls_action_res = result.get('cls_action')
        if cls_action_res is not None and mot_res is not None:
            cls_labels = None
            if hasattr(self, 'cls_action_predictor') and hasattr(
                    self.cls_action_predictor, 'pred_config'):
                cls_labels = self.cls_action_predictor.pred_config.labels
            if not cls_labels:
                cls_labels = ['normal', 'using_phone', 'sleeping']
            cls_labels = [localize_label(l) for l in cls_labels]

            if isinstance(cls_action_res, dict):
                id_to_res = cls_action_res
            else:
                id_to_res = {}
                for item in cls_action_res:
                    try:
                        tid, res = item
                    except Exception:
                        continue
                    id_to_res[int(tid)] = res

            cls_texts = []
            for mot_box in mot_res['boxes']:
                track_id = int(mot_box[0])
                res = id_to_res.get(track_id)
                if res is None:
                    cls_texts.append([])
                    continue

                cls_id = None
                score = None
                scores_vec = None
                if isinstance(res, dict):
                    if 'class' in res:
                        cls_id = int(res.get('class', 0))
                        score = res.get('score', None)
                        score = float(score) if score is not None else None
                    scores_vec = res.get('scores')
                elif isinstance(res, (list, tuple, np.ndarray)):
                    if len(res) == 2 and not isinstance(
                            res[0], (list, tuple, np.ndarray)):
                        cls_id = int(res[0])
                        score = float(res[1])
                    else:
                        scores_vec = res

                if scores_vec is not None and cls_id is None:
                    scores = np.array(scores_vec, dtype=np.float32).flatten()
                    if scores.size > 0:
                        cls_id = int(scores.argmax())
                        score = float(scores.max())
                if score is None and scores_vec is not None and cls_id is not None:
                    try:
                        score = float(scores_vec[cls_id])
                    except Exception:
                        score = None

                if cls_id is None:
                    cls_texts.append([])
                    continue

                label = cls_labels[cls_id] if cls_id < len(
                    cls_labels) else str(cls_id)
                state_name = None
                if isinstance(res, dict):
                    state_name = res.get('state')
                suffix = ""
                if state_name and str(state_name).upper() != 'NORMAL':
                    suffix = f" | {state_name}"
                if score is None:
                    cls_texts.append([f"{label}{suffix}"])
                else:
                    cls_texts.append([f"{label}: {score:.2f}{suffix}"])

            image = visualize_attr(image, cls_texts, mot_res['boxes'][:, 1:])

        if len(visual_helper_for_display) > 0:
            mot_boxes = mot_res['boxes'] if mot_res is not None else None
            image = visualize_action(image, mot_boxes,
                                     visual_helper_for_display,
                                     action_to_display)

        return image

    def visualize_image(self, im_files, images, result):
        start_idx, boxes_num_i = 0, 0
        det_res = result.get('det')
        human_attr_res = result.get('attr')

        for i, (im_file, im) in enumerate(zip(im_files, images)):
            if det_res is not None:
                det_res_i = {}
                boxes_num_i = det_res['boxes_num'][i]
                det_res_i['boxes'] = det_res['boxes'][start_idx:start_idx +
                                                      boxes_num_i, :]
                det_labels = ['hoc_sinh']
                if hasattr(self, 'det_predictor') and hasattr(
                        self.det_predictor, 'pred_config'):
                    pred_labels = getattr(self.det_predictor.pred_config,
                                          'labels', None)
                    if pred_labels:
                        det_labels = pred_labels
                im = visualize_box_mask(
                    im,
                    det_res_i,
                    labels=det_labels,
                    threshold=self.cfg['crop_thresh'])
                im = np.ascontiguousarray(np.copy(im))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            if human_attr_res is not None:
                human_attr_res_i = human_attr_res['output'][start_idx:start_idx
                                                            + boxes_num_i]
                im = visualize_attr(im, human_attr_res_i, det_res_i['boxes'])

            img_name = os.path.split(im_file)[-1]
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, img_name)
            cv2.imwrite(out_path, im)
            _safe_print("save result to: " + out_path)
            start_idx += boxes_num_i

    def apply_behavior_filter(self, cls_action_res, now=None):
        if cls_action_res is None or self.cls_action_postprocessor is None:
            return cls_action_res
        now_ts = time.time() if now is None else float(now)
        self.latest_behavior_events = []
        id_to_res, events, _ = self.cls_action_postprocessor.process(
            cls_action_res, now_ts)
        self._record_behavior_events(events)
        if isinstance(cls_action_res, dict):
            return id_to_res
        return list(id_to_res.items())


def main():
    cfg = merge_cfg(FLAGS)  # use command params to update config
    print_arguments(cfg)

    pipeline = Pipeline(FLAGS, cfg)
    # pipeline.run()
    pipeline.run_multithreads()


if __name__ == '__main__':
    paddle.enable_static()

    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, XPU or NPU"

    main()
