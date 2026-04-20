from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from behavior_filter import BehaviorFilter
from behavior_state_machine import BehaviorStateMachine
from pphuman.action_infer import ClsActionRecognizer


def _normalize_cls_action_res(cls_action_res) -> Dict[int, dict]:
    if cls_action_res is None:
        return {}
    if isinstance(cls_action_res, dict):
        return {int(tid): res for tid, res in cls_action_res.items()}

    id_to_res = {}
    for item in cls_action_res:
        try:
            tid, res = item
        except Exception:
            continue
        id_to_res[int(tid)] = res
    return id_to_res


def _parse_cls_action_entry(res) -> Tuple[int, Optional[float], Optional[Iterable[float]]]:
    cls_id = 0
    score = None
    scores_vec = None

    if isinstance(res, dict):
        cls_id = int(res.get('class', 0))
        raw_score = res.get('score', None)
        score = float(raw_score) if raw_score is not None else None
        scores_vec = res.get('scores')
    elif isinstance(res, (list, tuple, np.ndarray)):
        if len(res) == 2 and not isinstance(res[0], (list, tuple, np.ndarray)):
            cls_id = int(res[0])
            score = float(res[1])
        else:
            scores_vec = res

    if scores_vec is not None and score is None:
        scores = np.array(scores_vec, dtype=np.float32).flatten()
        if scores.size > 0:
            cls_id = int(scores.argmax())
            score = float(scores.max())

    return cls_id, score, scores_vec


def _resolve_behavior_class_ids(labels):
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


class ClsActionPostProcessor(object):
    def __init__(self, behavior_filter, behavior_state_machine):
        self.behavior_filter = behavior_filter
        self.behavior_state_machine = behavior_state_machine

    def drop_track_cache(self, track_ids):
        stale_ids = {int(tid) for tid in track_ids}
        if not stale_ids:
            return
        for tid in stale_ids:
            self.behavior_filter.history.pop(tid, None)
            self.behavior_filter.last_state.pop(tid, None)
            self.behavior_state_machine.tracks.pop(tid, None)

    def reset(self):
        self.behavior_filter.history.clear()
        self.behavior_filter.last_state.clear()
        self.behavior_state_machine.tracks.clear()

    def cleanup_stale(self, now_ts):
        track_ids_before = set(self.behavior_state_machine.tracks.keys())
        events = self.behavior_state_machine.cleanup_stale(now_ts)
        stale_ids = track_ids_before - set(self.behavior_state_machine.tracks.keys())
        self.drop_track_cache(stale_ids)
        return events, sorted(int(tid) for tid in stale_ids)

    def cleanup_only(self, now_ts):
        return {}, *self.cleanup_stale(now_ts)

    def process(self, cls_action_res, now_ts):
        id_to_res = _normalize_cls_action_res(cls_action_res)
        events = []

        for tid, res in id_to_res.items():
            cls_id, score, scores_vec = _parse_cls_action_entry(res)
            voted = self.behavior_filter.update(int(tid), int(cls_id), score)
            if voted is None:
                last_state = self.behavior_filter.get_last(int(tid))
                if last_state is not None:
                    voted_cls, voted_score = last_state
                else:
                    voted_cls, voted_score = cls_id, score
            else:
                voted_cls, voted_score = voted

            state_name, state_events = self.behavior_state_machine.update(
                track_id=int(tid),
                behavior=int(voted_cls),
                now=now_ts,
                score=voted_score)
            events.extend(state_events)

            if isinstance(res, dict):
                output = dict(res)
            else:
                output = {}

            output['class'] = int(voted_cls)
            if voted_score is None and scores_vec is not None:
                try:
                    voted_score = float(scores_vec[int(voted_cls)])
                except Exception:
                    voted_score = None
            if voted_score is None and score is not None:
                voted_score = float(score)
            output['score'] = voted_score
            output['scores'] = scores_vec if scores_vec is not None else output.get(
                'scores')
            output['voted'] = voted is not None
            output['state'] = state_name
            if state_events:
                output['alert_event'] = state_events[-1]
            elif 'alert_event' in output:
                del output['alert_event']
            id_to_res[int(tid)] = output

        stale_events, stale_ids = self.cleanup_stale(now_ts)
        events.extend(stale_events)
        return id_to_res, events, stale_ids


@dataclass
class AsyncClsActionTask:
    frame_id: int
    now_ts: float
    crop_input: List[np.ndarray]
    mot_res: Optional[dict]
    mode: str = 'infer'


@dataclass
class AsyncClsActionPacket:
    frame_id: int
    now_ts: float
    mot_res: Optional[dict]
    results: Dict[int, dict]
    events: List[dict]
    stale_track_ids: List[int]


class AsyncClsActionWorker(object):
    def __init__(self,
                 predictor,
                 postprocessor,
                 queue_size=1,
                 worker_device='CPU',
                 worker_run_mode='paddle'):
        queue_size = min(4, max(1, int(queue_size)))
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.worker_device = str(worker_device).upper()
        self.worker_run_mode = str(worker_run_mode)
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=max(4, queue_size * 2 + 2))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name='async-cls-action-worker')
        self._submit_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.last_error = None

    @classmethod
    def init_with_cfg(cls,
                      args,
                      cfg):
        async_cfg = cfg.get('async_worker', {}) or {}
        worker_cfg = copy.deepcopy(cfg)
        worker_cfg['skip_frame_num'] = 0

        worker_args = copy.deepcopy(args)
        worker_device = str(async_cfg.get('worker_device', 'same')).strip().lower()
        if worker_device in ('', 'same', 'auto'):
            resolved_device = str(args.device).upper()
        else:
            resolved_device = worker_device.upper()

        worker_run_mode = str(async_cfg.get('worker_run_mode', 'same')).strip()
        if resolved_device == 'CPU':
            resolved_run_mode = 'paddle'
        elif worker_run_mode in ('', 'same', 'auto'):
            resolved_run_mode = str(args.run_mode)
        else:
            resolved_run_mode = worker_run_mode

        worker_args.device = resolved_device
        worker_args.run_mode = resolved_run_mode

        predictor = ClsActionRecognizer.init_with_cfg(worker_args, worker_cfg)
        labels = getattr(getattr(predictor, 'pred_config', None), 'labels', None)
        normal_id, phone_id, sleeping_id = _resolve_behavior_class_ids(labels)
        postprocessor = ClsActionPostProcessor(
            behavior_filter=BehaviorFilter(window_size=60, majority_ratio=0.6),
            behavior_state_machine=BehaviorStateMachine(
                normal_id=normal_id,
                phone_id=phone_id,
                sleeping_id=sleeping_id,
                sleep_warn_seconds=cfg.get('state_machine', {}).get(
                    'sleep_warn_seconds', 5.0),
                sleep_alert_seconds=cfg.get('state_machine', {}).get(
                    'sleep_alert_seconds', 12.0),
                phone_warn_seconds=cfg.get('state_machine', {}).get(
                    'phone_warn_seconds', 6.0),
                phone_alert_seconds=cfg.get('state_machine', {}).get(
                    'phone_alert_seconds', 15.0),
                warn_cooldown_seconds=cfg.get('state_machine', {}).get(
                    'warn_cooldown_seconds', 8.0),
                alert_cooldown_seconds=cfg.get('state_machine', {}).get(
                    'alert_cooldown_seconds', 15.0),
                stale_track_seconds=cfg.get('state_machine', {}).get(
                    'stale_track_seconds', 2.0),
                min_confidence=cfg.get('state_machine', {}).get(
                    'min_confidence', None),
                emit_end_event=cfg.get('state_machine', {}).get(
                    'emit_end_event', True)))
        return cls(
            predictor=predictor,
            postprocessor=postprocessor,
            queue_size=async_cfg.get('queue_size', 1),
            worker_device=resolved_device,
            worker_run_mode=resolved_run_mode)

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self, timeout=2.0):
        self.stop_event.set()
        try:
            self.task_queue.put_nowait(
                AsyncClsActionTask(
                    frame_id=-1, now_ts=0.0, crop_input=[], mot_res=None, mode='stop'))
        except Exception:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def reset(self):
        with self._state_lock:
            self.postprocessor.reset()
            self._reset_predictor_cache()
        self._drain_queue(self.task_queue)
        self._drain_queue(self.output_queue)

    def submit_infer(self, frame_id, now_ts, crop_input, mot_res):
        task = AsyncClsActionTask(
            frame_id=int(frame_id),
            now_ts=float(now_ts),
            crop_input=list(crop_input or []),
            mot_res=copy.deepcopy(mot_res),
            mode='infer')
        self._replace_pending_task(task)

    def submit_cleanup(self, frame_id, now_ts):
        task = AsyncClsActionTask(
            frame_id=int(frame_id),
            now_ts=float(now_ts),
            crop_input=[],
            mot_res=None,
            mode='cleanup')
        self._replace_pending_task(task)

    def poll_packets(self):
        packets = []
        while True:
            try:
                packets.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return packets

    def _replace_pending_task(self, task):
        with self._submit_lock:
            while True:
                try:
                    self.task_queue.put_nowait(task)
                    return
                except queue.Full:
                    try:
                        self.task_queue.get_nowait()
                    except queue.Empty:
                        continue

    def _publish_packet(self, packet):
        while True:
            try:
                self.output_queue.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    continue

    def _run(self):
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task.mode == 'stop':
                break

            try:
                packet = self._process_task(task)
            except Exception as exc:
                self.last_error = exc
                self.stop_event.set()
                break

            if packet is not None:
                self._publish_packet(packet)

    def _process_task(self, task):
        with self._state_lock:
            if task.mode == 'cleanup':
                results, events, stale_track_ids = self.postprocessor.cleanup_only(
                    task.now_ts)
                self._drop_predictor_track_cache(stale_track_ids)
            else:
                mot_res = task.mot_res or {}
                boxes = mot_res.get('boxes')
                if boxes is None or len(boxes) == 0 or len(task.crop_input) == 0:
                    results, events, stale_track_ids = self.postprocessor.cleanup_only(
                        task.now_ts)
                    self._drop_predictor_track_cache(stale_track_ids)
                else:
                    raw_result = self.predictor.predict_with_mot(task.crop_input,
                                                                 mot_res)
                    results, events, stale_track_ids = self.postprocessor.process(
                        raw_result, task.now_ts)
                    self._drop_predictor_track_cache(stale_track_ids)

        return AsyncClsActionPacket(
            frame_id=task.frame_id,
            now_ts=task.now_ts,
            mot_res=copy.deepcopy(task.mot_res),
            results=results,
            events=events,
            stale_track_ids=stale_track_ids)

    def _drop_predictor_track_cache(self, track_ids):
        stale_ids = {int(tid) for tid in track_ids}
        if not stale_ids:
            return
        if hasattr(self.predictor, 'result_history'):
            for tid in stale_ids:
                self.predictor.result_history.pop(tid, None)
        if hasattr(self.predictor, 'id_in_last_frame'):
            self.predictor.id_in_last_frame = [
                tid for tid in self.predictor.id_in_last_frame
                if int(tid) not in stale_ids
            ]

    def _reset_predictor_cache(self):
        if hasattr(self.predictor, 'result_history'):
            self.predictor.result_history.clear()
        if hasattr(self.predictor, 'id_in_last_frame'):
            self.predictor.id_in_last_frame = []
        if hasattr(self.predictor, 'skip_frame_cnt'):
            self.predictor.skip_frame_cnt = 0

    @staticmethod
    def _drain_queue(q):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return
