"""
Behavior state machine for per-track classroom monitoring.

Input per update:
  - stable behavior label/class of one track_id
  - current timestamp (seconds)
  - optional confidence score

Output per update:
  - current state (NORMAL/WARN/ALERT variant)
  - generated events (if any)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


NORMAL = "NORMAL"
SLEEPING_WARN = "SLEEPING_WARN"
SLEEPING_ALERT = "SLEEPING_ALERT"
PHONE_WARN = "PHONE_WARN"
PHONE_ALERT = "PHONE_ALERT"

_WARN_LEVEL = "warn"
_ALERT_LEVEL = "alert"


@dataclass
class TrackState:
    state: str = NORMAL
    active_behavior: Optional[str] = None  # sleeping | phone
    behavior_start_time: Optional[float] = None
    last_alert_time: Dict[str, float] = field(default_factory=dict)
    last_seen_time: float = 0.0


class BehaviorStateMachine:
    """
    Per-track state machine with duration thresholds and cooldown policy.
    """

    def __init__(
            self,
            normal_id: int = 0,
            sleeping_id: int = 2,
            phone_id: int = 1,
            sleep_warn_seconds: float = 5.0,
            sleep_alert_seconds: float = 12.0,
            phone_warn_seconds: float = 6.0,
            phone_alert_seconds: float = 15.0,
            warn_cooldown_seconds: float = 8.0,
            alert_cooldown_seconds: float = 15.0,
            stale_track_seconds: float = 2.0,
            min_confidence: Optional[float] = None,
            emit_end_event: bool = True):
        self.normal_id = int(normal_id)
        self.sleeping_id = int(sleeping_id)
        self.phone_id = int(phone_id)

        self.sleep_warn_seconds = float(sleep_warn_seconds)
        self.sleep_alert_seconds = float(sleep_alert_seconds)
        self.phone_warn_seconds = float(phone_warn_seconds)
        self.phone_alert_seconds = float(phone_alert_seconds)
        self.warn_cooldown_seconds = float(warn_cooldown_seconds)
        self.alert_cooldown_seconds = float(alert_cooldown_seconds)
        self.stale_track_seconds = float(stale_track_seconds)
        self.min_confidence = None if min_confidence is None else float(
            min_confidence)
        self.emit_end_event = bool(emit_end_event)

        self.tracks: Dict[int, TrackState] = {}

    @staticmethod
    def _elapsed(now: float, start_time: Optional[float]) -> float:
        if start_time is None:
            return 0.0
        return max(0.0, float(now) - float(start_time))

    def update(
            self,
            track_id: int,
            behavior: Union[int, str],
            now: float,
            score: Optional[float] = None) -> Tuple[str, List[dict]]:
        """
        Update one track and return (state, events).
        """
        tid = int(track_id)
        now = float(now)
        score_val = None if score is None else float(score)
        b = self._canonical_behavior(behavior)
        ctx = self.tracks.get(tid)
        if ctx is None:
            ctx = TrackState(last_seen_time=now)
            self.tracks[tid] = ctx
        ctx.last_seen_time = now

        events: List[dict] = []

        if self.min_confidence is not None and score_val is not None:
            if score_val < self.min_confidence:
                return ctx.state, events

        if b == "normal":
            if ctx.active_behavior is not None and self.emit_end_event:
                duration = self._elapsed(now, ctx.behavior_start_time)
                events.append(
                    self._build_event(
                        track_id=tid,
                        behavior=ctx.active_behavior,
                        level="end",
                        event_kind="end",
                        trigger_time=now,
                        start_time=ctx.behavior_start_time,
                        duration=duration,
                        state=ctx.state,
                        score=score_val))
            self._reset_to_normal(ctx)
            return ctx.state, events

        # New behavior session
        if ctx.active_behavior != b or ctx.behavior_start_time is None:
            if ctx.active_behavior is not None and self.emit_end_event:
                duration = self._elapsed(now, ctx.behavior_start_time)
                events.append(
                    self._build_event(
                        track_id=tid,
                        behavior=ctx.active_behavior,
                        level="end",
                        event_kind="switch_end",
                        trigger_time=now,
                        start_time=ctx.behavior_start_time,
                        duration=duration,
                        state=ctx.state,
                        score=score_val))
            ctx.active_behavior = b
            ctx.behavior_start_time = now
            ctx.state = NORMAL

        duration = self._elapsed(now, ctx.behavior_start_time)
        target_state, target_level = self._state_from_duration(b, duration)
        prev_state = ctx.state
        ctx.state = target_state

        # Enter level event
        if prev_state != target_state and target_state != NORMAL:
            if self._allow_alert(ctx, b, target_level, now):
                events.append(
                    self._build_event(
                        track_id=tid,
                        behavior=b,
                        level=target_level,
                        event_kind="enter",
                        trigger_time=now,
                        start_time=ctx.behavior_start_time,
                        duration=duration,
                        state=target_state,
                        score=score_val))
                ctx.last_alert_time[f"{b}:{target_level}"] = now
        # Repeat alert event while staying in ALERT, gated by cooldown
        elif target_level == _ALERT_LEVEL and target_state != NORMAL:
            if self._allow_alert(ctx, b, _ALERT_LEVEL, now):
                events.append(
                    self._build_event(
                        track_id=tid,
                        behavior=b,
                        level=_ALERT_LEVEL,
                        event_kind="repeat",
                        trigger_time=now,
                        start_time=ctx.behavior_start_time,
                        duration=duration,
                        state=target_state,
                        score=score_val))
                ctx.last_alert_time[f"{b}:{_ALERT_LEVEL}"] = now

        return ctx.state, events

    def cleanup_stale(self, now: float) -> List[dict]:
        """
        Remove stale track_ids and optionally emit closing events.
        """
        now = float(now)
        expired = []
        for tid, ctx in self.tracks.items():
            if now - ctx.last_seen_time > self.stale_track_seconds:
                expired.append((tid, ctx))

        events: List[dict] = []
        for tid, ctx in expired:
            if ctx.active_behavior is not None and self.emit_end_event:
                duration = self._elapsed(now, ctx.behavior_start_time)
                events.append(
                    self._build_event(
                        track_id=tid,
                        behavior=ctx.active_behavior,
                        level="end",
                        event_kind="timeout_end",
                        trigger_time=now,
                        start_time=ctx.behavior_start_time,
                        duration=duration,
                        state=ctx.state,
                        score=None))
            del self.tracks[tid]
        return events

    def get_state(self, track_id: int) -> str:
        ctx = self.tracks.get(int(track_id))
        return ctx.state if ctx is not None else NORMAL

    def snapshot_states(self) -> Dict[int, str]:
        return {tid: ctx.state for tid, ctx in self.tracks.items()}

    def _state_from_duration(self, behavior: str,
                             duration: float) -> Tuple[str, Optional[str]]:
        if behavior == "sleeping":
            if duration >= self.sleep_alert_seconds:
                return SLEEPING_ALERT, _ALERT_LEVEL
            if duration >= self.sleep_warn_seconds:
                return SLEEPING_WARN, _WARN_LEVEL
            return NORMAL, None
        if behavior == "phone":
            if duration >= self.phone_alert_seconds:
                return PHONE_ALERT, _ALERT_LEVEL
            if duration >= self.phone_warn_seconds:
                return PHONE_WARN, _WARN_LEVEL
            return NORMAL, None
        return NORMAL, None

    def _allow_alert(self, ctx: TrackState, behavior: str, level: str,
                     now: float) -> bool:
        key = f"{behavior}:{level}"
        last = ctx.last_alert_time.get(key)
        if last is None:
            return True
        cooldown = self.warn_cooldown_seconds if level == _WARN_LEVEL else self.alert_cooldown_seconds
        return (now - last) >= cooldown

    def _reset_to_normal(self, ctx: TrackState) -> None:
        ctx.state = NORMAL
        ctx.active_behavior = None
        ctx.behavior_start_time = None
        ctx.last_alert_time.clear()

    def _canonical_behavior(self, behavior: Union[int, str]) -> str:
        if isinstance(behavior, str):
            text = behavior.strip().lower()
            if "sleep" in text:
                return "sleeping"
            if "phone" in text:
                return "phone"
            return "normal"
        try:
            cid = int(behavior)
        except Exception:
            return "normal"
        if cid == self.sleeping_id:
            return "sleeping"
        if cid == self.phone_id:
            return "phone"
        return "normal"

    def _build_event(
            self,
            track_id: int,
            behavior: str,
            level: str,
            event_kind: str,
            trigger_time: float,
            start_time: Optional[float],
            duration: float,
            state: str,
            score: Optional[float]) -> dict:
        level_upper = level.upper()
        behavior_upper = behavior.upper()
        display = f"{behavior_upper}_{level_upper}"
        if level in (_WARN_LEVEL, _ALERT_LEVEL):
            display = f"{display} {duration:.1f}s"
        return {
            "track_id": int(track_id),
            "event_type": behavior,
            "alert_level": level,
            "event_kind": event_kind,
            "start_time": start_time,
            "trigger_time": float(trigger_time),
            "duration_at_trigger": float(duration),
            "state": state,
            "score": score,
            "display_text": display,
        }
