"""
Temporal voting behavior filter for multi-class tracking results.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Optional


class BehaviorFilter:
    def __init__(self,
                 window_size=60,
                 majority_ratio=0.6,
                 sleep_enter_ratio=0.7,
                 sleep_exit_ratio=0.6,
                 sleeping_id=2,
                 normal_id=0):
        self.window_size = window_size
        self.majority_ratio = majority_ratio
        self.sleep_enter_ratio = sleep_enter_ratio
        self.sleep_exit_ratio = sleep_exit_ratio
        self.sleeping_id = sleeping_id
        self.normal_id = normal_id
        self.history = defaultdict(lambda: deque(maxlen=self.window_size))
        self.last_state = {}

    def update(self, track_id: int, cls_id: int, score: Optional[float]):
        self.history[track_id].append((int(cls_id), score))
        return self.vote(track_id)

    def vote(self, track_id: int):
        seq = self.history.get(track_id)
        if not seq or len(seq) < self.window_size:
            return None
        total = float(len(seq))
        counts = Counter([c for c, _ in seq])
        voted_cls, voted_cnt = counts.most_common(1)[0]
        voted_ratio = voted_cnt / total

        sleeping_ratio = counts.get(self.sleeping_id, 0) / total
        normal_ratio = counts.get(self.normal_id, 0) / total
        last_state = self.last_state.get(track_id)
        last_cls = last_state[0] if last_state else None

        next_cls = None
        if last_cls == self.sleeping_id:
            if normal_ratio > self.sleep_exit_ratio:
                next_cls = self.normal_id
            elif sleeping_ratio > self.sleep_enter_ratio:
                next_cls = self.sleeping_id
            elif voted_ratio >= self.majority_ratio:
                next_cls = voted_cls
            else:
                next_cls = last_cls
        else:
            if sleeping_ratio > self.sleep_enter_ratio:
                next_cls = self.sleeping_id
            elif voted_ratio >= self.majority_ratio:
                next_cls = voted_cls
            elif last_cls is not None:
                next_cls = last_cls

        if next_cls is None:
            return None

        scores = [s for c, s in seq if c == next_cls and s is not None]
        voted_score = sum(scores) / len(scores) if scores else None
        self.last_state[track_id] = (next_cls, voted_score)
        return self.last_state[track_id]

    def get_last(self, track_id: int):
        return self.last_state.get(track_id)
