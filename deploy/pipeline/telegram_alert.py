import queue
import threading
import time
import uuid
import urllib.request

import cv2


def _encode_multipart(fields, files):
    boundary = uuid.uuid4().hex
    chunks = []

    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                "utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    for name, filename, content_type, data in files:
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            .encode("utf-8"))
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        chunks.append(data)
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return boundary, b"".join(chunks)


class TelegramAlertSink:
    def __init__(self,
                 bot_token,
                 chat_id,
                 send_rules=None,
                 queue_size=64,
                 timeout_seconds=10.0,
                 jpeg_quality=85,
                 send_retries=3,
                 retry_delay_seconds=1.0,
                 max_long_edge=1280):
        self.enabled = True
        self.url = "https://api.telegram.org/bot{}/sendPhoto".format(
            str(bot_token).strip())
        self.chat_id = str(chat_id).strip()
        self.send_rules = {
            "warn": {"enter"},
            "alert": {"enter", "repeat"},
        }
        if isinstance(send_rules, dict):
            for level, kinds in send_rules.items():
                self.send_rules[str(level).strip().lower()] = {
                    str(kind).strip().lower()
                    for kind in kinds
                }
        self.timeout_seconds = float(timeout_seconds)
        self.jpeg_quality = int(jpeg_quality)
        self.send_retries = max(1, int(send_retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self.max_long_edge = max(0, int(max_long_edge or 0))
        self.queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.stats_lock = threading.Lock()
        self._stats = {
            "accepted": 0,
            "enqueued": 0,
            "sent": 0,
            "failed": 0,
            "retried": 0,
            "queue_dropped": 0,
            "encode_failed": 0,
        }
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    @classmethod
    def from_cfg(cls, cfg):
        bot_token = cfg.get("bot_token")
        chat_id = cfg.get("chat_id")
        if not bot_token or not chat_id:
            raise ValueError(
                "TELEGRAM_ALERT requires bot_token and chat_id in config.")
        return cls(
            bot_token=bot_token,
            chat_id=chat_id,
            send_rules=cfg.get("send_rules"),
            queue_size=cfg.get("queue_size", 64),
            timeout_seconds=cfg.get("timeout_seconds", 10.0),
            jpeg_quality=cfg.get("jpeg_quality", 85),
            send_retries=cfg.get("send_retries", 3),
            retry_delay_seconds=cfg.get("retry_delay_seconds", 1.0),
            max_long_edge=cfg.get("max_long_edge", 1280))

    def accepts(self, event):
        level = str(event.get("alert_level", "")).strip().lower()
        kind = str(event.get("event_kind", "")).strip().lower()
        if not level or not kind:
            return False
        allowed_kinds = self.send_rules.get(level, set())
        return kind in allowed_kinds

    def enqueue(self, event, image_bgr, source_name, frame_id):
        if not self.accepts(event):
            return False
        self._bump_stat("accepted")

        image_bgr = self._resize_image(image_bgr)
        ok, encoded = cv2.imencode(
            ".jpg", image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            self._bump_stat("encode_failed")
            return False

        item = {
            "event": dict(event),
            "source_name": str(source_name or "camera"),
            "frame_id": int(frame_id),
            "photo_name": "{}_frame_{:06d}.jpg".format(
                str(source_name or "camera"), int(frame_id)),
            "photo_bytes": encoded.tobytes(),
        }

        try:
            self.queue.put_nowait(item)
            self._bump_stat("enqueued")
            return True
        except queue.Full:
            self._bump_stat("queue_dropped")
            return False

    def flush(self):
        self.queue.join()

    def stats(self):
        with self.stats_lock:
            return dict(self._stats)

    def _bump_stat(self, key, delta=1):
        with self.stats_lock:
            self._stats[key] = self._stats.get(key, 0) + int(delta)

    def _resize_image(self, image_bgr):
        if image_bgr is None or self.max_long_edge <= 0:
            return image_bgr
        height, width = image_bgr.shape[:2]
        long_edge = max(height, width)
        if long_edge <= self.max_long_edge:
            return image_bgr
        scale = float(self.max_long_edge) / float(long_edge)
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        return cv2.resize(
            image_bgr, (target_width, target_height),
            interpolation=cv2.INTER_AREA)

    def _worker(self):
        while True:
            item = self.queue.get()
            try:
                self._send(item)
            except Exception as exc:
                self._bump_stat("failed")
                print("[TelegramAlert] send failed: {}".format(exc))
            finally:
                self.queue.task_done()

    def _send(self, item):
        event = item["event"]
        caption = self._build_caption(
            event=event,
            source_name=item["source_name"],
            frame_id=item["frame_id"])
        fields = {
            "chat_id": self.chat_id,
            "caption": caption,
        }
        files = [("photo", item["photo_name"], "image/jpeg",
                  item["photo_bytes"])]
        boundary, body = _encode_multipart(fields, files)
        last_exc = None
        for attempt in range(self.send_retries):
            req = urllib.request.Request(
                self.url,
                data=body,
                method="POST",
                headers={
                    "Content-Type":
                    "multipart/form-data; boundary={}".format(boundary)
                })
            try:
                with urllib.request.urlopen(
                        req, timeout=self.timeout_seconds) as resp:
                    resp.read()
                self._bump_stat("sent")
                return
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= self.send_retries:
                    break
                self._bump_stat("retried")
                time.sleep(self.retry_delay_seconds * float(attempt + 1))
        raise last_exc

    def _build_caption(self, event, source_name, frame_id):
        event_type = str(event.get("event_type", "unknown")).strip().upper()
        level = str(event.get("alert_level", "unknown")).strip().upper()
        kind = str(event.get("event_kind", "unknown")).strip().upper()
        track_id = event.get("track_id", "n/a")
        state = str(event.get("state", "n/a")).strip()
        duration = float(event.get("duration_at_trigger", 0.0))
        trigger_time = float(event.get("trigger_time", 0.0))
        text = (
            "[Classroom Alert]\n"
            "Source: {}\n"
            "Frame: {}\n"
            "Track ID: {}\n"
            "Behavior: {}\n"
            "Level: {}\n"
            "Kind: {}\n"
            "Duration: {:.1f}s\n"
            "Video Time: {:.1f}s\n"
            "State: {}".format(source_name, frame_id, track_id, event_type,
                               level, kind, duration, trigger_time, state))
        return text[:900]
