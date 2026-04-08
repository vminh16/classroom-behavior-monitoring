import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(ROOT / "deploy" / "pptracking" / "python"))

from mot_sde_infer import SDE_Detector  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark OCSORT/BOTSORT on the classroom pipeline with "
            "project-specific tracking stability metrics."
        ))
    parser.add_argument(
        "--model_dir",
        default=str(ROOT / "output_inference" / "picodet_m_416_classroom"))
    parser.add_argument(
        "--video_file", default=str(ROOT / "test_data" / "data.mp4"))
    parser.add_argument(
        "--base_tracker_config",
        default=str(ROOT / "deploy" / "pipeline" / "config" /
                    "tracker_config.yml"))
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--run_mode", default="paddle")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip_frame_num", type=int, default=2)
    parser.add_argument("--warmup_frame", type=int, default=10)
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["BOTSORTTracker", "OCSORTTracker"])
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "output" / "tracker_benchmark"))
    parser.add_argument("--cls_skip_frame_num", type=int, default=2)
    parser.add_argument("--behavior_window_size", type=int, default=60)
    parser.add_argument("--sleep_warn_seconds", type=float, default=5.0)
    parser.add_argument("--phone_warn_seconds", type=float, default=6.0)
    return parser.parse_args()


def make_tracker_config(base_cfg_path, tracker_type, output_dir):
    with open(base_cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["type"] = tracker_type
    out_path = Path(output_dir) / f"tracker_config_{tracker_type.lower()}.yml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return out_path


def get_video_meta(video_file):
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def parse_mot_txt(result_txt):
    rows = []
    with open(result_txt, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items = line.split(",")
            if len(items) < 7:
                continue
            frame_id = int(float(items[0]))
            track_id = int(float(items[1]))
            x1 = float(items[2])
            y1 = float(items[3])
            w = float(items[4])
            h = float(items[5])
            score = float(items[6])
            rows.append((frame_id, track_id, x1, y1, w, h, score))
    return rows


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def summarize_tracks(rows, fps, total_frames, cls_skip_frame_num,
                     behavior_window_size, sleep_warn_seconds,
                     phone_warn_seconds):
    by_track = {}
    frame_track_counts = {}
    for frame_id, track_id, x1, y1, w, h, score in rows:
        info = by_track.setdefault(track_id, {
            "frames": [],
            "scores": [],
            "areas": [],
        })
        info["frames"].append(frame_id)
        info["scores"].append(score)
        info["areas"].append(max(0.0, w * h))
        frame_track_counts[frame_id] = frame_track_counts.get(frame_id, 0) + 1

    track_lengths = []
    track_spans = []
    track_coverages = []
    avg_scores = []
    avg_areas = []
    gap_counts = []
    short_lt_15 = 0
    short_lt_30 = 0
    new_id_rate = len(by_track) / max(total_frames, 1) * 100.0

    cls_window_frames = int(behavior_window_size * cls_skip_frame_num)
    sleep_warn_frames = int(math.ceil(sleep_warn_seconds * fps))
    phone_warn_frames = int(math.ceil(phone_warn_seconds * fps))

    eligible_cls_tracks = 0
    eligible_sleep_tracks = 0
    eligible_phone_tracks = 0
    eligible_cls_observations = 0
    eligible_sleep_observations = 0
    eligible_phone_observations = 0

    for info in by_track.values():
        frames = sorted(set(info["frames"]))
        obs = len(frames)
        span = (frames[-1] - frames[0] + 1) if frames else 0
        diffs = np.diff(frames) if len(frames) > 1 else []
        gaps = int(np.sum(np.asarray(diffs) > 1)) if len(frames) > 1 else 0
        coverage = (obs / span) if span > 0 else 0.0

        track_lengths.append(obs)
        track_spans.append(span)
        track_coverages.append(coverage)
        avg_scores.append(float(np.mean(info["scores"])))
        avg_areas.append(float(np.mean(info["areas"])))
        gap_counts.append(gaps)

        if obs < 15:
            short_lt_15 += 1
        if obs < 30:
            short_lt_30 += 1
        if obs >= cls_window_frames:
            eligible_cls_tracks += 1
            eligible_cls_observations += obs
        if obs >= sleep_warn_frames:
            eligible_sleep_tracks += 1
            eligible_sleep_observations += obs
        if obs >= phone_warn_frames:
            eligible_phone_tracks += 1
            eligible_phone_observations += obs

    total_observations = len(rows)
    avg_tracks_per_frame = (total_observations / max(total_frames, 1))

    return {
        "total_observations": total_observations,
        "unique_tracks": len(by_track),
        "avg_tracks_per_frame": avg_tracks_per_frame,
        "new_track_ids_per_100_frames": new_id_rate,
        "track_length_mean_frames": float(np.mean(track_lengths))
        if track_lengths else 0.0,
        "track_length_median_frames": float(np.median(track_lengths))
        if track_lengths else 0.0,
        "track_length_p90_frames": percentile(track_lengths, 90),
        "track_length_max_frames": max(track_lengths) if track_lengths else 0,
        "track_span_mean_frames": float(np.mean(track_spans))
        if track_spans else 0.0,
        "coverage_mean": float(np.mean(track_coverages))
        if track_coverages else 0.0,
        "coverage_median": float(np.median(track_coverages))
        if track_coverages else 0.0,
        "gap_count_mean": float(np.mean(gap_counts)) if gap_counts else 0.0,
        "avg_track_score": float(np.mean(avg_scores)) if avg_scores else 0.0,
        "avg_track_area": float(np.mean(avg_areas)) if avg_areas else 0.0,
        "short_track_ratio_lt15": short_lt_15 / max(len(by_track), 1),
        "short_track_ratio_lt30": short_lt_30 / max(len(by_track), 1),
        "eligible_cls_window_tracks": eligible_cls_tracks,
        "eligible_cls_window_track_ratio":
        eligible_cls_tracks / max(len(by_track), 1),
        "eligible_cls_window_observation_share":
        eligible_cls_observations / max(total_observations, 1),
        "eligible_sleep_warn_tracks": eligible_sleep_tracks,
        "eligible_sleep_warn_track_ratio":
        eligible_sleep_tracks / max(len(by_track), 1),
        "eligible_sleep_warn_observation_share":
        eligible_sleep_observations / max(total_observations, 1),
        "eligible_phone_warn_tracks": eligible_phone_tracks,
        "eligible_phone_warn_track_ratio":
        eligible_phone_tracks / max(len(by_track), 1),
        "eligible_phone_warn_observation_share":
        eligible_phone_observations / max(total_observations, 1),
        "cls_window_frames": cls_window_frames,
        "sleep_warn_frames": sleep_warn_frames,
        "phone_warn_frames": phone_warn_frames,
        "frame_track_count_mean": float(np.mean(list(frame_track_counts.values())))
        if frame_track_counts else 0.0,
        "frame_track_count_p95": percentile(list(frame_track_counts.values()), 95),
    }


def run_tracker(tracker_type, args, video_meta):
    out_dir = Path(args.output_dir) / tracker_type.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker_cfg_path = make_tracker_config(args.base_tracker_config, tracker_type,
                                          out_dir)

    detector = SDE_Detector(
        model_dir=args.model_dir,
        tracker_config=str(tracker_cfg_path),
        device=args.device,
        run_mode=args.run_mode,
        batch_size=1,
        output_dir=str(out_dir),
        threshold=args.threshold,
        save_images=False,
        save_mot_txts=True,
        skip_frame_num=args.skip_frame_num,
        warmup_frame=args.warmup_frame,
    )
    detector.predict_video(str(args.video_file), -1)
    timing = detector.det_times.report(average=True)

    result_txt = out_dir / Path(args.video_file).with_suffix(".txt").name
    rows = parse_mot_txt(result_txt)
    track_stats = summarize_tracks(
        rows=rows,
        fps=video_meta["fps"],
        total_frames=video_meta["frame_count"],
        cls_skip_frame_num=args.cls_skip_frame_num,
        behavior_window_size=args.behavior_window_size,
        sleep_warn_seconds=args.sleep_warn_seconds,
        phone_warn_seconds=args.phone_warn_seconds,
    )

    total_time_s = float(timing.get("total_time_s", 0.0))
    img_num = int(timing.get("img_num", 0))
    avg_latency_ms = (total_time_s * 1000.0 / max(img_num, 1))
    fps = (img_num / total_time_s) if total_time_s > 0 else 0.0

    summary = {
        "tracker_type": tracker_type,
        "video_file": str(args.video_file),
        "tracker_config": str(tracker_cfg_path),
        "timing": {
            "img_num": img_num,
            "avg_latency_ms": avg_latency_ms,
            "fps": fps,
            "preprocess_ms_per_frame":
            float(timing.get("preprocess_time_s", 0.0) * 1000.0),
            "inference_ms_per_frame":
            float(timing.get("inference_time_s", 0.0) * 1000.0),
            "postprocess_ms_per_frame":
            float(timing.get("postprocess_time_s", 0.0) * 1000.0),
            "tracking_ms_per_frame":
            float(timing.get("tracking_time_s", 0.0) * 1000.0),
        },
        "tracking_stats": track_stats,
        "mot_result_txt": str(result_txt),
    }
    return summary


def build_report(results):
    lines = []
    lines.append("Tracker benchmark for classroom monitoring pipeline")
    lines.append("")
    for result in results:
        timing = result["timing"]
        stats = result["tracking_stats"]
        lines.append(f"[{result['tracker_type']}]")
        lines.append(
            f"avg_latency_ms: {timing['avg_latency_ms']:.4f}, fps: {timing['fps']:.4f}"
        )
        lines.append(
            "module_ms_per_frame: "
            f"pre={timing['preprocess_ms_per_frame']:.4f}, "
            f"infer={timing['inference_ms_per_frame']:.4f}, "
            f"post={timing['postprocess_ms_per_frame']:.4f}, "
            f"track={timing['tracking_ms_per_frame']:.4f}"
        )
        lines.append(
            f"unique_tracks: {stats['unique_tracks']}, "
            f"avg_tracks_per_frame: {stats['avg_tracks_per_frame']:.4f}, "
            f"new_track_ids_per_100_frames: {stats['new_track_ids_per_100_frames']:.4f}"
        )
        lines.append(
            f"track_length_mean_frames: {stats['track_length_mean_frames']:.4f}, "
            f"median: {stats['track_length_median_frames']:.4f}, "
            f"p90: {stats['track_length_p90_frames']:.4f}, "
            f"max: {stats['track_length_max_frames']}"
        )
        lines.append(
            f"coverage_mean: {stats['coverage_mean']:.4f}, "
            f"coverage_median: {stats['coverage_median']:.4f}, "
            f"gap_count_mean: {stats['gap_count_mean']:.4f}"
        )
        lines.append(
            f"short_track_ratio_lt15: {stats['short_track_ratio_lt15']:.4f}, "
            f"short_track_ratio_lt30: {stats['short_track_ratio_lt30']:.4f}"
        )
        lines.append(
            f"eligible_cls_window_track_ratio(>={stats['cls_window_frames']} frames): "
            f"{stats['eligible_cls_window_track_ratio']:.4f}, "
            f"observation_share: {stats['eligible_cls_window_observation_share']:.4f}"
        )
        lines.append(
            f"eligible_sleep_warn_track_ratio(>={stats['sleep_warn_frames']} frames): "
            f"{stats['eligible_sleep_warn_track_ratio']:.4f}, "
            f"observation_share: {stats['eligible_sleep_warn_observation_share']:.4f}"
        )
        lines.append(
            f"eligible_phone_warn_track_ratio(>={stats['phone_warn_frames']} frames): "
            f"{stats['eligible_phone_warn_track_ratio']:.4f}, "
            f"observation_share: {stats['eligible_phone_warn_observation_share']:.4f}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_meta = get_video_meta(args.video_file)
    results = []
    for tracker_type in args.trackers:
        print(f"Running tracker benchmark: {tracker_type}")
        results.append(run_tracker(tracker_type, args, video_meta))

    report_text = build_report(results)
    report_path = output_dir / "tracker_benchmark_report.txt"
    json_path = output_dir / "tracker_benchmark_summary.json"
    report_path.write_text(report_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "video_meta": video_meta,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(report_text)
    print(f"saved_report: {report_path}")
    print(f"saved_json: {json_path}")


if __name__ == "__main__":
    main()
