import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(ROOT / "deploy" / "pipeline"))

from pphuman.action_infer import ClsActionRecognizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LCNet classifier accuracy across crop modes.")
    parser.add_argument(
        "--model_dir",
        default=str(ROOT / "output_inference" / "pplcnet_behavior"))
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--list_file", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_batches", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["full", "upper:0.5"],
        help="Examples: full upper:0.5 upper:0.65")
    parser.add_argument(
        "--report_path",
        default=str(ROOT / "output" / "cls_crop_benchmark" /
                    "val_crop_mode_report.txt"))
    return parser.parse_args()


def parse_label_list(dataset_root):
    label_file = Path(dataset_root) / "label_list.txt"
    with open(label_file, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def parse_samples(dataset_root, list_file):
    samples = []
    with open(list_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(" ", 1)
            samples.append((str(Path(dataset_root) / rel_path), int(label)))
    return samples


def parse_mode(mode_text):
    raw = str(mode_text).strip().lower()
    if raw in ("full", "full_body"):
        return {
            "name": "full",
            "crop_mode": "full",
            "upper_crop_ratio": 1.0,
            "display_name": "full",
        }
    if raw.startswith("upper"):
        ratio = 0.5
        if ":" in raw:
            ratio = float(raw.split(":", 1)[1])
        return {
            "name": raw,
            "crop_mode": "upper",
            "upper_crop_ratio": ratio,
            "display_name": f"upper({ratio:.2f})",
        }
    raise ValueError(f"Unsupported mode: {mode_text}")


def format_metrics(confusion, labels):
    lines = []
    total = confusion.sum()
    correct = int(np.trace(confusion))
    accuracy = correct / total if total else 0.0
    lines.append(f"accuracy: {accuracy:.6f} ({correct}/{total})")
    lines.append("confusion (rows=true, cols=pred):")
    lines.append(str(confusion))
    lines.append("per-class:")
    for idx, label in enumerate(labels):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - tp)
        fn = float(confusion[idx, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (
            precision + recall) > 0 else 0.0
        lines.append(
            f"  {label}: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    return accuracy, lines


def run_batch(predictor, batch_rgb):
    predict_input = predictor.crop_person_region(batch_rgb)
    start = time.perf_counter()
    outputs = predictor.predict_image(predict_input, visual=False)["output"]
    elapsed = time.perf_counter() - start
    return outputs, elapsed


def warmup_predictor(predictor, samples, batch_size, warmup_batches):
    if warmup_batches <= 0:
        return
    batch_rgb = []
    warmed = 0
    for img_path, _ in samples:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        batch_rgb.append(image_rgb)
        if len(batch_rgb) >= batch_size:
            predictor.predict_image(
                predictor.crop_person_region(batch_rgb), visual=False)
            warmed += 1
            batch_rgb = []
            if warmed >= warmup_batches:
                return
    if batch_rgb:
        predictor.predict_image(
            predictor.crop_person_region(batch_rgb), visual=False)


def evaluate_mode(mode_cfg, args, labels, samples):
    predictor = ClsActionRecognizer(
        model_dir=args.model_dir,
        device=args.device,
        batch_size=args.batch_size,
        threshold=0.0,
        display_frames=1,
        skip_frame_num=0,
        crop_mode=mode_cfg["crop_mode"],
        upper_crop_ratio=mode_cfg["upper_crop_ratio"])
    warmup_predictor(predictor, samples, args.batch_size, args.warmup_batches)

    confusion = np.zeros((len(labels), len(labels)), dtype=np.int32)
    skipped = []
    batch_rgb = []
    batch_labels = []
    predict_time = 0.0
    predicted_scores = []

    def flush_batch():
        nonlocal predict_time, batch_rgb, batch_labels
        if not batch_rgb:
            return
        outputs, elapsed = run_batch(predictor, batch_rgb)
        predict_time += elapsed
        for true_label, scores in zip(batch_labels, outputs):
            pred = int(np.argmax(scores))
            confusion[true_label, pred] += 1
            predicted_scores.append(float(np.max(scores)))
        batch_rgb = []
        batch_labels = []

    for img_path, true_label in samples:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            skipped.append(img_path)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        batch_rgb.append(image_rgb)
        batch_labels.append(true_label)
        if len(batch_rgb) >= args.batch_size:
            flush_batch()

    flush_batch()

    accuracy, metric_lines = format_metrics(confusion, labels)
    valid_count = int(confusion.sum())
    ms_per_image = (predict_time * 1000.0 / valid_count) if valid_count else 0.0
    return {
        "mode": mode_cfg["display_name"],
        "accuracy": accuracy,
        "valid_count": valid_count,
        "skipped_count": len(skipped),
        "predict_time_s": predict_time,
        "ms_per_image": ms_per_image,
        "confusion": confusion,
        "metric_lines": metric_lines,
    }


def build_report(results):
    lines = []
    lines.append("LCNet crop-mode benchmark")
    lines.append("")
    for result in results:
        lines.append(f"[{result['mode']}]")
        lines.append(f"valid_images: {result['valid_count']}")
        lines.append(f"skipped_images: {result['skipped_count']}")
        lines.append(f"predict_time_s: {result['predict_time_s']:.4f}")
        lines.append(f"ms_per_image: {result['ms_per_image']:.4f}")
        lines.extend(result["metric_lines"])
        lines.append("")
    if len(results) >= 2:
        base = results[0]
        for other in results[1:]:
            delta_acc = other["accuracy"] - base["accuracy"]
            delta_ms = other["ms_per_image"] - base["ms_per_image"]
            lines.append(
                f"delta {other['mode']} vs {base['mode']}: accuracy={delta_acc:+.6f}, ms_per_image={delta_ms:+.4f}"
            )
    return "\n".join(lines).strip() + "\n"


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    list_file = Path(args.list_file)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset_root: {dataset_root}")
    if not list_file.exists():
        raise FileNotFoundError(f"Missing list_file: {list_file}")
    labels = parse_label_list(dataset_root)
    samples = parse_samples(dataset_root, list_file)
    modes = [parse_mode(mode) for mode in args.modes]

    results = []
    for mode_cfg in modes:
        print(f"Running mode: {mode_cfg['display_name']}")
        results.append(evaluate_mode(mode_cfg, args, labels, samples))

    report_text = build_report(results)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"saved_report: {report_path}")


if __name__ == "__main__":
    main()
