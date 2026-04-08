import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def get_latest_session_lines(log_path: Path):
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    starts = [i for i, line in enumerate(lines) if "PaddleClas is powered by PaddlePaddle" in line]
    if not starts:
        raise RuntimeError("No PaddleClas session found in train.log")
    starts.append(len(lines))
    return lines[starts[-2]:starts[-1]]


def parse_latest_session(lines):
    train_rows = []
    eval_rows = []
    re_train = re.compile(
        r"\[Train\]\[Epoch (\d+)/\d+\]\[Avg\]top1: ([0-9.]+), top3: ([0-9.]+), CELoss: ([0-9.]+), loss: ([0-9.]+)"
    )
    re_eval = re.compile(
        r"\[Eval\]\[Epoch (\d+)\]\[Avg\]CELoss: ([0-9.]+), loss: ([0-9.]+), top1: ([0-9.]+), top3: ([0-9.]+)"
    )
    for line in lines:
        m = re_train.search(line)
        if m:
            train_rows.append(
                {
                    "epoch": int(m.group(1)),
                    "train_top1": float(m.group(2)),
                    "train_top3": float(m.group(3)),
                    "train_loss": float(m.group(4)),
                }
            )
            continue
        m = re_eval.search(line)
        if m:
            eval_rows.append(
                {
                    "epoch": int(m.group(1)),
                    "eval_loss": float(m.group(2)),
                    "eval_top1": float(m.group(4)),
                    "eval_top3": float(m.group(5)),
                }
            )

    if len(train_rows) != len(eval_rows):
        raise RuntimeError(
            f"Unexpected row count mismatch: train={len(train_rows)} eval={len(eval_rows)}"
        )

    merged = []
    eval_by_epoch = {row["epoch"]: row for row in eval_rows}
    for row in train_rows:
        epoch = row["epoch"]
        merged.append({**row, **eval_by_epoch[epoch]})
    return merged


def parse_eval_report(report_path: Path):
    text = report_path.read_text(encoding="utf-8", errors="ignore")

    order_match = re.search(r"Label order \(report\): (.+)", text)
    labels = [x.strip() for x in order_match.group(1).split(",")] if order_match else []

    matrix_match = re.search(
        r"Confusion Matrix \(rows=true, cols=pred\):\s*\n(\[\[.*?\]\])",
        text,
        re.S,
    )
    if not matrix_match:
        raise RuntimeError("Confusion matrix not found in eval_report.txt")
    matrix_lines = matrix_match.group(1).strip().splitlines()
    matrix = []
    for line in matrix_lines:
        nums = [int(x) for x in re.findall(r"\d+", line)]
        matrix.append(nums)
    cm = np.array(matrix, dtype=np.int64)

    metrics = []
    for label in labels:
        m = re.search(
            rf"{re.escape(label)}: P=([0-9.]+) R=([0-9.]+) F1=([0-9.]+)",
            text,
        )
        if m:
            metrics.append(
                {
                    "label": label,
                    "precision": float(m.group(1)),
                    "recall": float(m.group(2)),
                    "f1": float(m.group(3)),
                }
            )

    acc_match = re.search(r"Overall Accuracy: ([0-9.]+)", text)
    overall_acc = float(acc_match.group(1)) if acc_match else None
    return labels, cm, metrics, overall_acc


def save_metrics_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "eval_loss",
                "train_top1",
                "eval_top1",
                "train_top3",
                "eval_top3",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows, out_path: Path):
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    eval_loss = [r["eval_loss"] for r in rows]
    train_top1 = [r["train_top1"] for r in rows]
    eval_top1 = [r["eval_top1"] for r in rows]

    best_row = max(rows, key=lambda r: r["eval_top1"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)

    axes[0].plot(epochs, train_loss, color="#1f77b4", linewidth=2, label="Train loss")
    axes[0].plot(epochs, eval_loss, color="#f39c12", linewidth=2, label="Validation loss")
    axes[0].scatter([best_row["epoch"]], [best_row["eval_loss"]], color="#c0392b", s=35, zorder=5)
    axes[0].annotate(
        f"best epoch={best_row['epoch']}\nval loss={best_row['eval_loss']:.4f}",
        (best_row["epoch"], best_row["eval_loss"]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=8,
    )
    axes[0].set_title("Loss theo epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(epochs, train_top1, color="#2ecc71", linewidth=2, label="Train top-1")
    axes[1].plot(epochs, eval_top1, color="#8e44ad", linewidth=2, label="Validation top-1")
    axes[1].scatter([best_row["epoch"]], [best_row["eval_top1"]], color="#c0392b", s=35, zorder=5)
    axes[1].annotate(
        f"best epoch={best_row['epoch']}\nval top1={best_row['eval_top1']:.4f}",
        (best_row["epoch"], best_row["eval_top1"]),
        textcoords="offset points",
        xytext=(8, -26),
        fontsize=8,
    )
    axes[1].set_title("Top-1 accuracy theo epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.55, 1.0)
    axes[1].legend()

    fig.suptitle("PPLCNet_x1_0 fine-tune tren bo pseudo_labels_crops", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(labels, cm, out_path: Path):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=180)
    im = ax.imshow(cm, cmap="YlGnBu")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix tren tap validation")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.45 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_class_metrics(metrics, out_path: Path):
    labels = [m["label"] for m in metrics]
    precision = [m["precision"] for m in metrics]
    recall = [m["recall"] for m in metrics]
    f1 = [m["f1"] for m in metrics]

    x = np.arange(len(labels))
    width = 0.24

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=180)
    ax.bar(x - width, precision, width, label="Precision", color="#3498db")
    ax.bar(x, recall, width, label="Recall", color="#2ecc71")
    ax.bar(x + width, f1, width, label="F1-score", color="#9b59b6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Chi so theo lop tren tap validation")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot summary figures for one final PPLCNet training session.")
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--eval-report", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "report_figures"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_log = Path(args.train_log)
    eval_report = Path(args.eval_report)
    out_dir = Path(args.output_dir)

    if not train_log.exists():
        raise FileNotFoundError(f"Missing train log: {train_log}")
    if not eval_report.exists():
        raise FileNotFoundError(f"Missing eval report: {eval_report}")

    out_dir.mkdir(parents=True, exist_ok=True)
    latest_lines = get_latest_session_lines(train_log)
    rows = parse_latest_session(latest_lines)
    labels, cm, metrics, overall_acc = parse_eval_report(eval_report)

    save_metrics_csv(rows, out_dir / "lcnet_final_session_metrics.csv")
    plot_curves(rows, out_dir / "lcnet_final_session_curves.png")
    plot_confusion_matrix(labels, cm, out_dir / "lcnet_final_session_confusion_matrix.png")
    plot_class_metrics(metrics, out_dir / "lcnet_final_session_class_metrics.png")

    best_row = max(rows, key=lambda r: r["eval_top1"])
    print("overall_acc", overall_acc)
    print("best_epoch", best_row["epoch"])
    print("best_eval_top1", best_row["eval_top1"])
    print("best_eval_loss", best_row["eval_loss"])
    print("final_eval_top1", rows[-1]["eval_top1"])
    print("final_eval_loss", rows[-1]["eval_loss"])


if __name__ == "__main__":
    main()
