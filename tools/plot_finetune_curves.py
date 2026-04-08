import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate plots. "
            "Please install it in the active environment."
        ) from exc
    return plt


def _read_picodet_epoch_metrics(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            train_loss = _to_float(row.get("train_loss_mean"))
            val_loss = _to_float(row.get("val_loss"))
            val_metric = _to_float(row.get("val_metric"))
            rows.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metric": val_metric,
            })
    return rows


def _read_lcnet_train_log(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    train_avg_re = re.compile(
        r"\[Train\]\[Epoch (\d+)/(\d+)\]\[Avg\]top1: ([0-9.]+), top3: ([0-9.]+), CELoss: ([0-9.]+), loss: ([0-9.]+)"
    )
    eval_avg_re = re.compile(
        r"\[Eval\]\[Epoch (\d+)\]\[Avg\]CELoss: ([0-9.]+), loss: ([0-9.]+), top1: ([0-9.]+), top3: ([0-9.]+)"
    )

    train_by_epoch = {}
    eval_by_epoch = {}

    for line in text:
        m = train_avg_re.search(line)
        if m:
            epoch = int(m.group(1))
            train_by_epoch[epoch] = {
                "epoch": epoch,
                "train_top1": float(m.group(3)),
                "train_top3": float(m.group(4)),
                "train_celoss": float(m.group(5)),
                "train_loss": float(m.group(6)),
            }
            continue
        m = eval_avg_re.search(line)
        if m:
            epoch = int(m.group(1))
            eval_by_epoch[epoch] = {
                "epoch": epoch,
                "eval_celoss": float(m.group(2)),
                "eval_loss": float(m.group(3)),
                "eval_top1": float(m.group(4)),
                "eval_top3": float(m.group(5)),
            }

    epochs = sorted(set(train_by_epoch.keys()) | set(eval_by_epoch.keys()))
    rows = []
    for epoch in epochs:
        train_row = train_by_epoch.get(epoch, {})
        eval_row = eval_by_epoch.get(epoch, {})
        rows.append({
            "epoch": epoch,
            "train_loss": train_row.get("train_loss"),
            "eval_loss": eval_row.get("eval_loss"),
            "train_top1": train_row.get("train_top1"),
            "eval_top1": eval_row.get("eval_top1"),
        })
    return rows


def _plot_picodet(rows, output_path, plt):
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_metric_epochs = [r["epoch"] for r in rows if r["val_metric"] is not None]
    val_metric = [r["val_metric"] for r in rows if r["val_metric"] is not None]
    val_loss = [r["val_loss"] for r in rows if r["val_loss"] is not None]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(epochs, train_loss, color="#1f77b4", linewidth=2.0, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    if val_metric_epochs:
        ax2.plot(
            val_metric_epochs,
            val_metric,
            color="#d62728",
            linewidth=2.0,
            marker="o",
            label="Val bbox-mAP",
        )
    ax2.set_ylabel("Validation bbox-mAP", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    best_epoch = None
    best_metric = None
    if val_metric:
        best_idx = max(range(len(val_metric)), key=lambda i: val_metric[i])
        best_epoch = val_metric_epochs[best_idx]
        best_metric = val_metric[best_idx]
        ax2.scatter([best_epoch], [best_metric], color="#d62728", s=45, zorder=5)
        ax2.annotate(
            f"best mAP={best_metric:.4f} @ epoch {best_epoch}",
            xy=(best_epoch, best_metric),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=9,
            color="#d62728",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    title = "PicoDet-M-416 fine-tuning"
    if val_loss:
        title += " (train/val loss available)"
    else:
        title += " (validation loss not stored; bbox-mAP used instead)"
        fig.text(
            0.02,
            0.02,
            "Note: current artifacts contain train loss and validation bbox-mAP only.\n"
            "Validation loss was not found in the exported detector logs.",
            fontsize=9,
            color="#444444",
        )
    ax1.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_lcnet_losses(rows, output_path, plt):
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    eval_loss = [r["eval_loss"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(epochs, train_loss, linewidth=2.0, color="#1f77b4", label="Train loss")
    ax.plot(epochs, eval_loss, linewidth=2.0, color="#ff7f0e", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PPLCNet_x1_0 fine-tuning loss curves")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    best_val_idx = min(
        (i for i, v in enumerate(eval_loss) if v is not None),
        key=lambda i: eval_loss[i],
    )
    ax.scatter([epochs[best_val_idx]], [eval_loss[best_val_idx]], color="#ff7f0e", s=45)
    ax.annotate(
        f"best val loss={eval_loss[best_val_idx]:.4f} @ epoch {epochs[best_val_idx]}",
        xy=(epochs[best_val_idx], eval_loss[best_val_idx]),
        xytext=(10, -18),
        textcoords="offset points",
        fontsize=9,
        color="#ff7f0e",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_lcnet_accuracy(rows, output_path, plt):
    epochs = [r["epoch"] for r in rows]
    train_top1 = [r["train_top1"] for r in rows]
    eval_top1 = [r["eval_top1"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(epochs, train_top1, linewidth=2.0, color="#2ca02c", label="Train top-1")
    ax.plot(epochs, eval_top1, linewidth=2.0, color="#9467bd", label="Validation top-1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("PPLCNet_x1_0 fine-tuning accuracy curves")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

    best_idx = max(
        (i for i, v in enumerate(eval_top1) if v is not None),
        key=lambda i: eval_top1[i],
    )
    ax.scatter([epochs[best_idx]], [eval_top1[best_idx]], color="#9467bd", s=45)
    ax.annotate(
        f"best val top-1={eval_top1[best_idx]:.4f} @ epoch {epochs[best_idx]}",
        xy=(epochs[best_idx], eval_top1[best_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color="#9467bd",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_readme(output_dir, picodet_csv, lcnet_log):
    readme = output_dir / "README.txt"
    lines = [
        "Generated training-curve figures",
        "",
        f"PicoDet source: {picodet_csv}",
        "  - Current detector artifacts do not contain validation loss.",
        "  - The detector figure therefore plots train loss and validation bbox-mAP.",
        "",
        f"LCNet source: {lcnet_log}",
        "  - The classifier artifacts contain both train and validation loss in train.log.",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")


def _to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Plot fine-tuning curves for detector and classifier.")
    parser.add_argument(
        "--picodet-csv",
        required=True,
    )
    parser.add_argument(
        "--lcnet-log",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "report_figures"),
    )
    args = parser.parse_args()

    plt = _require_matplotlib()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    picodet_csv = Path(args.picodet_csv)
    lcnet_log = Path(args.lcnet_log)

    if not picodet_csv.exists():
        raise FileNotFoundError(f"Missing detector CSV: {picodet_csv}")
    if not lcnet_log.exists():
        raise FileNotFoundError(f"Missing classifier train log: {lcnet_log}")

    picodet_rows = _read_picodet_epoch_metrics(picodet_csv)
    lcnet_rows = _read_lcnet_train_log(lcnet_log)

    if not picodet_rows:
        raise RuntimeError("No detector rows parsed from epoch_metrics.csv")
    if not lcnet_rows:
        raise RuntimeError("No classifier rows parsed from train.log")

    picodet_plot = output_dir / "picodet_finetune_curves.png"
    lcnet_loss_plot = output_dir / "lcnet_loss_curves.png"
    lcnet_acc_plot = output_dir / "lcnet_accuracy_curves.png"

    _plot_picodet(picodet_rows, picodet_plot, plt)
    _plot_lcnet_losses(lcnet_rows, lcnet_loss_plot, plt)
    _plot_lcnet_accuracy(lcnet_rows, lcnet_acc_plot, plt)
    _write_readme(output_dir, picodet_csv, lcnet_log)

    print(f"Saved: {picodet_plot}")
    print(f"Saved: {lcnet_loss_plot}")
    print(f"Saved: {lcnet_acc_plot}")
    print(f"Saved: {output_dir / 'README.txt'}")


if __name__ == "__main__":
    main()
