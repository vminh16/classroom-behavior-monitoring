import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


SCENARIOS = ["Headless", "Visual", "Visual + Telegram"]
LATENCY = np.array([40.62, 44.05, 50.08], dtype=float)
LATENCY_MIN = np.array([38.05, 43.27, 47.77], dtype=float)
LATENCY_MAX = np.array([47.58, 49.49, 54.38], dtype=float)
FPS = np.array([24.62, 22.70, 19.97], dtype=float)

MOT = np.array([13.70, 15.34, 17.55], dtype=float)
CLS = np.array([26.52, 28.29, 30.92], dtype=float)
OTHER = np.array([0.40, 0.42, 1.61], dtype=float)

OVERHEAD = {
    "Visualization": 3.43,
    "Telegram": 6.03,
}

TG_STATS = {
    "Accepted": 96,
    "Enqueued": 96,
    "Sent": 94,
    "Failed": 2,
    "Retried": 10,
    "Queue dropped": 0,
    "Encode failed": 0,
}


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "font.family": "DejaVu Sans",
        }
    )


def add_bar_labels(ax, bars, fmt="{:.2f}", dy=0.4):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_latency_fps():
    x = np.arange(len(SCENARIOS))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    latency_err = np.vstack([LATENCY - LATENCY_MIN, LATENCY_MAX - LATENCY])
    bars = axes[0].bar(
        x,
        LATENCY,
        yerr=latency_err,
        capsize=6,
        color=["#4C956C", "#F4A259", "#BC4749"],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].set_title("End-to-End Latency")
    axes[0].set_ylabel("Latency (ms/frame)")
    axes[0].set_xticks(x, SCENARIOS)
    axes[0].set_ylim(0, 60)
    add_bar_labels(axes[0], bars)

    bars = axes[1].bar(
        x,
        FPS,
        color=["#4C956C", "#F4A259", "#BC4749"],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].axhline(30, color="#2D3142", linestyle="--", linewidth=1.5, label="30 FPS")
    axes[1].set_title("Throughput")
    axes[1].set_ylabel("FPS")
    axes[1].set_xticks(x, SCENARIOS)
    axes[1].set_ylim(0, 35)
    axes[1].legend(loc="upper right")
    add_bar_labels(axes[1], bars, dy=0.25)

    fig.suptitle("Section 3.6: Overall Pipeline Timing", fontsize=17, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_6_latency_fps.png", bbox_inches="tight")
    plt.close(fig)


def plot_breakdown():
    x = np.arange(len(SCENARIOS))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    b1 = ax.bar(x, MOT, color="#577590", label="MOT", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(
        x, CLS, bottom=MOT, color="#F3722C", label="Cls Action", edgecolor="black", linewidth=0.6
    )
    b3 = ax.bar(
        x,
        OTHER,
        bottom=MOT + CLS,
        color="#90BE6D",
        label="Other",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title("Per-Frame Time Breakdown by Module")
    ax.set_ylabel("Time (ms/frame)")
    ax.set_xticks(x, SCENARIOS)
    ax.set_ylim(0, 56)
    ax.legend(loc="upper left", ncols=3)

    for i in range(len(SCENARIOS)):
        ax.text(x[i], MOT[i] / 2, f"{MOT[i]:.2f}", ha="center", va="center", color="white", fontsize=10)
        ax.text(
            x[i],
            MOT[i] + CLS[i] / 2,
            f"{CLS[i]:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )
        ax.text(
            x[i],
            MOT[i] + CLS[i] + OTHER[i] + 0.6,
            f"Total {LATENCY[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_6_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def plot_overhead():
    labels = ["Headless baseline", "+ Visualization", "+ Telegram"]
    values = [40.62, 3.43, 6.03]
    cumulative = np.cumsum(values)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    ax.bar(0, values[0], color="#4C956C", edgecolor="black", linewidth=0.6)
    ax.bar(1, values[1], bottom=values[0], color="#F4A259", edgecolor="black", linewidth=0.6)
    ax.bar(2, values[2], bottom=values[0] + values[1], color="#BC4749", edgecolor="black", linewidth=0.6)

    for i, top in enumerate(cumulative):
        ax.text(i, top + 0.7, f"{top:.2f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Latency (ms/frame)")
    ax.set_title("Incremental Cost from Baseline to Full Deployment")
    ax.set_ylim(0, 56)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C956C"),
        plt.Rectangle((0, 0), 1, 1, color="#F4A259"),
        plt.Rectangle((0, 0), 1, 1, color="#BC4749"),
    ]
    ax.legend(legend_handles, ["Baseline", "Visualization", "Telegram"], loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_6_overhead.png", bbox_inches="tight")
    plt.close(fig)


def plot_telegram_stats():
    labels = list(TG_STATS.keys())
    values = list(TG_STATS.values())
    colors = ["#577590", "#4D908E", "#43AA8B", "#F94144", "#F9C74F", "#BFC0C0", "#ADB5BD"]

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title("Telegram Alert Transport Statistics (3 Full Runs)")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=18)
    add_bar_labels(ax, bars, fmt="{:.0f}", dy=1.2)

    success_rate = TG_STATS["Sent"] / TG_STATS["Enqueued"] * 100
    ax.text(
        0.98,
        0.95,
        f"Send success rate: {success_rate:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "#999999", "boxstyle": "round,pad=0.3"},
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "section_3_6_telegram_stats.png", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate section 3.6 benchmark summary charts.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "report_figures"),
    )
    return parser.parse_args()


def main():
    global OUTPUT_DIR
    args = parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_latency_fps()
    plot_breakdown()
    plot_overhead()
    plot_telegram_stats()


if __name__ == "__main__":
    main()
