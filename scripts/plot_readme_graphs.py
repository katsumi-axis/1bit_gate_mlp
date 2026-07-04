from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Series:
    label: str
    path: str
    color: str
    linestyle: str = "-"


def read_metric(path: str, key: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            values.append(float(row[key]))
    return epochs, values


def plot_accuracy(
    output_path: str,
    *,
    title: str,
    series: list[Series],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)

    for item in series:
        epochs, train_acc = read_metric(item.path, "train_acc")
        ax.plot(
            epochs,
            train_acc,
            label=item.label,
            color=item.color,
            linestyle=item.linestyle,
            linewidth=2.2,
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Train accuracy", fontsize=11)
    ax.set_xlim(1, 100)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([1, 25, 50, 75, 100])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, color="#d9e2ec", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.94,
        fontsize=9,
        ncol=2 if len(series) > 5 else 1,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_accuracy(
        "docs/assets/mlp_vs_onebit_accuracy.png",
        title="Plain MLP vs 1bit SiLU-gated MLP",
        series=[
            Series("MLP h128", "runs/readme_mlp_h128_100e.csv", "#2563eb"),
            Series("MLP h256", "runs/readme_mlp_h256_100e.csv", "#06b6d4"),
            Series("1bit gate h128", "runs/readme_onebit_gate_mlp_h128_100e.csv", "#dc2626"),
            Series("1bit gate h256", "runs/readme_onebit_gate_mlp_h256_100e.csv", "#f97316"),
        ],
    )
    plot_accuracy(
        "docs/assets/attention_vs_onebit_accuracy.png",
        title="Attention MLP vs attention + 1bit SiLU-gated MLP",
        series=[
            Series("attention h128", "runs/readme_attention_mlp_h128_100e.csv", "#2563eb"),
            Series("attention h256", "runs/readme_attention_mlp_h256_100e.csv", "#06b6d4"),
            Series("t1 h128", "runs/readme_attention_onebit_b1_h128_100e.csv", "#dc2626"),
            Series("t4 h128", "runs/readme_attention_onebit_b4_h128_100e.csv", "#65a30d"),
            Series("t8 h128", "runs/readme_attention_onebit_b8_h128_100e.csv", "#16a34a"),
            Series("t1 h256", "runs/readme_attention_onebit_b1_h256_100e.csv", "#9333ea", "--"),
            Series("t4 h256", "runs/readme_attention_onebit_b4_h256_100e.csv", "#64748b", "--"),
            Series("t8 h256", "runs/readme_attention_onebit_b8_h256_100e.csv", "#0f766e", "--"),
        ],
    )


if __name__ == "__main__":
    main()
