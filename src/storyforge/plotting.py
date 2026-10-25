import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curves(config):
    metrics_path = Path(config["logging"]["metrics_path"])

    with metrics_path.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    training = [
        record
        for record in records
        if record["train_loss"] is not None
    ]

    validation = [
        record
        for record in records
        if record["val_loss"] is not None
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    if training:
        ax.plot(
            [r["processed_tokens"] / 1_000_000 for r in training],
            [r["train_loss"] for r in training],
            label="Training",
            color="C0",
        )

    if validation:
        ax.plot(
            [r["processed_tokens"] / 1_000_000 for r in validation],
            [r["val_loss"] for r in validation],
            label="Validation",
            color="C1",
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Processed training target tokens (millions)")
    ax.set_ylabel("Cross-entropy loss (nats per token)")
    ax.set_title("ForgeGPT training")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    output_path = Path(config["logging"]["loss_plot_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path