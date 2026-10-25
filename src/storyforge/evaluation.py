import numpy as np
import torch


def evaluate(model, val_tokens, config, device):
    settings = config["evaluation"]

    context = config["model"]["max_context"]
    batch_size = settings["batch_size"]

    target_count = min(
        settings["target_tokens"],
        len(val_tokens) - 1,
    )

    target_count = (target_count // context) * context

    if target_count == 0:
        raise ValueError(
            "Validation data must contain at least one full window."
        )

    previous_training_mode = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(
            0,
            target_count,
            batch_size * context,
        ):
            stop = min(
                start + batch_size * context,
                target_count,
            )

            starts = range(start, stop, context)

            sequences = np.stack([
                val_tokens[s:s + context + 1]
                for s in starts
            ]).astype(np.int64)

            inputs = torch.from_numpy(
                sequences[:, :-1].copy()
            ).to(device)

            targets = torch.from_numpy(
                sequences[:, 1:].copy()
            ).to(device)

            _, loss = model(inputs, targets)

            count = targets.numel()

            total_loss += loss.item() * count
            total_tokens += count

    model.train(previous_training_mode)

    return total_loss / total_tokens