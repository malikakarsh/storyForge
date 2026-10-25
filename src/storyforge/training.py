import torch
from torch import nn

from storyforge.batching import get_batch


def train_update(
    model,
    optimizer,
    train_tokens,
    config,
    rng,
    device,
):
    settings = config["training"]

    accumulation_steps = settings["gradient_accumulation_steps"]

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_tokens = 0

    for _ in range(accumulation_steps):
        inputs, targets = get_batch(
            tokens=train_tokens,
            batch_size=settings["batch_size"],
            context=config["model"]["max_context"],
            rng=rng,
            device=device,
        )

        _, loss = model(inputs, targets)

        if not torch.isfinite(loss):
            raise RuntimeError("Training produced a non-finite loss.")

        count = targets.numel()
        total_loss += loss.item() * count
        total_tokens += count

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

    nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=settings["gradient_clip"],
        error_if_nonfinite=True,
    )

    optimizer.step()

    return {
        "train_loss": total_loss / total_tokens,
        "processed_tokens": total_tokens,
    }