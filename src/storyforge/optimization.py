import math
import torch

def learning_rate_for_step(step, total_steps, config):
    settings = config["training"]

    peak_lr = settings["learning_rate"]
    min_lr = settings["min_learning_rate"]
    warmup_steps = settings["warmup_steps"]

    if settings["lr_schedule"] != "cosine":
        raise ValueError("Only the cosine schedule is implemented.")

    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "warmup_steps must be smaller than total_steps."
        )

    if not 1 <= step <= total_steps:
        raise ValueError(
            "step must be between 1 and total_steps."
        )

    if step <= warmup_steps:
        return peak_lr * step / warmup_steps

    progress = (
        (step - warmup_steps)
        / (total_steps - warmup_steps)
    )

    decay_factor = 0.5 * (
        1 + math.cos(math.pi * progress)
    )

    return min_lr + (peak_lr - min_lr) * decay_factor

def build_optimizer(model, config):
    settings = config["training"]

    parameters = list(model.parameters())

    matrix_parameters = [
        parameter
        for parameter in parameters
        if parameter.ndim >= 2
    ]

    vector_parameters = [
        parameter
        for parameter in parameters
        if parameter.ndim < 2
    ]

    parameter_groups = [
        {
            "params": matrix_parameters,
            "weight_decay": settings["weight_decay"],
        },
        {
            "params": vector_parameters,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=settings["learning_rate"],
        betas=(
            settings["beta1"],
            settings["beta2"],
        ),
    )