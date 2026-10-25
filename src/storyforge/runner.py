import json
import math
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from storyforge.checkpoint import save_checkpoint
from storyforge.evaluation import evaluate
from storyforge.generation import generate
from storyforge.metric import record_metrics
from storyforge.optimization import learning_rate_for_step
from storyforge.training import train_update


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_training(
    model,
    optimizer,
    train_tokens,
    val_tokens,
    config,
    rng,
    device,
    initial_validation_loss,
):
    settings = config["training"]

    tokens_per_update = (
        settings["batch_size"]
        * config["model"]["max_context"]
        * settings["gradient_accumulation_steps"]
    )

    total_steps = math.ceil(
        settings["token_budget"] / tokens_per_update
    )

    halfway_step = max(1, total_steps // 2)

    metrics_path = Path(config["logging"]["metrics_path"])

    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]

    if (
        len(records) != 1
        or records[0]["step"] != 0
        or records[0]["processed_tokens"] != 0
    ):
        raise ValueError(
            "Expected only the initial validation record. "
            "This runner starts from zero; it does not resume a run."
        )

    samples_path = Path(config["generation"]["samples_path"])

    with samples_path.open(encoding="utf-8") as file:
        sample_history = json.load(file)

    frozen_settings = sample_history["initial"]["generation_config"]

    for key in (
        "prompts",
        "max_new_tokens",
        "temperature",
        "top_k",
        "seed",
    ):
        if config["generation"][key] != frozen_settings[key]:
            raise ValueError(
                f"Generation setting changed after baseline: {key}"
            )

    tokenizer = Tokenizer.from_file(
        config["tokenizer"]["path"]
    )

    best_val_loss = initial_validation_loss
    best_step = 0
    best_tokens = 0

    processed_tokens = 0
    training_seconds = 0.0

    torch.manual_seed(config["seed"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)
    wall_start = time.perf_counter()

    for step in range(1, total_steps + 1):
        lr = learning_rate_for_step(
            step,
            total_steps,
            config,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        synchronize(device)
        update_start = time.perf_counter()

        result = train_update(
            model,
            optimizer,
            train_tokens,
            config,
            rng,
            device,
        )

        synchronize(device)
        training_seconds += (
            time.perf_counter() - update_start
        )

        processed_tokens += result["processed_tokens"]

        validation_loss = None

        should_evaluate = (
            step % settings["eval_every"] == 0
            or step == halfway_step
            or step == total_steps
        )

        if should_evaluate:
            validation_loss = evaluate(
                model,
                val_tokens,
                config,
                device,
            )

            if not math.isfinite(validation_loss):
                raise RuntimeError(
                    "Validation produced a non-finite loss."
                )

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_step = step
                best_tokens = processed_tokens

                save_checkpoint(
                    path=config["checkpoints"]["best_path"],
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    processed_tokens=processed_tokens,
                    val_loss=validation_loss,
                )

            save_checkpoint(
                path=config["checkpoints"]["last_path"],
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                processed_tokens=processed_tokens,
                val_loss=validation_loss,
            )

        stages = []

        if step == halfway_step:
            stages.append("halfway")

        if step == total_steps:
            stages.append("final")

        for stage in stages:
            samples = []

            for index, prompt in enumerate(
                config["generation"]["prompts"]
            ):
                seed = config["generation"]["seed"] + index

                text = generate(
                    model,
                    tokenizer,
                    prompt,
                    config,
                    device,
                    seed,
                )

                samples.append({
                    "prompt": prompt,
                    "seed": seed,
                    "text": text,
                })

            sample_history[stage] = {
                "processed_tokens": processed_tokens,
                "generation_config": dict(
                    config["generation"]
                ),
                "samples": samples,
            }

            with samples_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    sample_history,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

        synchronize(device)

        record_metrics(
            metrics_path,
            {
                "step": step,
                "processed_tokens": processed_tokens,
                "train_loss": result["train_loss"],
                "val_loss": validation_loss,
                "learning_rate": lr,
                "training_seconds": training_seconds,
                "elapsed_seconds": (
                    time.perf_counter() - wall_start
                ),
            },
        )

        if (
            step == 1
            or step % settings["log_every"] == 0
            or should_evaluate
        ):
            print(
                f"Update {step}/{total_steps} | "
                f"tokens {processed_tokens:,} | "
                f"train {result['train_loss']:.4f} | "
                f"validation {validation_loss}",
                flush=True,
            )

    summary = {
        "processed_tokens": processed_tokens,
        "optimizer_updates": total_steps,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": math.exp(best_val_loss),
        "best_step": best_step,
        "best_processed_tokens": best_tokens,
        "training_seconds": training_seconds,
        "training_tokens_per_second": (
            processed_tokens / training_seconds
        ),
        "wall_seconds_excluding_initial_validation": (
            time.perf_counter() - wall_start
        ),
        "device": str(device),
        "parameter_dtype": str(
            next(model.parameters()).dtype
        ),
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None
        ),
        "peak_cuda_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda"
            else None
        ),
    }

    summary_path = Path(
        config["logging"]["summary_path"]
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary
