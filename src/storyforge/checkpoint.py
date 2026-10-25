import copy
import hashlib
from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    config,
    step,
    processed_tokens,
    val_loss,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_hash = hashlib.sha256(
        Path(config["tokenizer"]["path"]).read_bytes()
    ).hexdigest()

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": copy.deepcopy(config),
        "vocab_size": model.vocab_size,
        "tokenizer_sha256": tokenizer_hash,
        "step": step,
        "processed_tokens": processed_tokens,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)