import numpy as np
import torch


def get_batch(tokens, batch_size, context, rng, device):
    assert len(tokens) > context

    starts = rng.integers(
        low=0,
        high=len(tokens) - context,
        size=batch_size,
    )

    sequences = np.stack([
        tokens[start:start + context + 1]
        for start in starts
    ]).astype(np.int64)

    inputs = torch.from_numpy(
        sequences[:, :-1].copy()
    ).to(device)

    targets = torch.from_numpy(
        sequences[:, 1:].copy()
    ).to(device)

    return inputs, targets