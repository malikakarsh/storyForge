import math
import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        settings = config["model"]

        d_model = settings["d_model"]
        max_context = settings["max_context"]
        dropout = settings["dropout"]

        self.heads = settings["n_head"]
        self.max_context = max_context

        assert d_model % self.heads == 0
        self.head_dim = d_model // self.heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        allowed = torch.ones(
            max_context,
            max_context,
            dtype=torch.bool,
        ).tril()

        self.register_buffer(
            "allowed",
            allowed,
            persistent=False,
        )

    def forward(self, x, story_mask=None):
        batch_size, sequence_length, d_model = x.shape

        assert 0 < sequence_length <= self.max_context

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q, k, v = [
            tensor.reshape(
                batch_size,
                sequence_length,
                self.heads,
                self.head_dim,
            ).transpose(1, 2)
            for tensor in (q, k, v)
        ]

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        allowed = self.allowed[
            :sequence_length,
            :sequence_length,
        ]
        
        if story_mask is not None:
            allowed = allowed & story_mask

        scores = scores.masked_fill(
            ~allowed,
            float("-inf"),
        )

        weights = torch.softmax(
            scores.float(),
            dim=-1,
        ).to(v.dtype)

        out = self.attn_drop(weights) @ v

        out = out.transpose(1, 2).contiguous().reshape(
            batch_size,
            sequence_length,
            d_model,
        )

        return self.out_drop(self.proj(out))