import math

import torch
from torch import nn
from torch.nn import functional as F

from storyforge.block import TransformerBlock


class ForgeGPT(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()

        settings = config["model"]

        d_model = settings["d_model"]
        n_layer = settings["n_layer"]

        self.vocab_size = vocab_size
        self.max_context = settings["max_context"]

        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(self.max_context, d_model)

        self.drop = nn.Dropout(settings["dropout"])

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(n_layer)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.apply(self._initialize)

        residual_std = 0.02 / math.sqrt(2 * n_layer)

        for block in self.blocks:
            nn.init.normal_(
                block.attn.proj.weight,
                std=residual_std,
            )
            nn.init.normal_(
                block.mlp[2].weight,
                std=residual_std,
            )

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        batch_size, sequence_length = ids.shape

        assert 0 < sequence_length <= self.max_context

        positions = torch.arange(
            sequence_length,
            device=ids.device,
        )

        token_vectors = self.token(ids)
        position_vectors = self.position(positions)

        hidden = self.drop(
            token_vectors + position_vectors
        )

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.norm(hidden)

        logits = F.linear(
            hidden,
            self.token.weight,
        )

        loss = None

        if targets is not None:
            assert targets.shape == ids.shape

            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss