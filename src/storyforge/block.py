from torch import nn

from storyforge.attention import CausalAttention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        settings = config["model"]

        d_model = settings["d_model"]
        mlp_dim = d_model * settings["mlp_multiplier"]
        dropout = settings["dropout"]

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalAttention(config)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attention_input = self.norm1(x)
        attention_update = self.attn(attention_input)

        x = x + attention_update

        mlp_input = self.norm2(x)
        mlp_update = self.mlp(mlp_input)

        return x + mlp_update