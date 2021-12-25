import torch
from torch import nn

from alibi.config import ALiBiConfig
from alibi.layers import ALiBiTransformerLayer


class ALiBiTransformer(nn.Module):
    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[ALiBiTransformerLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.layers(x)
