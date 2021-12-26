import math

import torch
from torch import nn
from torch.nn import functional as F

from alibi.config import ALiBiConfig


def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


class ALiBiMultiHeadAttention(nn.Module):
    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        self.causal = config.causal
        self.num_heads = config.num_heads
        self.scale = math.sqrt(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.m = nn.Parameter(torch.randn(config.num_heads, 1, 1))
        self.kqv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        if config.causal:
            self.register_buffer(
                "mask", torch.tril(torch.ones(1, 1, config.max_len, config.max_len))
            )

    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, _ = x.shape

        key, query, value = self.kqv(x).chunk(3, dim=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # key.shape == (batch_size, num_heads, d_head, seq_len)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # qv.shape == (batch_size, num_heads, seq_len, d_head)

        bias = (self.m * get_relative_positions(seq_len)).unsqueeze(0)
        # bias.shape == (1, num_heads, seq_len, seq_len)

        score = torch.matmul(query, key) / self.scale + bias
        # score.shape == (batch_size, num_heads, seq_len, seq_len)

        if self.causal:
            score = score.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )

        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, value)
        # out.shape == (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.dropout(out)

        return out
