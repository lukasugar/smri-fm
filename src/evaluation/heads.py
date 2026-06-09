import torch.nn as nn
from torch import Tensor


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, pooling: str):
        super().__init__()
        if pooling not in {"first", "mean"}:
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"LinearHead expects token sequence shaped [B, T, D], got {tokens.shape}"
            )
        if self.pooling == "first":
            features = tokens[:, 0]
        elif self.pooling == "mean":
            features = tokens.mean(dim=1)
        else:
            raise AssertionError(f"unreachable pooling: {self.pooling}")
        return self.linear(features)
