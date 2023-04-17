from __future__ import annotations

from typing import Annotated

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401
from einops import einsum


class NeuralKNN(nn.Module):
    k: int
    dim: int
    temp: float
    num: int
    feature: int

    def __init__(
        self, k: int, dim: int, temp: float, num: int, feature: int | None = None
    ) -> None:
        super().__init__()
        self.temp = temp

        self.k = k
        self.num = num

        self.dim = dim
        if feature is None:
            self.feature = dim

    def similarity(
        self,
        query: Annotated[torch.Tensor, '*B', 'D'],
        key: Annotated[torch.Tensor, '*B', 'D', 'N'],
    ) -> Annotated[torch.Tensor, '*B', 'D', 'N']:
        assert query.shape[-1] == key.shape[-1] == self.dim
        return -einsum('*B D, *B D N -> *B D N', query, key) / (self.dim**0.5)

    def forward(
        self,
        query: Annotated[torch.Tensor, '*B', 'D'],
        keys: Annotated[torch.Tensor, '*B', 'D', 'N'],
        values: Annotated[torch.Tensor, '*B', 'F', 'N'] | None = None,
    ) -> Annotated[torch.Tensor, '*B', 'K', 'D']:
        ...
