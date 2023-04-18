from __future__ import annotations

from typing import Annotated

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


@torch.jit.script
def _compute_omega(
    s: Annotated[torch.Tensor, '*B', 'N'], k: int, t: float
) -> Annotated[torch.Tensor, '*B', 'N', 'K']:
    alpha = F.softmax(s, dim=-1)
    omega = torch.empty(*s.shape, k)

    omega[..., 0] = F.softmax(alpha / t)
    for i in range(1, k):
        alpha = alpha + torch.log(1 - omega[..., i - 1])
        omega[..., i] = F.softmax(alpha / t)

    return omega


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
    ) -> Annotated[torch.Tensor, '*B', 'N']:
        assert query.shape[-1] == key.shape[-1] == self.dim
        return -einsum('*B D, *B D N -> *B N', query, key) / (self.dim**0.5)

    def forward(
        self,
        query: Annotated[torch.Tensor, '*B', 'D'],
        keys: Annotated[torch.Tensor, '*B', 'D', 'N'],
        values: Annotated[torch.Tensor, '*B', 'F', 'N'] | None = None,
    ) -> Annotated[torch.Tensor, '*B', 'K', 'F']:
        sims = self.similarity(query, keys)
        omega = _compute_omega(s=sims, k=self.k, t=self.temp)
        return einsum(omega, values, '*B N K, *B F N -> *B K F')
