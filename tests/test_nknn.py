from __future__ import annotations

import pytest
import torch

from nknn import NKNN


@pytest.mark.parametrize(
    'k, n, b, dim, feat',
    [
        pytest.param(1, 1024, tuple(), 256, 128),
        pytest.param(2, 1024, (2,), 256, None),
        pytest.param(16, 1024, (2, 2), 256, None),
    ],
)
def test_nknn(k, n, b, dim, feat):
    nknn = NKNN(k=k, dim=dim, temp=0.1, feature=feat)

    query = torch.rand(*b, dim)
    keys = torch.rand(*b, dim, n)
    values = torch.rand(*b, feat or dim, n)

    k_nearest = nknn(query, keys, values)

    assert k_nearest.shape[:-2] == b
    assert k_nearest.shape[-1] == feat or dim
    assert k_nearest.shape[-2] == k
