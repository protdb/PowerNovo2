from typing import Tuple, List

import torch


def squeeze(x: torch.Tensor, mask: torch.Tensor, factor: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:

    assert factor >= 1
    if factor == 1:
        return x

    batch, length, features = x.size()
    assert length % factor == 0
    x = x.contiguous().view(batch, length // factor, factor * features)
    mask = mask.view(batch, length // factor, factor).sum(dim=2).clamp(max=1.0)
    return x, mask


def unsqueeze(x: torch.Tensor, factor: int = 2) -> torch.Tensor:

    assert factor >= 1
    if factor == 1:
        return x

    batch, length, features = x.size()
    assert features % factor == 0
    x = x.view(batch, length * factor, features // factor)
    return x


def split(x: torch.Tensor, z1_features) -> Tuple[torch.Tensor, torch.Tensor]:
    z1 = x[:, :, :z1_features]
    z2 = x[:, :, z1_features:]
    return z1, z2


def unsplit(xs: List[torch.Tensor]) -> torch.Tensor:

    return torch.cat(xs, dim=2)


def make_positions(tensor, padding_idx):
    mask = tensor.ne(padding_idx).long()
    return torch.cumsum(mask, dim=1) * mask

