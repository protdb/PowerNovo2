from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class InvertibleLinearFlow(nn.Module):
    def __init__(self, in_features, inverse=False):
        super(InvertibleLinearFlow, self).__init__()
        self.inverse = inverse
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features, in_features))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())


    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = input.dim()
        out = F.linear(input, self.weight)
        _, logdet = torch.slogdet(self.weight)
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet


    def backward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        dim = input.dim()
        out = F.linear(input, self.weight_inv)
        _, logdet = torch.slogdet(self.weight_inv)
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet


    def extra_repr(self):
        return 'inverse={}, in_features={}'.format(self.inverse, self.in_features)

    @classmethod
    def from_params(cls, params: Dict) -> "InvertibleLinearFlow":
        return InvertibleLinearFlow(**params)


class InvertibleMultiHeadFlow(nn.Module):

    @staticmethod
    def _get_heads(in_features):
        units = [32, 16, 8]
        for unit in units:
            if in_features % unit == 0:
                return in_features // unit
        assert in_features < 8, 'features={}'.format(in_features)
        return 1

    def __init__(self, in_features, heads=None, type='A', inverse=False):
        super(InvertibleMultiHeadFlow, self).__init__()
        self.in_features = in_features
        self.inverse = inverse

        if heads is None:
            heads = InvertibleMultiHeadFlow._get_heads(in_features)
        self.heads = heads
        self.type = type
        assert in_features % heads == 0, 'features ({}) should be divided by heads ({})'.format(in_features, heads)
        assert type in ['A', 'B'], 'type should belong to [A, B]'
        self.weight = Parameter(torch.Tensor(in_features // heads, in_features // heads))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())


    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        size = input.size()
        dim = input.dim()

        if self.type == 'A':
            out = input.view(*size[:-1], self.heads, self.in_features // self.heads)
        else:
            out = input.view(*size[:-1], self.in_features // self.heads, self.heads).transpose(-2, -1)

        out = F.linear(out, self.weight)
        if self.type == 'B':
            out = out.transpose(-2, -1).contiguous()
        out = out.view(*size)

        _, logdet = torch.slogdet(self.weight)
        if dim > 2:
            num = mask.view(size[0], -1).sum(dim=1) * self.heads
            logdet = logdet * num
        return out, logdet


    def backward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        size = input.size()
        dim = input.dim()
        # [batch, N1, N2, ..., heads, in_features/ heads]
        if self.type == 'A':
            out = input.view(*size[:-1], self.heads, self.in_features // self.heads)
        else:
            out = input.view(*size[:-1], self.in_features // self.heads, self.heads).transpose(-2, -1)

        out = F.linear(out, self.weight_inv)
        if self.type == 'B':
            out = out.transpose(-2, -1).contiguous()
        out = out.view(*size)

        _, logdet = torch.slogdet(self.weight_inv)
        if dim > 2:
            num = mask.view(size[0], -1).sum(dim=1) * self.heads
            logdet = logdet * num
        return out, logdet


    def init(self, data: torch.Tensor, mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data, mask)


    def extra_repr(self):
        return 'inverse={}, in_features={}, heads={}, type={}'.format(self.inverse, self.in_features, self.heads, self.type)

    @classmethod
    def from_params(cls, params: Dict) -> "InvertibleMultiHeadFlow":
        return InvertibleMultiHeadFlow(**params)

