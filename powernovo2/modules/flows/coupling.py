import math
from typing import Tuple, Dict

import torch
import torch.nn as nn

from powernovo2.modules.transformers.attention import MultiHeadAttention
from powernovo2.modules.transformers.layer_norm import LinearWeightNorm
from powernovo2.modules.transformers.positional_encoding import PositionalEncoding
from powernovo2.modules.transformers.transformer import TransformerDecoderLayer


class NICESelfAttnBlock(nn.Module):
    def __init__(self, src_features, in_features, out_features, hidden_features, heads, dropout=0.0,
                 pos_enc='add', max_length=100):
        super(NICESelfAttnBlock, self).__init__()
        assert pos_enc in ['add', 'attn']
        self.src_proj = nn.Linear(src_features, in_features, bias=False) if src_features != in_features else None
        self.pos_enc = PositionalEncoding(in_features, padding_idx=None, init_size=max_length + 1)
        self.pos_attn = MultiHeadAttention(in_features, heads, dropout=dropout) if pos_enc == 'attn' else None
        self.transformer = TransformerDecoderLayer(in_features, hidden_features, heads, dropout=dropout)
        self.linear = LinearWeightNorm(in_features, out_features, bias=True)

    def forward(self, x, mask, src, src_mask):
        if self.src_proj is not None:
            src = self.src_proj(src)

        key_mask = mask.eq(0)
        pos_enc = self.pos_enc(x) * mask.unsqueeze(2)
        if self.pos_attn is None:
            x = x + pos_enc
        else:
            x = self.pos_attn(pos_enc, x, x, key_mask)

        x = self.transformer(x, key_mask, src, src_mask.eq(0))
        return self.linear(x)

    def init(self, x, mask, src, src_mask, init_scale=1.0):
        if self.src_proj is not None:
            src = self.src_proj(src)

        key_mask = mask.eq(0)
        pos_enc = self.pos_enc(x) * mask.unsqueeze(2)
        if self.pos_attn is None:
            x = x + pos_enc
        else:
            x = self.pos_attn(pos_enc, x, x, key_mask)

        x = self.transformer.init(x, key_mask, src, src_mask.eq(0), init_scale=init_scale)
        x = x * mask.unsqueeze(2)
        return self.linear.init(x, init_scale=0.0)


class NICE(nn.Module):

    def __init__(self, src_features, features, hidden_features=None, inverse=False, split_dim=2, split_type='continuous', order='up', factor=2,
                 transform='affine', type='conv', kernel=3, rnn_mode='LSTM', heads=1, dropout=0.0, pos_enc='add', max_length=100):
        super(NICE, self).__init__()
        self.inverse = inverse
        self.features = features
        assert split_dim in [1, 2]
        assert split_type in ['continuous', 'skip']
        if split_dim == 1:
            assert split_type == 'skip'
        if factor != 2:
            assert split_type == 'continuous'
        assert order in ['up', 'down']
        self.split_dim = split_dim
        self.split_type = split_type
        self.up = order == 'up'
        if split_dim == 2:
            out_features = features // factor
            in_features = features - out_features
            self.z1_channels = in_features if self.up else out_features
        else:
            in_features = features
            out_features = features
            self.z1_channels = None
        assert transform in ['additive', 'affine', 'nlsq']
        if transform == 'additive':
            self.transform = Additive
        elif transform == 'affine':
            self.transform = Affine
            out_features = out_features * 2
        elif transform == 'nlsq':
            self.transform = NLSQ
            out_features = out_features * 5
        else:
            raise ValueError('unknown transform: {}'.format(transform))

        if hidden_features is None:
            hidden_features = min(2 * in_features, 1024)
        assert type in ['conv', 'self_attn', 'rnn']

        self.net = NICESelfAttnBlock(src_features, in_features, out_features, hidden_features,
                                         heads=heads, dropout=dropout, pos_enc=pos_enc, max_length=max_length)

    def split(self, z, mask):
        split_dim = self.split_dim
        split_type = self.split_type
        dim = z.size(split_dim)
        if split_type == 'continuous':
            return z.split([self.z1_channels, dim - self.z1_channels], dim=split_dim), mask
        elif split_type == 'skip':
            idx1 = torch.tensor(list(range(0, dim, 2))).to(z.device)
            idx2 = torch.tensor(list(range(1, dim, 2))).to(z.device)
            z1 = z.index_select(split_dim, idx1)
            z2 = z.index_select(split_dim, idx2)
            if split_dim == 1:
                mask = mask.index_select(split_dim, idx1)
            return (z1, z2), mask
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def unsplit(self, z1, z2):
        split_dim = self.split_dim
        split_type = self.split_type
        if split_type == 'continuous':
            return torch.cat([z1, z2], dim=split_dim)
        elif split_type == 'skip':
            z = torch.cat([z1, z2], dim=split_dim)
            dim = z1.size(split_dim)
            idx = torch.tensor([i // 2 if i % 2 == 0 else i // 2 + dim for i in range(dim * 2)]).to(z.device)
            return z.index_select(split_dim, idx)
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def calc_params(self, z: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor):
        params = self.net(z, mask, src, src_mask)
        return params

    def init_net(self, z: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0):
        params = self.net.init(z, mask, src, src_mask, init_scale=init_scale)
        return params


    def forward(self, input: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        (z1, z2), mask = self.split(input, mask)

        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.calc_params(z, mask, src, src_mask)
        zp, logdet = self.transform.fwd(zp, mask, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet


    def backward(self, input: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        (z1, z2), mask = self.split(input, mask)
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.calc_params(z, mask, src, src_mask)
        zp, logdet = self.transform.bwd(zp, mask, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet


    def init(self, data: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        (z1, z2), mask = self.split(data, mask)
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.init_net(z, mask, src, src_mask, init_scale=init_scale)
        zp, logdet = self.transform.fwd(zp, mask, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet


    def extra_repr(self):
        return 'inverse={}, in_channels={}, scale={}'.format(self.inverse, self.in_channels, self.scale)

    @classmethod
    def from_params(cls, params: Dict) -> "NICE":
        return NICE(**params)


class Transform(object):
    @staticmethod
    def fwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def bwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Additive(Transform):
    @staticmethod

    def fwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = params
        z = (z + mu).mul(mask.unsqueeze(2))
        logdet = z.new_zeros(z.size(0))
        return z, logdet

    @staticmethod

    def bwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = params
        z = (z - mu).mul(mask.unsqueeze(2))
        logdet = z.new_zeros(z.size(0))
        return z, logdet


class Affine(Transform):
    @staticmethod
    def fwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_scale = params.chunk(2, dim=2)
        scale = torch.sigmoid(torch.add(log_scale, 2.0))
        z = torch.mul(scale * z + mu, mask.unsqueeze(2))
        logdet = scale.log().mul(mask.unsqueeze(2)).view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    def bwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_scale = params.chunk(2, dim=2)
        scale = log_scale.add_(2.0).sigmoid_()
        z = (z - mu).div(scale + 1e-12).mul(mask.unsqueeze(2))
        logdet = scale.log().mul(mask.unsqueeze(2)).view(z.size(0), -1).sum(dim=1) * -1.0
        return z, logdet


def arccosh(x):
    return torch.log(x + torch.sqrt(x.pow(2) - 1))


def arcsinh(x):
    return torch.log(x + torch.sqrt(x.pow(2) + 1))


class NLSQ(Transform):
    # A = 8 * math.sqrt(3) / 9 - 0.05  # 0.05 is a small number to prevent exactly 0 slope
    logA = math.log(8 * math.sqrt(3) / 9 - 0.05)  # 0.05 is a small number to prevent exactly 0 slope

    @staticmethod
    def get_pseudo_params(params):
        a, logb, cprime, logd, g = params.chunk(5, dim=2)

        # for stability
        logb = logb.mul_(0.4)
        cprime = cprime.mul_(0.3)
        logd = logd.mul_(0.4)

        # b = logb.add_(2.0).sigmoid_()
        # d = logd.add_(2.0).sigmoid_()
        # c = (NLSQ.A * b / d).mul(cprime.tanh_())

        c = (NLSQ.logA + logb - logd).exp_().mul(cprime.tanh_())
        b = logb.exp_()
        d = logd.exp_()
        return a, b, c, d, g

    @staticmethod

    def fwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b, c, d, g = NLSQ.get_pseudo_params(params)

        arg = (d * z).add_(g)
        denom = arg.pow(2).add_(1)
        c = c / denom
        z = (b * z + a + c).mul(mask.unsqueeze(2))
        logdet = torch.log(b - 2 * c * d * arg / denom)
        logdet = logdet.mul(mask.unsqueeze(2)).view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod

    def bwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b, c, d, g = NLSQ.get_pseudo_params(params)

        # double needed for stability. No effect on overall speed
        a = a.double()
        b = b.double()
        c = c.double()
        d = d.double()
        g = g.double()
        z = z.double()

        aa = -b * d.pow(2)
        bb = (z - a) * d.pow(2) - 2 * b * d * g
        cc = (z - a) * 2 * d * g - b * (1 + g.pow(2))
        dd = (z - a) * (1 + g.pow(2)) - c

        p = (3 * aa * cc - bb.pow(2)) / (3 * aa.pow(2))
        q = (2 * bb.pow(3) - 9 * aa * bb * cc + 27 * aa.pow(2) * dd) / (27 * aa.pow(3))

        t = -2 * torch.abs(q) / q * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = -3 * torch.abs(q) / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * arccosh(torch.abs(inter_term1 - 1) + 1)
        t = t * torch.cosh(inter_term2)

        tpos = -2 * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = 3 * q / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * arcsinh(inter_term1)
        tpos = tpos * torch.sinh(inter_term2)

        t[p > 0] = tpos[p > 0]
        z = t - bb / (3 * aa)
        arg = d * z + g
        denom = arg.pow(2) + 1
        logdet = torch.log(b - 2 * c * d * arg / denom.pow(2))

        z = z.float().mul(mask.unsqueeze(2))
        logdet = logdet.float().mul(mask.unsqueeze(2)).view(z.size(0), -1).sum(dim=1) * -1.0
        return z, logdet