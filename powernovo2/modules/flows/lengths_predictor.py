from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LengthPredictor(nn.Module):
    def __init__(self, features, max_tgt_length, min_tgt_length, diff_range, n_mix=1, dropout=0.0):
        super(LengthPredictor, self).__init__()
        self.max_tgt_length = max_tgt_length
        self.range = diff_range
        self.n_mix = n_mix
        self.features = features
        self.dropout = dropout
        self.ctx_proj = None
        self.diff = None
        self.min_tgt_length = min_tgt_length
        self.length_unit = None

    def set_length_unit(self, length_unit):
        self.length_unit = length_unit
        self.ctx_proj = nn.Sequential(nn.Linear(self.features, self.features), nn.ELU())
        self.diff = nn.Linear(self.features, 3 * self.n_mix)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.ctx_proj[0].bias, 0.)
        nn.init.uniform_(self.diff.weight, -0.1, 0.1)
        nn.init.constant_(self.diff.bias, 0.)

    def forward(self, ctx):
        ctx = F.dropout(self.ctx_proj(ctx), p=self.dropout, training=self.training)
        # [batch, 3 * nmix]
        coeffs = self.diff(ctx)
        # [batch, nmix]
        logit_probs = F.log_softmax(coeffs[:, :self.n_mix], dim=1)
        mu = coeffs[:, self.n_mix:self.n_mix * 2]
        log_scale = coeffs[:, self.n_mix * 2:]
        return mu, log_scale, logit_probs


    def loss(self, ctx: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt_lengths = tgt_mask.sum(dim=1).float()
        src_lengths = torch.as_tensor(self.max_tgt_length // 2, device=src_mask.device)
        src_lengths = src_lengths.repeat(src_mask.size(0))

        mu, log_scale, logit_probs = self(ctx)
        x = (tgt_lengths - src_lengths).div(self.range).clamp(min=-1, max=1)
        bin_size = 0.5 / self.range
        lower = bin_size - 1.0
        upper = 1.0 - bin_size
        loss = length_loss(x, mu, log_scale, logit_probs, bin_size, lower, upper)


        return loss


    def predict(self, ctx: torch.Tensor, src_mask:torch.Tensor, topk: int = 1) -> tuple[Tensor, Tensor]:
        bin_size = 0.5 / self.range
        lower = bin_size - 1.0
        upper = 1.0 - bin_size

        src_lengths = torch.as_tensor(self.max_tgt_length // 2, device=src_mask.device)
        src_lengths = src_lengths.repeat(src_mask.size(0))

        mu, log_scale, logit_probs = self(ctx)
        log_probs, diffs = lengths_topk(mu,
                                        log_scale,
                                        logit_probs,
                                        self.range,
                                        bin_size,
                                        lower,
                                        upper,
                                        topk=topk)


        lengths = (diffs + src_lengths.unsqueeze(1)).clamp(min=self.min_tgt_length) - 1

        return lengths, log_probs

    @classmethod
    def from_params(cls, params: Dict) -> 'LengthPredictor':
        return LengthPredictor(**params)



def length_loss(x, means, logscales, logit_probs,
                bin_size, lower, upper) -> torch.Tensor:

    eps = 1e-12
    x = x.unsqueeze(1)

    centered_x = x - means
    if isinstance(logscales, float):
        inv_stdv = np.exp(-logscales)
    else:
        inv_stdv = torch.exp(-logscales)

    min_in = inv_stdv * (centered_x - bin_size)
    plus_in = inv_stdv * (centered_x + bin_size)
    x_in = inv_stdv * centered_x

    cdf_min = torch.sigmoid(min_in)
    cdf_plus = torch.sigmoid(plus_in)
    cdf_delta = cdf_plus - cdf_min
    log_cdf_mid = torch.log(cdf_delta + eps)
    log_cdf_approx = x_in - logscales - 2. * F.softplus(x_in) + np.log(2 * bin_size)

    log_cdf_low = plus_in - F.softplus(plus_in)

    log_cdf_up = -F.softplus(min_in)

    log_cdf = torch.where(cdf_delta.gt(1e-5), log_cdf_mid, log_cdf_approx)
    log_cdf = torch.where(x.ge(lower), log_cdf, log_cdf_low)
    log_cdf = torch.where(x.le(upper), log_cdf, log_cdf_up)

    loss = torch.logsumexp(log_cdf + logit_probs, dim=1) * -1.
    return loss


def lengths_topk(means, logscales, logit_probs, range,
                 bin_size, lower, upper, topk=1) -> Tuple[torch.Tensor, torch.LongTensor]:

    eps = 1e-12

    means = means.unsqueeze(1)
    logscales = logscales.unsqueeze(1)
    logit_probs = logit_probs.unsqueeze(1)

    x = torch.arange(-range, range - 1, 1., device=means.device).unsqueeze(0).unsqueeze(2)
    x = x.div(range)
    centered_x = x - means

    if isinstance(logscales, float):
        inv_stdv = np.exp(-logscales)
    else:
        inv_stdv = torch.exp(-logscales)

    min_in = inv_stdv * (centered_x - bin_size)
    plus_in = inv_stdv * (centered_x + bin_size)
    x_in = inv_stdv * centered_x

    cdf_min = torch.sigmoid(min_in)
    cdf_plus = torch.sigmoid(plus_in)
    cdf_delta = cdf_plus - cdf_min
    log_cdf_mid = torch.log(cdf_delta + eps)
    log_cdf_approx = x_in - logscales - 2. * F.softplus(x_in) + np.log(2 * bin_size)

    log_cdf_low = plus_in - F.softplus(plus_in)

    log_cdf_up = -F.softplus(min_in)

    log_cdf = torch.where(cdf_delta.gt(1e-5), log_cdf_mid, log_cdf_approx)
    log_cdf = torch.where(x.ge(lower), log_cdf, log_cdf_low)
    log_cdf = torch.where(x.le(upper), log_cdf, log_cdf_up)
    log_probs = torch.logsumexp(log_cdf + logit_probs, dim=2)
    log_probs, idx = log_probs.topk(topk, dim=1)

    return log_probs, idx - range