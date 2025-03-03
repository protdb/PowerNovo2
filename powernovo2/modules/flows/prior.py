import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from powernovo2.config.default_config import MAX_PEP_LEN
from powernovo2.modules.flows.generative_flow import PWNGenerativeFlow
from powernovo2.modules.flows.lengths_predictor import \
    LengthPredictor


class Prior(nn.Module):

    def __init__(self, flow: PWNGenerativeFlow, length_predictor: LengthPredictor):
        super(Prior, self).__init__()
        self.flow = flow
        self.length_unit = max(2, 2 ** (self.flow.levels - 1))
        self.features = self.flow.features
        self._length_predictor = length_predictor
        self._length_predictor.set_length_unit(self.length_unit)

    def sync(self):
        self.flow.sync()

    def predict_length(self, ctx: torch.Tensor, src_mask: torch.Tensor, topk: int = 1) -> tuple[Tensor, Tensor]:
        return self._length_predictor.predict(ctx, src_mask, topk=topk)

    def length_loss(self, ctx: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        return self._length_predictor.loss(ctx, src_mask, tgt_mask)

    def decode(self, epsilon: torch.Tensor, tgt_mask: torch.Tensor,
               src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        z, logdet = self.flow.fwdpass(epsilon, tgt_mask, src, src_mask)
        log_probs = epsilon.mul(epsilon) + math.log(math.pi * 2.0)
        log_probs = log_probs.mul(tgt_mask.unsqueeze(2))
        log_probs = log_probs.view(z.size(0), -1).sum(dim=1).mul(-0.5) + logdet
        return z, log_probs

    def sample(self, nlengths: int, nsamples: int, src: torch.Tensor,
               ctx: torch.Tensor, src_mask: torch.Tensor,
               tau=0.0, include_zero=False) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        batch = src.size(0)
        batch_nlen = batch * nlengths
        lengths, log_probs_length = self.predict_length(ctx, src_mask, topk=nlengths)

        log_probs_length = log_probs_length.view(-1)
        lengths = lengths.view(-1)
        max_length = MAX_PEP_LEN
        tgt_mask = torch.arange(max_length).to(src.device).unsqueeze(0).expand(batch_nlen, max_length).lt(lengths.unsqueeze(1)).float()

        epsilon = src.new_empty(batch_nlen, nsamples, max_length, self.features).normal_()
        epsilon = epsilon.mul(tgt_mask.view(batch_nlen, 1, max_length, 1)) * tau
        if include_zero:
            epsilon[:, 0].zero_()

        epsilon = epsilon.view(-1, max_length, self.features)
        if nsamples * nlengths > 1:

            src = src.unsqueeze(1) + src.new_zeros(batch, nlengths * nsamples, *src.size()[1:])
            src = src.view(batch_nlen * nsamples, *src.size()[2:])
            ctx = ctx.unsqueeze(1) + ctx.new_zeros(batch, nlengths * nsamples, ctx.size(1))
            ctx = ctx.view(batch_nlen * nsamples, ctx.size(2))
            src_mask = src_mask.unsqueeze(1) + src_mask.new_zeros(batch, nlengths * nsamples, src_mask.size(1))
            src_mask = src_mask.view(batch_nlen * nsamples, src_mask.size(2))
            tgt_mask = tgt_mask.unsqueeze(1) + tgt_mask.new_zeros(batch_nlen, nsamples, tgt_mask.size(1))
            tgt_mask = tgt_mask.view(batch_nlen * nsamples, tgt_mask.size(2))

        z, log_probs = self.decode(epsilon, tgt_mask, src, src_mask)
        return (z, log_probs, tgt_mask), (lengths, log_probs_length), (src, ctx, src_mask)

    def log_probability(self, z: torch.Tensor, tgt_mask: torch.Tensor,
                        src: torch.Tensor, ctx: torch.Tensor, src_mask: torch.Tensor,
                        length_loss: bool = True) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:

        loss_length = self.length_loss(ctx, src_mask, tgt_mask) if length_loss else None


        epsilon, logdet = self.flow.bwdpass(z, tgt_mask, src, src_mask)
        log_probs = epsilon.mul(epsilon) + math.log(math.pi * 2.0)
        log_probs = log_probs.mul(tgt_mask.unsqueeze(2))
        log_probs = log_probs.view(z.size(0), -1).sum(dim=1).mul(-0.5) + logdet
        return log_probs, loss_length

    def init(self, z, tgt_mask, src, src_mask, init_scale=1.0):
        return self.flow.bwdpass(z, tgt_mask, src, src_mask, init=True, init_scale=init_scale)



    @classmethod
    def from_params(cls, params: Dict) -> "Prior":
        flow_params = params['flow']
        flow = PWNGenerativeFlow.from_params(flow_params)
        predictor_params = params['length_predictor']
        length_predictor = LengthPredictor.from_params(predictor_params)
        return Prior(flow, length_predictor)



