import os.path
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from powernovo2.config.default_config import MAX_PEP_LEN, NUM_TOKENS, MIN_PEP_LEN
from powernovo2.modules.flows.posterior import TransformerPosterior
from powernovo2.modules.flows.prior import Prior
from powernovo2.modules.transformers.attention import MultiHeadAttention
from powernovo2.modules.transformers.criterion import LabelSmoothedCrossEntropyLoss
from powernovo2.modules.transformers.positional_encoding import (PositionalEncoding, PeakEncoder,
                                                                 SpectraPositionalEncoder, FloatEncoder)
from powernovo2.modules.transformers.transformer import TransformerEncoderLayer, TransformerDecoderLayer
from powernovo2.utils.utils import PeptideHelper_


class BaseModelEncoder(nn.Module):
    def __init__(self, *args, **kwargs: Any):
        super().__init__()
        hidden_size = kwargs['dim']
        num_layers = kwargs['n_layers']
        heads = kwargs['n_heads']
        latent_dim = hidden_size // 2

        try:
            dropout = kwargs['dropout']
        except KeyError:
            dropout = 0

        layers = [TransformerEncoderLayer(latent_dim, hidden_size, heads, dropout=dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)

        self.peak_encoder = PeakEncoder(latent_dim)
        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, latent_dim))

        try:
            self.max_charge = int(kwargs['max_charge'])
        except KeyError:
            self.max_charge: int = 10.0

        try:
            self.default_charge = int(kwargs['default_charge'])
        except KeyError:
            self.default_charge: int = 1

        self.mass_encoder = FloatEncoder(latent_dim)
        self.positional_encoder = SpectraPositionalEncoder(latent_dim)
        self.charge_encoder = torch.nn.Embedding(self.max_charge + 1, latent_dim)



    def forward(self, spectra: torch.Tensor, masks: torch.Tensor, precursors:torch.Tensor):

        peaks = self.peak_encoder(spectra)
        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)
        peaks = torch.cat([latent_spectra, peaks], dim=1)

        masses = self.mass_encoder(precursors[:, None, 0])
        charges = precursors[:, 1].int() - 1
        charges_mask = charges <= 0
        charges[charges_mask] = self.default_charge
        charges_mask = charges >= self.max_charge
        charges[charges_mask] = self.default_charge
        charges = self.charge_encoder(charges)
        precursors = masses + charges[:, None, :]
        peaks = torch.cat([precursors, peaks], dim=1)

        key_mask = masks.eq(0)
        x = peaks

        if not key_mask.any():
            key_mask = None

        for layer in self.layers:
            x = layer(x, key_mask)

        x *= masks.unsqueeze(2)

        batch = spectra.size(0)
        idx = masks.sum(dim=1).long() - 1
        batch_idx = torch.arange(0, batch).long().to(idx.device)
        ctx = x[batch_idx, idx]
        return x, ctx



class BaseModelDecoder(nn.Module):
    def __init__(self, *args, **kwargs: Any):
        hidden_size = kwargs['dim']
        latent_dim = hidden_size // 2

        super(BaseModelDecoder, self).__init__()

        hidden_size = kwargs['dim']
        num_layers = kwargs['n_layers']
        heads = kwargs['n_heads']

        try:
            self.dropout = kwargs['dropout']
        except KeyError:
            self.dropout = 0

        try:
            label_smoothing = kwargs['label_smoothing']
        except KeyError:
            label_smoothing = 0

        self.pos_attn = MultiHeadAttention(latent_dim, heads, dropout=self.dropout)
        layers = [TransformerDecoderLayer(latent_dim, hidden_size, heads,
                                          dropout=self.dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.pos_enc = PositionalEncoding(latent_dim, None, MAX_PEP_LEN + 1)

        self.readout = nn.Linear(latent_dim, NUM_TOKENS, bias=True)
        self.reset_parameters()

        if label_smoothing < 1e-5:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif 1e-5 < label_smoothing < 1.0:
            self.criterion = LabelSmoothedCrossEntropyLoss(label_smoothing)


    def reset_parameters(self):
        nn.init.uniform_(self.readout.weight, -0.1, 0.1)
        nn.init.constant_(self.readout.bias, 0.)


    def forward(self,
                z,
                mask,
                src,
                src_mask
                ):

        z = F.dropout1d(z, p=self.dropout, training=self.training)

        pos_enc = self.pos_enc(z) *  mask.unsqueeze(2)
        key_mask = mask.eq(0)
        ctx = self.pos_attn(pos_enc, z, z, key_mask)

        src_mask = src_mask.eq(0)
        for layer in self.layers:
            ctx = layer(ctx, key_mask, src, src_mask)

        return self.readout(ctx)

    def decode(self,
               z: torch.Tensor,
               mask: torch.Tensor,
               src: torch.Tensor,
               src_mask: torch.Tensor,
               zero_probs: bool = True):

        probs = self(z, mask, src, src_mask)
        log_probs = F.log_softmax(probs, dim=2)

        _, dec = log_probs.max(dim=2)
        decode = dec * mask.long()

        if zero_probs:
            probs = torch.exp(log_probs)
            probs *= mask.unsqueeze(-1).float()
        else:
            probs = log_probs

        return decode, probs

    def loss(self, z: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        logits = self(z, mask, src, src_mask).transpose(1, 2)
        loss = self.criterion(logits, target).mul(mask)

        return loss.sum(dim=1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device



class BaseModel(nn.Module):
    def __init__(self, encoder: BaseModelEncoder, decoder: BaseModelDecoder, config:dict):
        super().__init__()
        self.prior = Prior.from_params(config['prior_params'])
        self.posterior = TransformerPosterior.from_params(config['posterior_params'])
        self.decoder = decoder
        self.encoder = encoder

    def sample_from_prior(self,
                          spectra: torch.Tensor,
                          spectra_masks: torch.Tensor,
                          precursors: torch.Tensor,
                          n_lengths: int = 1,
                          n_samples: int = 1,
                          tau: float = 0.0,
                          include_zero=False) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor]]:

        src_enc, ctx = self.encoder(spectra=spectra, masks=spectra_masks, precursors=precursors)

        return self.prior.sample(n_lengths, n_samples, src_enc, ctx, spectra_masks, tau=tau,
                                 include_zero=include_zero)

    def sample_from_posterior(self, tgt_tokens: torch, tgt_masks: torch.Tensor,
                              src_enc: torch.Tensor, src_masks: torch.Tensor,
                              n_samples: int = 1, random=True) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.posterior.sample(tgt_tokens, tgt_masks, src_enc, src_masks, nsamples=n_samples, random=random)

    def reconstruct_loss(self, src_spectra: torch.Tensor, tgt_tokens: torch,
                         src_masks: torch.Tensor, tgt_masks: torch.Tensor, precursors: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:

        src_enc, ctx = self.encoder(src_spectra, masks=src_masks, precursors=precursors)
        z, _ = self.sample_from_posterior(tgt_tokens, tgt_masks, src_enc, src_masks, random=False)

        z = z.squeeze(1)

        loss_length = self.prior.length_loss(ctx, src_masks, tgt_masks)
        recon_err = self.decoder.loss(z, tgt_tokens, tgt_masks, src_enc, src_masks)
        return recon_err, loss_length

    def generative_loss(self, src_spectra: torch.Tensor, tgt_tokens: torch,
                        src_masks: torch.Tensor, tgt_masks: torch.Tensor,
                        precursors: torch.Tensor, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        src_enc, ctx = self.encoder(src_spectra, masks=src_masks, precursors=precursors)
        z, log_probs_posterior = self.sample_from_posterior(tgt_tokens, tgt_masks, src_enc, src_masks,
                                                            n_samples=n_samples, random=True)

        batch, _, length, nz = z.size()
        if n_samples > 1:
            src_enc = src_enc.unsqueeze(1) + src_enc.new_zeros(batch, n_samples, *src_enc.size()[1:])
            src_enc = src_enc.view(batch * n_samples, *src_enc.size()[2:])
            ctx = ctx.unsqueeze(1) + ctx.new_zeros(batch, n_samples, ctx.size(1))
            ctx = ctx.view(batch * n_samples, ctx.size(2))
            src_masks = src_masks.unsqueeze(1) + src_masks.new_zeros(batch, n_samples, src_masks.size(1))
            src_masks = src_masks.view(batch * n_samples, src_masks.size(2))
            tgt_masks = tgt_masks.unsqueeze(1) + tgt_masks.new_zeros(batch, n_samples, tgt_masks.size(1))
            tgt_masks = tgt_masks.view(batch * n_samples, tgt_masks.size(2))
            tgt_tokens = tgt_tokens.unsqueeze(1) + tgt_tokens.new_zeros(batch, n_samples, tgt_tokens.size(1))
            tgt_tokens = tgt_tokens.view(batch * n_samples, tgt_tokens.size(2))

        z = z.view(-1, length, nz)

        log_probs_prior, loss_length = self.prior.log_probability(z, tgt_masks, src_enc, ctx, src_masks,
                                                                  length_loss=True)
        log_probs_prior = log_probs_prior.view(batch, n_samples)
        loss_length = loss_length.view(batch, n_samples)
        KL = (log_probs_posterior - log_probs_prior).mean(dim=1)
        loss_length = loss_length.mean(dim=1)
        recon_err = self.decoder.loss(z, tgt_tokens, tgt_masks, src_enc, src_masks).view(batch, n_samples).mean(dim=1)

        return recon_err, KL, loss_length

    def forward(self, src_spectra: torch.Tensor, tgt_tokens: torch, src_masks: torch.Tensor, tgt_masks: torch.Tensor,
                precursors: torch.Tensor, n_samples: int = 1, only_recon_loss=False):
        if only_recon_loss:
            return self.reconstruct_loss(src_spectra, tgt_tokens, src_masks, tgt_masks, precursors=precursors)
        else:
            return self.generative_loss(src_spectra, tgt_tokens, src_masks, tgt_masks,
                                        precursors=precursors, n_samples=n_samples)


class PWNFlow(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        config = kwargs.pop('configs')
        if not config:
            raise AttributeError(f"Model parameters are not specified in the configuration: {kwargs}")
        model_config = config['model_config']

        super().__init__(*args, **kwargs)
        self.encoder = BaseModelEncoder(**model_config['encoder'])
        self.decoder = BaseModelDecoder(**model_config['decoder'])
        self.model = BaseModel(self.encoder, self.decoder, config=model_config)
        self.env_config = config['run_config']['environment']
        self.helper = PeptideHelper_(device=self.device)

    def load_pretrained(self) -> str:
        model_path = self.env_config['base_model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Path to pretrained model not found: {model_path}")

        ckpt_data = torch.load(model_path, map_location=self.device)
        self.load_state_dict(ckpt_data['state_dict'])
        return model_path

    @torch.no_grad()
    def decode(self,  spectra: torch.Tensor, precursors: torch.Tensor):
        zeros = ~spectra.sum(dim=2).bool()
        dummy_ = torch.zeros(spectra.shape[0], 1, device=self.device, dtype=torch.bool)

        mask = [
            torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
            zeros,
        ]

        mask = torch.cat(mask, dim=1).bool()
        mask = torch.cat((mask, dummy_), dim=-1)

        spectra_masks = ~mask

        (z, log_probs, tgt_mask), (lengths, log_probs_length), (src, _, src_mask) = self.model.sample_from_prior(spectra,
                                                                                      spectra_masks,
                                                                                      precursors=precursors)
        tokens, decoder_prob = self.decoder.decode(z, tgt_mask, src, spectra_masks)
        tokens = tokens.mul(tgt_mask).long()

        return {'tokens': tokens, 'lengths':  lengths, 'probs': decoder_prob,  'encoder_output':(src, spectra_masks)}


    @torch.no_grad()
    def reconstruct(self, tokens: torch.Tensor, encoder_output:tuple[torch.Tensor, torch.Tensor], random=False):
        src_enc,  src_masks = encoder_output
        mask = tokens > 0
        z, _ = self.model.sample_from_posterior(tokens, mask, src_enc, src_masks, random=random)

        z = z.squeeze(1)
        reconstruct_out, reconstruct_prob = self.decoder.decode(z, mask, src_enc, src_masks)

        return reconstruct_out, reconstruct_prob


    @torch.no_grad()
    def resample(self, tokens: torch.Tensor,
                 encoder_output:tuple[torch.Tensor, torch.Tensor],
                 z:torch.Tensor,
                 ):
        src_enc,  src_masks = encoder_output
        batch_size = src_enc.size(0)
        mask = tokens > 0
        z = torch.zeros((batch_size, tokens.size(1), src_enc.size(-1)), device=self.device)
        reconstruct_out, reconstruct_prob = self.decoder.decode(z, mask, src_enc, src_masks)

        return reconstruct_out, reconstruct_prob


    @torch.no_grad()
    def sample(self, spectra: torch.Tensor, precursors: torch.Tensor, n_samples=5):
        batch_size = spectra.shape[0]
        zeros = ~spectra.sum(dim=2).bool()
        dummy_ = torch.zeros(spectra.shape[0], 1, device=self.device, dtype=torch.bool)

        mask = [torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
                zeros]

        mask = torch.cat(mask, dim=1).bool()
        mask = torch.cat((mask, dummy_), dim=-1)

        spectra_masks = ~mask

        (z, log_probs, tgt_mask), (_, log_probs_length), (
            src, ctx, spectra_masks) = self.model.sample_from_prior(
            spectra,
            spectra_masks,
            n_lengths=n_samples,
            n_samples=1,
            precursors=precursors)


        # [batch, n_len]
        tokens, decoder_prob1 = self.decoder.decode(z, tgt_mask, src, spectra_masks)

        _, _, remaining_length, _ = self.helper.ensure_mass(tokens, precursors)
        current_length = tgt_mask.sum(-1).long()
        current_length += remaining_length.long()
        current_length.clip(MIN_PEP_LEN, MAX_PEP_LEN - 1)
        new_mask = torch.arange(MAX_PEP_LEN).to(self.device).unsqueeze(0).expand(batch_size * n_samples,MAX_PEP_LEN).lt(
            current_length.unsqueeze(1)).float()

        tokens, decoder_prob2 = self.decoder.decode(z, new_mask, src, spectra_masks)

        decoder_prob = (decoder_prob1 + decoder_prob2) * decoder_prob2

        completed = torch.zeros(batch_size, tokens.size(-1), dtype=torch.long, device=self.device)
        completed_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        probs = torch.zeros_like(completed, dtype=torch.float32)

        (completed_tokens, completed_ids, uncompleted_tokens, completed_probs, uncompleted_probs,
         src, spectra_masks, precursors) = self.helper.filter_samples(precursors,
                                                                      tokens,
                                                                      decoder_prob,
                                                                      src,
                                                                      spectra_masks,
                                                                      mass_filter=True)

        if torch.numel(completed_ids):
            completed[completed_ids] = completed_tokens
            completed_mask[completed_ids] = True
            px = torch.gather(completed_probs, 2, completed_tokens.unsqueeze(-1))
            probs[completed_ids] = px.squeeze(-1)

        return (completed, completed_mask, uncompleted_tokens, probs,
                uncompleted_probs, src, spectra_masks, precursors)

    @torch.no_grad()
    def smart_sample(self, spectra: torch.Tensor, precursors: torch.Tensor, length_max=40):
        batch_size = spectra.shape[0]
        zeros = ~spectra.sum(dim=2).bool()
        dummy_ = torch.zeros(spectra.shape[0], 1, device=self.device, dtype=torch.bool)

        mask = [torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
                zeros]

        mask = torch.cat(mask, dim=1).bool()
        mask = torch.cat((mask, dummy_), dim=-1)

        spectra_masks = ~mask

        n_samples = length_max - MIN_PEP_LEN + 1

        batch_n_length = n_samples * batch_size

        src, _ = self.encoder(spectra, spectra_masks, precursors)
        src = src.unsqueeze(1) + src.new_zeros(batch_size,  n_samples, *src.size()[1:])
        src = src.view(batch_n_length, *src.size()[2:])
        spectra_masks = spectra_masks.unsqueeze(1) + spectra_masks.new_zeros(batch_size,
                                                                             n_samples,
                                                                             spectra_masks.size(1))
        spectra_masks = spectra_masks.view(batch_n_length, spectra_masks.size(2))

        lengths = [[i + MIN_PEP_LEN for i in range(n_samples)]] * batch_size
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        lengths = lengths.view(-1)

        tgt_mask = torch.arange(MAX_PEP_LEN).to(src.device).unsqueeze(0).expand(batch_n_length,
                                                                                MAX_PEP_LEN).lt(
            lengths.unsqueeze(1)).float()


        z = torch.zeros(batch_n_length, MAX_PEP_LEN, src.size(-1), device=self.device)

        # [batch, n_len]
        tokens, decoder_prob = self.decoder.decode(z, tgt_mask, src, spectra_masks)

        completed = torch.zeros(batch_size, tokens.size(-1), dtype=torch.long, device=self.device)
        completed_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        probs = torch.zeros_like(completed, dtype=torch.float32)

        (completed_tokens, completed_ids, uncompleted_tokens, completed_probs, uncompleted_probs,
         src, spectra_masks, precursors) = self.helper.filter_samples(precursors,
                                                                      tokens,
                                                                      decoder_prob,
                                                                      src,
                                                                      spectra_masks,
                                                                      mass_filter=True)

        if torch.numel(completed_ids):
            completed[completed_ids] = completed_tokens
            completed_mask[completed_ids] = True
            px = torch.gather(completed_probs, 2, completed_tokens.unsqueeze(-1))
            probs[completed_ids] = px.squeeze(-1)


        return (completed, completed_mask, uncompleted_tokens, probs,
                uncompleted_probs, src, spectra_masks, precursors)




    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device



