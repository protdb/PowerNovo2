import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from powernovo2.modules.transformers.layer_norm import LinearWeightNorm
from powernovo2.modules.transformers.positional_encoding import PositionalEncoding
from powernovo2.modules.transformers.transformer import TransformerDecoderLayer


class Posterior(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx):
        super(Posterior, self).__init__()
        self.tgt_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.tgt_embed.weight, -0.1, 0.1)
        if self.tgt_embed.padding_idx is not None:
            with torch.no_grad():
                self.tgt_embed.weight[self.tgt_embed.padding_idx].fill_(0)

    def target_embed_weight(self):
        raise NotImplementedError

    @staticmethod
    def reparameterize(mu, logvar, mask, nsamples=1, random=True):
        size = mu.size()
        std = logvar.mul(0.5).exp()

        if random:
            eps = torch.randn(size[0], nsamples, *size[1:], device=mu.device)
            eps *= mask.view(size[0], 1, size[1], 1)
        else:
            eps = mu.new_zeros(size[0], nsamples, *size[1:])
        return eps.mul(std.unsqueeze(1)).add(mu.unsqueeze(1)), eps


    @staticmethod
    def log_probability(z, eps, mu, logvar, mask):
        size = eps.size()
        nz = size[3]
        log_probs = logvar.unsqueeze(1) + eps.pow(2)
        cc = mask.sum(dim=1, keepdim=True) * (math.log(math.pi * 2.) * nz)
        log_probs = log_probs.view(size[0], size[1], -1).sum(dim=2) + cc
        return log_probs * -0.5


class TransformerCore(nn.Module):
    def __init__(self, embed, num_layers, latent_dim, hidden_size, heads, dropout=0.01, max_length=100):
        super(TransformerCore, self).__init__()
        self.tgt_embed = embed
        self.padding_idx = embed.padding_idx
        embed_dim = embed.embedding_dim
        self.embed_scale = math.sqrt(embed_dim)
        assert embed_dim == latent_dim
        self.dropout = dropout
        layers = [TransformerDecoderLayer(latent_dim, hidden_size, heads, dropout=dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.pos_enc = PositionalEncoding(latent_dim, self.padding_idx, max_length + 1)
        self.mu = LinearWeightNorm(latent_dim, latent_dim, bias=True)
        self.logvar = LinearWeightNorm(latent_dim, latent_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        pass


    def forward(self, tgt_sents, tgt_masks, src_enc, src_masks):
        x = self.embed_scale * self.tgt_embed(tgt_sents)
        x = F.dropout1d(x, p=self.dropout, training=self.training)
        x += self.pos_enc(tgt_sents)
        x = F.dropout(x, p=self.dropout, training=self.training)

        mask = tgt_masks.eq(0)
        key_mask = src_masks.eq(0)
        for layer in self.layers:
            x = layer(x, mask, src_enc, key_mask)

        mu = self.mu(x) * tgt_masks.unsqueeze(2)
        logvar = self.logvar(x) * tgt_masks.unsqueeze(2)

        return mu, logvar




class TransformerPosterior(Posterior):
    def __init__(self, vocab_size, embed_dim, padding_idx, num_layers, latent_dim, hidden_size, heads,
                 dropout=0.0, max_length=100):
        super(TransformerPosterior, self).__init__(vocab_size, embed_dim, padding_idx)
        self.core = TransformerCore(self.tgt_embed, num_layers, latent_dim, hidden_size, heads,
                                    dropout=dropout, max_length=max_length)

    def target_embed_weight(self):
        if isinstance(self.core, nn.DataParallel):
            return self.core.module.tgt_embedd.weight
        else:
            return self.core.tgt_embed.weight


    def forward(self, tgt_sents, tgt_masks, src_enc, src_masks):
        return self.core(tgt_sents, tgt_masks, src_enc, src_masks)


    def sample(self, tgt_sents: torch.Tensor, tgt_masks: torch.Tensor,
               src_enc: torch.Tensor, src_masks: torch.Tensor,
               nsamples: int =1, random=True) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.core(tgt_sents, tgt_masks, src_enc, src_masks)
        z, eps = Posterior.reparameterize(mu, logvar, tgt_masks, nsamples=nsamples, random=random)
        log_probs = Posterior.log_probability(z, eps, mu, logvar, tgt_masks)
        return z, log_probs


    @classmethod
    def from_params(cls, params: Dict) -> "TransformerPosterior":
        return TransformerPosterior(**params)



