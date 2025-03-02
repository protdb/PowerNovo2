import re
from itertools import groupby
from typing import Tuple

import torch
from Bio.PDB.Polypeptide import aa1
from pyteomics import mass

from powernovo2.config.default_config import MIN_PEP_LEN, aa_residues
from powernovo2.modules.data.primitives import MASSIVE_KB_MOD_MAP
from powernovo2.modules.data.primitives import MASS_SCALE, H2O, MAX_MASS, PROTON
from powernovo2.modules.tokenizers.peptides import PeptideTokenizer


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def to_canonical(seq: str) -> str:
    canonical_seq = re.sub(r"\[.*?\]", '', seq)
    canonical_seq = ''.join([s for s in canonical_seq if s in aa1])
    canonical_seq = canonical_seq.replace('-', '')
    return canonical_seq


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def from_proforma(sequence: str) -> Tuple[str, dict]:
    sequence_mod = []
    mod_dict = {}
    for k, v in MASSIVE_KB_MOD_MAP.items():
        sequence = sequence.replace(k, v).strip()
        if v:
            if v in sequence:
                sequence_mod.append(v)

    if sequence_mod:
        for mod in sequence_mod:
            mod_pos = [i for i in range(len(sequence)) if sequence.startswith(mod, i)]
            mod_dict.update({mod: mod_pos})
    return sequence, mod_dict





class PeptideHelper_(metaclass=Singleton):
    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 reverse:bool = False,
                 mass_scale: int = MASS_SCALE,
                 precursor_tolerance:float = 1e-5,
                 max_mass: int = MAX_MASS
                 ):
        self._tokenizer = PeptideTokenizer(residues=aa_residues,
                                           replace_isoleucine_with_leucine=True,
                                           reverse=reverse)

        self._mass_scale = mass_scale
        self._device = device
        self._residue_masses = torch.zeros(len(self.tokenizer) + 1, dtype=torch.long, device=device)
        self._masses_unscale = torch.zeros(len(self.tokenizer) + 1, dtype=torch.long, device=device)

        for k, v in self._tokenizer.index.items():
            if k in self._tokenizer.residues:
                mass = self._tokenizer.residues[k]
                self._residue_masses[v] = round(mass * self._mass_scale)
                self._masses_unscale[v] = mass

        self.precursor_tolerance = precursor_tolerance
        self._max_mass = max_mass * self.mass_scale
        self._residue_masses[self._residue_masses < 0] = self._max_mass


    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def mass_scale(self):
        return self._mass_scale

    @property
    def device(self):
        return self._device

    @property
    def residue_masses(self):
        return self._residue_masses

    @property
    def max_mass(self):
        return self._max_mass

    def filter_samples(self, precursors: torch.Tensor,
                       tokens:torch.Tensor,
                       probs: torch.Tensor,
                       spectra: torch.Tensor,
                       spectra_mask:torch.Tensor,
                       mass_filter:int = 0,
                       ):
        batch_size, token_len = tokens.size()
        bs = precursors.size(0)
        n_samples = batch_size // bs
        completed_ids = torch.as_tensor(range(bs), dtype=torch.long, device=self._device)
        low, up, remaining_length, _ = self.ensure_mass(tokens,
                                                        precursors)
        completed_mask = torch.logical_and(low, up) * mass_filter
        completed_tokens = tokens.view(bs, n_samples, token_len)
        completed_mask = completed_mask.view(bs, n_samples)
        probs = probs.view(bs, n_samples, probs.size(1), -1)
        up = up.view(bs, n_samples)
        remaining_length = remaining_length.view(bs, n_samples)

        for i in range(bs):
            if completed_mask[i].any():
                if completed_mask[i].sum() > 1:
                    completed_mask[i] &= False


        completed_tokens = completed_tokens[completed_mask]
        completed_probs = probs[completed_mask]
        uncompleted_mask = ~torch.sum(completed_mask, dim=-1).bool()
        remaining_length = remaining_length[uncompleted_mask]
        uncompleted_tokens = tokens.view(bs, n_samples, token_len)
        uncompleted_tokens = uncompleted_tokens[uncompleted_mask]
        uncompleted_spectra = spectra.view(bs, n_samples, spectra.size(1), spectra.size(2))
        uncompleted_spectra = uncompleted_spectra[uncompleted_mask]
        uncompleted_spectra_mask = spectra_mask.view(bs, n_samples, spectra_mask.size(1))
        uncompleted_spectra_mask = uncompleted_spectra_mask[uncompleted_mask]

        # uncompleted tokens, select max probs
        bs -= torch.sum(~uncompleted_mask)
        probs = probs[uncompleted_mask]
        up = up[uncompleted_mask]
        tokens_mask = uncompleted_tokens > 0
        tokens_len = tokens_mask.sum(-1)
        maxp, idx_ = torch.max(probs, dim=-1)

        sump = torch.sum(maxp[:, :, :token_len - 1], dim=-1) / (tokens_len +
                                                            torch.abs(remaining_length) * up.long())

        sump = torch.nan_to_num(sump, nan=0.0)
        sump = sump.view(bs, n_samples)
        max_sp, _ = torch.max(sump, 1)
        max_sp = max_sp.view(bs, 1)
        max_mask = sump == max_sp

        for i in range(bs):
            if torch.sum(max_mask[i]) > 1:
                idx = torch.where(max_mask[i] == 1)
                max_mask[i] &= False
                max_mask[i][idx[0][0]] = True

        uncompleted_tokens = uncompleted_tokens[max_mask]
        maxp, _ = probs.max(1)
        uncompleted_probs = probs[max_mask] + maxp * 1e-2
        uncompleted_spectra = uncompleted_spectra[:, 0, :, :]
        uncompleted_spectra_mask = uncompleted_spectra_mask[:, 0, :]
        completed_ids = completed_ids[~uncompleted_mask]
        precursors = precursors[uncompleted_mask]


        return (completed_tokens, completed_ids, uncompleted_tokens, completed_probs,
                uncompleted_probs, uncompleted_spectra, uncompleted_spectra_mask, precursors)

    def ensure_mass(self,
                    tokens: torch.Tensor,
                    precursors:torch.Tensor,
                    clip_low: int = -MIN_PEP_LEN,
                    clip_up: int = MIN_PEP_LEN,
                    k:int = 2
                    ):
        batch_size, _ = tokens.size()
        residue_masses = self._residue_masses.repeat(batch_size, 1)
        precursor_masses, tolerance = self.precursor_masses(precursors)
        bs = precursor_masses.size(0)
        n_samples = batch_size // bs
        precursor_masses = precursor_masses.repeat_interleave(n_samples)
        tolerance = tolerance.repeat_interleave(n_samples)
        residue_masses = torch.gather(residue_masses, 1, index=tokens).sum(-1)
        low = (precursor_masses - tolerance <= residue_masses)
        up = (precursor_masses + tolerance >= residue_masses)
        residue_masses[residue_masses == self.max_mass] = 0
        mass_err = torch.abs(precursor_masses - residue_masses)
        tokens_len = (tokens > 0).sum(-1)
        mean_mass = torch.round(residue_masses.float() / tokens_len) * k
        remaining_length = (precursor_masses - residue_masses).div(mean_mass).long()
        remaining_length = remaining_length.clip(clip_low, clip_up)
        return low, up, remaining_length, mass_err


    def precursor_masses(self, precursors:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        precursor_masses = (torch.round(self.mass_scale * precursors[:, 0]))
        precursor_masses = precursor_masses - round(self.mass_scale * H2O)
        tolerance = ((self.precursor_tolerance * precursor_masses).round().long())
        return precursor_masses, tolerance


    @staticmethod
    def check_completed(precursor_mass: torch.Tensor, predicted_mass: torch.Tensor, tolerance:torch.Tensor):
        return precursor_mass - tolerance <= predicted_mass <= precursor_mass + tolerance

    def calc_residue_masses(self, tokens:torch.Tensor):
        residue_masses =  torch.gather(self.residue_masses, 0, index=tokens).sum(-1)
        return residue_masses


    def calc_mass_diff(self, tokens:torch.Tensor, precursors:torch.Tensor, isotope=0):
        masses_unscale = self._masses_unscale.repeat(tokens.size(0), 1)
        residue_masses = torch.gather(masses_unscale, -1, index=tokens).sum(-1) + H2O
        charges = precursors[:, 1].int()
        obs_mz = precursors[:, -1]
        residues_mz = (residue_masses / charges) + PROTON
        mass_err = (residues_mz - obs_mz) / obs_mz * 10 ** 6

        print(self.tokenizer.detokenize(tokens))
        print(mass_err / self._mass_scale)
        return mass_err




def calc_ppm(peptide:str, charge:int, precursor_mass:int, ion_type:str='M'):
    mass_seq = mass.fast_mass(peptide, ion_type, charge)
    mass_seq += 57.021 * peptide.count('C')
    ppm = (precursor_mass - mass_seq) / mass_seq * 10**6
    return ppm



