"""Tokenizers for peptides."""
from __future__ import annotations

import re
from collections.abc import Iterable

import numba as nb
from pyteomics.proforma import GenericModification, MassModification

from powernovo2.modules.data.primitives import MASSIVE_KB_MOD, Peptide
from powernovo2.modules.tokenizers.tokenizer import Tokenizer


class PeptideTokenizer(Tokenizer):

    residues = nb.typed.Dict.empty(
        nb.types.unicode_type,
        nb.types.float64,
    )
    residues.update(
        G=57.021463735,
        A=71.037113805,
        S=87.032028435,
        P=97.052763875,
        V=99.068413945,
        T=101.047678505,
        C=103.009184505,
        L=113.084064015,
        I=113.084064015,
        N=114.042927470,
        D=115.026943065,
        Q=128.058577540,
        K=128.094963050,
        E=129.042593135,
        M=131.040484645,
        H=137.058911875,
        F=147.068413945,
        R=156.101111050,
        Y=163.063328575,
        W=186.079312980,
    )

    # The peptide parsing function:
    _parse_peptide = Peptide.from_proforma

    def __init__(
            self,
            residues: dict[str, float] | None = None,
            replace_isoleucine_with_leucine: bool = False,
            reverse: bool = False,
    ) -> None:
        """Initialize a PeptideTokenizer."""
        self.replace_isoleucine_with_leucine = replace_isoleucine_with_leucine
        self.reverse = reverse
        self.residues = self.residues.copy()
        if residues is not None:
            self.residues.update(residues)

        if self.replace_isoleucine_with_leucine:
            del self.residues["I"]

        super().__init__(list(self.residues.keys()))

    def split(self, sequence: str) -> list[str]:
        """Split a ProForma peptide sequence.

        Parameters
        ----------
        sequence : str
            The peptide sequence.

        Returns
        -------
        list[str]
            The tokens that compprise the peptide sequence.
        """

        pep = self._parse_peptide(sequence)
        if self.replace_isoleucine_with_leucine:
            pep.sequence = pep.sequence.replace("I", "L")

        pep = pep.split()
        if self.reverse:
            pep.reverse()

        return pep



    @classmethod
    def from_proforma(
            cls,
            sequences: Iterable[str],
            replace_isoleucine_with_leucine: bool = True,
            reverse: bool = True,
    ) -> PeptideTokenizer:
        if isinstance(sequences, str):
            sequences = [sequences]

        # Parse modifications:
        new_res = cls.residues.copy()
        for peptide in sequences:
            parsed = Peptide.from_proforma(peptide).split()
            for token in parsed:
                if token in new_res.keys():
                    continue

                if token == "-":
                    continue

                match = re.search(r"(.*)\[(.*)\]", token)
                try:
                    res, mod = match.groups()
                    if res and res != "-":
                        res_mass = new_res[res]
                    else:
                        res_mass = 0
                except (AttributeError, KeyError) as err:
                    raise ValueError("Unrecognized token {token}.") from err

                try:
                    mod = MassModification(mod)
                except ValueError:
                    mod = GenericModification(mod)

                new_res[token] = res_mass + mod.mass

        return cls(new_res, replace_isoleucine_with_leucine, reverse)

    @staticmethod
    def from_massivekb(
            replace_isoleucine_with_leucine: bool = True,
            reverse: bool = True,
    ) -> MskbPeptideTokenizer:
        return MskbPeptideTokenizer.from_proforma(
            [f"{mod}A" for mod in MASSIVE_KB_MOD.values()],
            replace_isoleucine_with_leucine,
            reverse,
        )

    def get_n_term_aa(self):
        n_term_aa = []
        for aa, mass in self.residues.items():
            aa_idx = self.index[aa]
            if aa in list(MASSIVE_KB_MOD.values()):
                n_term_aa.append(aa_idx)
        return n_term_aa



class MskbPeptideTokenizer(PeptideTokenizer):
    _parse_peptide = Peptide.from_massivekb

