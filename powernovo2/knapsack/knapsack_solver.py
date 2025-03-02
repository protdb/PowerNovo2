import numpy as np
import torch

from powernovo2.config.default_config import MAX_PEP_LEN, MIN_PEP_LEN
from powernovo2.knapsack.cbc_solver import knapsackModel
from powernovo2.utils.utils import PeptideHelper_


class KnapsackSolver(object):
    def __init__(self,
                 k_iter: int = 1,
                 skip_residues: int = 3,
                 top_k: int = 7,
                 length_tolerance: int = MIN_PEP_LEN
                 ):

        self._k_iter = k_iter
        self.length_tolerance = length_tolerance
        self.top_k = top_k
        self.skip_residues = skip_residues
        self.helper = PeptideHelper_()
        self.residue_masses_np = self.helper.residue_masses.cpu().numpy()


    @property
    def k_iter(self):
        return self._k_iter

    @k_iter.setter
    def k_iter(self, value):
        self._k_iter = value if value > 0 else 1


    def solve(self,
              tokens:torch.Tensor,
              precursors: torch.Tensor,
              probability: torch.Tensor,
              lengths: torch.Tensor,
              max_peptide_len: int = MAX_PEP_LEN,
              n_iter: int = 0
              ):


        batch_size = probability.size(0)
        device = self.helper.device
        precursor_masses, tolerance = self.helper.precursor_masses(precursors)
        out = torch.zeros(batch_size, max_peptide_len, dtype=torch.long, device=device)
        is_completed = torch.zeros(batch_size, dtype=torch.bool, device=device)
        px_out = torch.zeros_like(out, dtype=torch.float)

        k = 2 * (n_iter + 1)
        tolerance *= k

        for i in range(batch_size):
            source_mass = self.helper.calc_residue_masses(tokens[i])
            remaining_mass = precursor_masses[i] + tolerance[i]
            lower_bound = precursor_masses[i] - tolerance[i]

            if self.helper.check_completed(precursor_mass=precursor_masses[i],
                                           predicted_mass=source_mass,
                                           tolerance=tolerance[i]):
                solution = tokens[i][tokens[i] > 0]
                is_completed[i] = True
            else:
                residue_masses = self.residue_masses_np
                local_prob = probability[i, :lengths[i] - 1, :]
                local_prob = local_prob.flip(0)
                local_prob = local_prob[:, self.skip_residues:]
                local_prob[:, -self.skip_residues - 1:] = 0
                top_prob, top_idx = torch.topk(local_prob, self.top_k - n_iter, dim=-1)
                top_prob[:, 1:] /=  self.helper.mass_scale ** n_iter
                residue_masses = residue_masses[self.skip_residues:]
                cost = top_prob.view(-1).cpu().numpy()
                indices = top_idx.view(-1).cpu().numpy()
                weights = residue_masses[indices]
                indices += self.skip_residues




                solver = knapsackModel(residues_weights=weights,
                                       remaining_mass=remaining_mass.long().item(),
                                       cost=cost * self.helper.mass_scale**2,
                                       length=lengths[i].cpu().item(),
                                       max_pep_len=max_peptide_len,
                                       low_bound=lower_bound.item()
                                       )

                sol = solver.solve_cylp(n_iter=n_iter)

                mask = sol > 0
                indices = np.array(indices, dtype=int)
                solved_residues = indices[solver.mask][mask]
                solved_residues = solved_residues[::-1].copy()
                solution = torch.from_numpy(solved_residues).to(device)
                solved_mass = self.helper.calc_residue_masses(solution)
                is_completed[i] = self.helper.check_completed(solved_mass, precursor_masses[i], tolerance[i])

                if len(solution) >= MAX_PEP_LEN - 2:
                    solution = tokens[i][:MAX_PEP_LEN - 2]

            px = torch.gather(probability[i], 1, index=solution.unsqueeze(1))
            px = px.squeeze(1)
            px = self.norm_probability(px)
            px_out[i, :px.size(0)] = px

            out[i, :len(solution)] = solution
            out[i, len(solution)] = 1

        return out, is_completed, px_out

    @staticmethod
    def norm_probability(p: torch.Tensor, alpha=0.1):
            p[p > 1.0] = 1.0
            p[p < alpha] *= (10**torch.ceil(torch.log10(alpha / p[p < alpha]))) / 2
            return p









































