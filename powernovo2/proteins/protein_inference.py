import math
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import networkx as nx
import pandas as pd

from powernovo2.proteins.greedy_solver import ProteinInferenceGreedySolver
from powernovo2.proteins.output_builder import TableMaker
from powernovo2.proteins.protein_merger import ProteinMerger
from powernovo2.proteins.psm_network import PSMNetworkSolver
from powernovo2.proteins.sequences_tagger import SequencesTagger


class ProteinInference(object):
    def __init__(self,
                 protein_map_df: pd.DataFrame,
                 output_filename: str,
                 output_folder: str,
                 denovo_mapping: dict = None
                 ):
        self.scoring_method = ProteinInferenceGreedySolver
        self.protein_map = protein_map_df
        self.result_network = None
        self.output_filename = output_filename
        self.output_folder = Path(output_folder)
        self.denovo_mapping = denovo_mapping or {}

    def __build_network(self) -> nx.Graph:
        def clip01(x):
            try:
                x = float(x)
            except Exception:
                return None
            if math.isnan(x):
                return None
            return 0.0 if x < 0 else 1.0 if x > 1 else x

        w1, w2, w3 = 0.5, 0.3, 0.2
        gamma = 0.5
        ppm0 = 5.0

        network = nx.Graph()
        unique_seq = self.protein_map['peptide'].unique()
        unique_proteins = self.protein_map['protein_id'].unique()
        network.add_nodes_from(unique_seq, is_protein=0)
        network.add_nodes_from(unique_proteins, is_protein=1)

        for record in self.protein_map.to_dict(orient="records"):
            protein_id = record['protein_id']
            protein_name = record['protein_name']
            rec_id = str(record['id'])
            peptide_seq = record['peptide']
            identity = record['score']  # 0..100
            ppm_diff = record.get('pepide_ppm_diff', None)

            # Обновим информацию de novo в узле пептида до расчёта peptide_score
            dm = getattr(self, 'denovo_mapping', None) or {}
            denovo = dm.get(rec_id, {})
            node_pep = network.nodes[peptide_seq] if peptide_seq in network else None
            if node_pep is None:
                if peptide_seq not in network:
                    network.add_node(peptide_seq, is_protein=0)
                node_pep = network.nodes[peptide_seq]

            node_pep.setdefault('scan_id', rec_id)
            if 'denovo_sequence' not in node_pep or node_pep['denovo_sequence'] in (None, ''):
                if 'denovo_sequence' in denovo:
                    node_pep['denovo_sequence'] = denovo['denovo_sequence']
            if 'denovo_score' not in node_pep or node_pep['denovo_score'] is None:
                if 'denovo_score' in denovo:
                    node_pep['denovo_score'] = denovo['denovo_score']

            identity_norm = None
            if identity is not None:
                try:
                    x = max(0.0, min(1.0, float(identity)))

                    identity_norm = x ** gamma
                except (ValueError, Exception):
                    identity_norm = None

            denovo_norm = clip01(node_pep.get('denovo_score', None))

            ppm_score = None
            if ppm_diff is not None:
                try:
                    val = abs(float(ppm_diff))
                    g = math.exp(- (val / ppm0) ** 2)
                    if val <= ppm0:
                        ppm_score = 0.95 + 0.05 * g
                    else:
                        ppm_score = 0.95 * g
                except (ValueError, Exception):
                    ppm_score = None

            parts = [(identity_norm, w1), (denovo_norm, w2), (ppm_score, w3)]
            valid = [(v, w) for v, w in parts if v is not None]
            if valid:
                total_w = sum(w for _, w in valid)
                peptide_score = sum(v * (w / total_w) for v, w in valid)
            else:
                peptide_score = None

            network.add_edge(
                protein_id,
                peptide_seq,
                ids=rec_id,
                protein_name=protein_name,
                score=identity,              # исходный identity (0..100)
                pepide_ppm_diff=ppm_diff,
                peptide_score=peptide_score  # обновлённый комбинированный скор
            )

            node_p = network.nodes[protein_id]
            if ('name' not in node_p) or (node_p['name'] is None) or (node_p['name'] == ''):
                node_p['name'] = protein_name if protein_name not in (None, '') else str(protein_id)

        for pid in unique_proteins:
            node_p = network.nodes[pid]
            if ('name' not in node_p) or (node_p['name'] is None) or (node_p['name'] == ''):
                rows = self.protein_map.loc[self.protein_map['protein_id'] == pid]
                fallback = None
                if not rows.empty and 'protein_name' in rows:
                    fallback = rows['protein_name'].iloc[0]
                node_p['name'] = fallback if fallback not in (None, '') else str(pid)

        return network


    def inference(self):
        problem_network = self.__build_network()
        subnetworks = []
        for component in nx.connected_components(problem_network):
            subgraph = problem_network.subgraph(component)
            subnetworks.append(PSMNetworkSolver(subgraph))

        unique_tagged_network = self.parallel(
            subnetworks, SequencesTagger().run)

        self.safe_clear(subnetworks)
        solved_networks = self.parallel(unique_tagged_network, self.scoring_method().run)
        self.safe_clear(unique_tagged_network)
        self.result_network = self.parallel(solved_networks, ProteinMerger().run)
        self.safe_clear(solved_networks)

    def write_output(self):
        if self.result_network is None:
            return
        protein_table = TableMaker().get_system_protein_table(self.result_network)
        peptide_table = TableMaker().get_system_peptide_table(self.result_network)
        assert os.path.exists(self.output_folder), f"Output not found {self.output_folder}"
        protein_table_path = self.output_folder / f'{self.output_filename}_protein.csv'
        peptide_table_path = self.output_folder / f'{self.output_filename}_peptide.csv'
        peptide_table.to_csv(peptide_table_path, index=False, header=True)
        protein_table.to_csv(protein_table_path, index=False, header=True)
        self.safe_clear(self.result_network)

    def solve(self):
        self.inference()
        self.write_output()

    @staticmethod
    def parallel(pns, func):
        p = Pool(cpu_count())
        pns = p.map(func, pns)

        return pns

    @staticmethod
    def safe_clear(obj):
        if obj is not None:
            del obj
