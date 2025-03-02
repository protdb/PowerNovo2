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
                 output_folder: str
                 ):
        self.scoring_method = ProteinInferenceGreedySolver
        self.protein_map = protein_map_df
        self.result_network = None
        self.output_filename = output_filename
        self.output_folder = Path(output_folder)

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

    def __build_network(self) -> nx.Graph:
        network = nx.Graph()
        unique_seq = self.protein_map['peptide'].unique()
        unique_proteins = self.protein_map['protein_id'].unique()
        network.add_nodes_from(unique_seq, is_protein=0)
        network.add_nodes_from(unique_proteins, is_protein=1)



        for record in self.protein_map.to_dict(orient="records"):
            protein_id = record['protein_id']
            protein_name = record['protein_name']
            rec_id = record['id']
            score = record['score']

            network.add_edge(protein_id,
                             record['peptide'],
                             ids=rec_id,
                             protein_name=protein_name,
                             score=score)

            network.nodes[protein_id].update({'name': protein_name})

        return network

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
