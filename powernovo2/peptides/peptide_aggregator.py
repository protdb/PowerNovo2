import glob
import logging
import os.path
import subprocess
from pathlib import Path

import npysearch as npy
import numpy as np
import pandas as pd

from powernovo2.proteins.protein_inference import ProteinInference

logger = logging.getLogger("powernovo2")
logger.setLevel(logging.INFO)


DENOVO_COLUMNS = ['SCAN ID', 'TITLE', 'PEPTIDE', 'CANONICAL SEQ.', 'PPM DIFFERENCE', 'SCORE', 'POSITIONAL SCORES']

class PeptideAggregator(object):
    def __init__(self,
                 config: dict,
                 output_folder: str,
                 output_filename: str
                 ):
        self.output_folder = Path(output_folder)
        self.output_filename = output_filename
        self.peptides = {}
        self.use_alps = config['use_alps']
        self.use_protein_inference = config['protein_inference']
        self.alps_executable = config['alps_executable']
        self.proteins_fasta_path = config['fasta_path']
        self.n_contigs = config['num_contigs']
        self.kmers = config['kmers']
        self.minIdentity = config['proteins_minIdentity']
        self.denovo_results = {cl:[] for  cl in DENOVO_COLUMNS}
        self.denovo_filename = self.output_folder / f'{self.output_filename}_denovo.csv'



        if self.use_alps and not os.path.exists(self.alps_executable):
            raise FileNotFoundError(f"You specified use_alps = True, "
                                    "but the ALPS.jar executable file was not found "
                                    f"in the assembler folder: {config.assembler_folder}")

        if self.use_protein_inference and not os.path.exists(self.proteins_fasta_path):
            raise FileNotFoundError(f"You specified protein inference = True, "
                                    f"but the FASTA file was not found: {self.proteins_fasta_path}")

    def add_record(self,
                   scan_id: str,
                   annotation:str,
                   predicted_sequence: str,
                   canonical_sequence: str,
                   ppm_diff: float,
                   score: float,
                   aa_scores: np.ndarray,
                   _norm:int = 1e-2
                   ) -> dict:
        if not predicted_sequence:
            return {}


        positional_scores = np.round(aa_scores, 2)
        positional_scores[positional_scores == 0] = _norm
        positional_scores = list(map(str, positional_scores[positional_scores > 0]))

        if len(positional_scores) > len(canonical_sequence):
            positional_scores = positional_scores[:len(canonical_sequence)]

        assert len(canonical_sequence) == len(positional_scores)

        positional_scores_str = ' '.join(positional_scores).strip()


        scan_id = scan_id.replace('index=', '')
        record = {'annotation': annotation,
                  'predicted seq': predicted_sequence,
                  'canonical_seq': canonical_sequence,
                  'ppm_difference': np.round(ppm_diff, 8),
                  'score': score,
                  'positional scores': positional_scores_str
                  }

        self.denovo_results['SCAN ID'].append(scan_id)

        for  i, k in enumerate(record):
            self.denovo_results[list(self.denovo_results.keys())[i + 1]].append(record[k])

        return {scan_id: record}

    def solve(self):
        df_results = self.save_results()

        if self.use_alps:
            results = self.__assembly(df_results)

            if results and self.use_protein_inference:
                protein_map_df = self.__map_proteins(query=results)

                if not protein_map_df.empty:
                    self.inference_proteins(protein_map_df)

        elif self.use_protein_inference:
            df_results['SCAN ID'] = df_results['SCAN ID'].astype(str)
            query = pd.Series(df_results['PEPTIDE'].values, index=df_results['SCAN ID']).to_dict()
            protein_map_df = self.__map_proteins(query=query)


            if not protein_map_df.empty:
                self.inference_proteins(protein_map_df)


    def save_results(self)->pd.DataFrame:
        result_df = pd.DataFrame.from_dict(self.denovo_results)
        result_df = result_df.loc[:, ~(result_df == '').all()]
        logger.info(f'Predictions saved: {self.denovo_filename}')
        result_df.to_csv(self.denovo_filename, header=True, index=False)

        return result_df

    def __assembly(self, result_df:pd.DataFrame) -> dict:
        logger.info(f'Run assembler: {self.alps_executable}')
        assembly_filename = self.output_folder / f'{self.output_filename}_tmp.csv'
        assembly_df = result_df.drop(columns=['PPM DIFFERENCE', 'PEPTIDE'])

        if 'TITLE' in assembly_df:
            assembly_df = assembly_df.drop(columns=['TITLE'])

        assembly_df = assembly_df.rename({'CANONICAL SEQ.': 'PEPTIDE'}, axis='columns')

        assembly_df['POSITIONAL SCORES'], assembly_df['SCORE']  = (
            assembly_df['SCORE'], assembly_df['POSITIONAL SCORES'])

        assembly_df['Area'] = 1
        log_filepath = ''
        result = {}

        try:
            assembly_df.to_csv(assembly_filename, index=False, header=True)
            log_filepath = str(self.output_folder / f'{self.output_filename}_assembly.log')

            with open (log_filepath, 'w') as log_h:
                subprocess.run(
                            ('java', '-jar', f'{self.alps_executable}', str(assembly_filename),
                             str(self.kmers), str(self.n_contigs), '>>', log_filepath), stdout=log_h)

        except Exception as e:
            logger.info(f'Peptide assembly error: {e}')

        finally:
            assembled_fasta = glob.glob(f'{self.output_folder}/*.fasta')

            if not assembled_fasta:
                logger.info(f'Peptide assembly error. Somthing wrong... See log file for details')

            else:
                for file in assembled_fasta:
                    contigs = npy.read_fasta(file)
                    try:
                        key = list(result.keys())[-1] + 1
                    except IndexError:
                        key = 1
                    result.update({str(key + i): contigs[k] for i, k in enumerate(contigs)})

                logger.info('Peptide assembly successfully completed')

                if os.path.exists(log_filepath):
                    os.remove(log_filepath)

            if os.path.exists(assembly_filename):
                os.remove(assembly_filename)

        return result

    def __map_proteins(self, query:dict)->pd.DataFrame:
        logger.info(f'Start protein inference process')

        blast = npy.blast(query=query,
                            database=str(self.proteins_fasta_path),
                            minIdentity=float(self.minIdentity),
                            maxAccepts=5,
                            alphabet="protein")

        try:
            if len(blast['QueryId']) == 0:
                logger.info(f'Protein search returned no results. No proteins were found '
                            f'with identity {self.minIdentity}')
                return pd.DataFrame()

        except KeyError:
            logger.info(f'Protein search returned no results. No proteins were found '
                        f'with identity {self.minIdentity}')
            return pd.DataFrame()


        output = {'id':[], 'peptide': [], 'protein_id':[],  'protein_name':[], 'score':[]}

        for i in range(len(blast['QueryId'])):
            target_match = blast['TargetMatchSeq'][i]
            score = blast['Identity'][i]
            query_id = blast['QueryId'][i]
            try:
                target_protein = blast['TargetId'][i].split('|')
                protein_id = target_protein[1]
                protein_name = target_protein[2]
            except (IndexError, KeyError, Exception):
                protein_id = protein_name = blast['TargetId'][i]

            output['id'].append(query_id)
            output['peptide'].append(target_match)
            output['score'].append(score)
            output['protein_id'].append(protein_id)
            output['protein_name'].append(protein_name)

        output_df = pd.DataFrame.from_dict(output)

        return output_df


    def inference_proteins(self, protein_df:pd.DataFrame):

        logger.info('Solve protein problem network')

        inference = ProteinInference(protein_map_df=protein_df,
                                     output_folder=str(self.output_folder),
                                     output_filename=str(self.output_filename)
                                     )
        inference.solve()
        logger.info('All pipeline task has been completed')
