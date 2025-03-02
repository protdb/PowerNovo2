import os
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import pairwise2

from powernovo2.utils.utils import to_canonical


def pairwise_alignment(x_true, x_denovo):
    alignments = pairwise2.align.globalxx(to_canonical(x_true), x_denovo)
    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2, score, start, end = best_alignment
    matches = sum(res1 == res2 for res1, res2 in zip(aligned_seq1, aligned_seq2))
    identity = np.round(matches / len(aligned_seq1), 2)
    return identity


class IdentityCalculator(object):
    def __init__(self):
        self.identity_rec = {'SERVICE':[], 'FILE': [], 'SET':[], 'SUBSET':[]}

        for i in range(0, 20):
            self.identity_rec.update({str(i / 20):[]})

        self.identity_rec.update({'1.0': []})

    def calculate_identity(self, file, service, set_name, subset_name):
        try:
            df = pd.read_csv(file, index_col=0, header="infer")
            tmp = df['TITLE']
        except KeyError:
            df = pd.read_csv(file, header="infer")
        if df.empty:
            return
        try:
            identity = df['IDENTITY'].to_numpy()
            print('PASS')
            print()
        except KeyError:
            try:
                df['IDENTITY'] = df.apply(lambda x: pairwise_alignment(x['TITLE'], x['DENOVO']), axis=1)
                df.to_csv(file, index=False, header=True)
                identity = df['IDENTITY'].to_numpy()
            except TypeError:
                print(df)
                raise

        self.identity_rec['SERVICE'].append(service)
        self.identity_rec['FILE'].append(os.path.basename(file))
        self.identity_rec['SET'].append(set_name)
        self.identity_rec['SUBSET'].append(subset_name)

        total = df['TITLE'].count()

        for i in range(0, 20):
            threshold = i / 20.0
            values_greater_or_eq = (identity >= threshold).sum()
            key = str(threshold)
            self.identity_rec[key].append(np.round(values_greater_or_eq / total, 2))

        values_greater_or_eq = (identity == 1.0).sum()
        self.identity_rec['1.0'].append(np.round(values_greater_or_eq / total, 2))

    def save(self, output_file):
        df = pd.DataFrame.from_dict(self.identity_rec)
        df.to_csv(output_file, index=False, header=True)



def calculate_identity(dataset_root_folder: str):
    files = []

    for dirpath, dirnames, filenames in os.walk(dataset_root_folder):
        files.extend([os.path.join(dirpath, filename) for filename in filenames if filename.endswith('.csv')])

    calculator = IdentityCalculator()
    for file in files:
        sp = file.split('/')
        service = sp[-4]
        set_name = sp[-3]
        subset_name = sp[-2]

        calculator.calculate_identity(file, service, set_name, subset_name)

    identity_output_file = Path(DATASET_PATH30).parent / 'identity_(len=30).csv'


    calculator.save(identity_output_file)



DATASET_PATH30 = "/benchmark/results/denovo_results"


if __name__ == '__main__':
    calculate_identity(DATASET_PATH30)
