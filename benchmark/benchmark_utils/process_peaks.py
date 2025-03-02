import os
import pickle
from pathlib import Path

import npysearch as npy
import numpy as np
import pandas as pd

from powernovo2.utils.utils import to_canonical


class PeaksMeta(object):
    def __init__(self):
        self.all_candidates_path = None
        self.denovo_path = None
        self.dataset_filename = None
        self.dataset_desc = None
        self.setname = None
        self.subset_name = None

def build_peaks_table(dataset_folder: str, dataset_desc_folder, output_file: str):
    denovo_files = []
    all_candidates_files = []
    data = []

    for dirpath, dirnames, filenames in os.walk(dataset_folder):
        denovo_files.extend([os.path.join(dirpath, filename) for filename in filenames
                      if filename.endswith('denovo.csv')])

        all_candidates_files.extend([os.path.join(dirpath, filename) for filename in filenames
                             if filename.endswith('denovoAllCandidates.csv')])

    assert len(denovo_files) == len(all_candidates_files)

    for i, file in enumerate(denovo_files):
        meta = PeaksMeta()
        meta.denovo_path = file
        meta.all_candidates_path = all_candidates_files[i]
        df = pd.read_csv(file)
        source_files = df['Source File'].to_list()
        if not source_files:
            continue
        source_file = source_files[0]
        df_test = pd.read_csv(all_candidates_files[i])
        test_file_ = df_test['Source File'].to_list()[0]
        assert test_file_ == source_file
        meta.dataset_filename = source_file

        desc_file = []

        for f in Path(dataset_desc_folder).rglob(f'{Path(source_file).stem}.csv'):
            desc_file.append(f)

        assert len(desc_file) == 1

        desc_file = str(desc_file[0])
        meta.dataset_desc = desc_file

        desc_file_sp = desc_file.split('/')
        setname = desc_file_sp[-3]
        subset_name = desc_file_sp[-2]
        meta.setname = setname
        meta.subset_name = subset_name
        data.append(meta)

        with open(output_file, 'wb+') as fh:
            pickle.dump(data, fh)


class PEAKSSearch(object):
    def __init__(self, denovo_meta):
        self.df_denovo = pd.read_csv(denovo_meta.denovo_path)

    def search_denovo(self, scan_id):
        df = self.df_denovo.loc[self.df_denovo['Scan'] == scan_id]
        peptide = ''
        scores = ''
        if not df.empty:
            peptide = df['Peptide'].tolist()[0]
            scores = df['local confidence (%)'].tolist()[0]

        return peptide, scores



    def search(self, scan_id):
        peptide, scores = self.search_denovo(scan_id)

        if peptide:
           scores = np.array(list(map(int, scores.split(' '))), dtype=float)
           scores /= 100
           scores = np.round(scores, 2)
           scores = scores.tolist()
           scores = map(str, scores)
           scores = ' '.join(scores).strip()
        else:
            peptide = 'NOT PREDICTED'
            scores = ' '.join(['0.0' for _ in range(len(peptide))])
            scores = scores.strip()


        return peptide, scores

    def search_blast(self, true_peps: list):
        denovo_peps = self.df_denovo['Peptide'].to_list()
        scan_ids = self.df_denovo['Scan'].to_list()
        denovo_query = {}
        for i, scan_id in enumerate(scan_ids):
            denovo_query.update({str(scan_id): denovo_peps[i]})

        true_query = {str(i): convert_(true_peps[i]) for i in range(len(true_peps))}


        q_results = npy.blast(query=denovo_query,
                            database=true_query,
                            minIdentity=0.75,
                            maxAccepts=1,
                            alphabet="protein")

        results = {'TITLE': [], 'DENOVO': [], 'Positional Score': []}
        not_pred = set(true_query.keys())


        if q_results:
            q_ids = q_results['QueryId']
            target_ids = q_results['TargetId']
            tmp_dict = {(q, t, denovo_query[q], true_query[t]) for q, t in zip(q_ids, target_ids)}

            for item in tmp_dict:
                q_idx, t_idx, denovo, true_ = item
                peptide, score = self.search(int(q_idx))
                results['TITLE'].append(true_)
                results['DENOVO'].append(peptide)
                results['Positional Score'].append(score)
            not_pred = not_pred.difference(set(target_ids))

        for i in not_pred:
            true_ = true_query[i]
            score = ' '.join(['0.0' for _ in range(len('NOT PREDICTED'))])
            score = score.strip()
            results['TITLE'].append(to_canonical(true_))
            results['DENOVO'].append('NOT PREDICTED')
            results['Positional Score'].append(score)
        return results


def convert_(s: str):
    s = to_canonical(s)
    s = s.replace('I', 'L')
    return s









def process_peaks(peaks_meta_path:str, output_folder:str):
    with open(peaks_meta_path,'rb') as fh:
        data = pickle.load(fh)

    for rec in data:
        dataset_file = rec.dataset_desc
        dataset_df = pd.read_csv(dataset_file)
        denovo_search = PEAKSSearch(rec)
        setname = rec.setname
        subset_name = rec.subset_name

        scans = dataset_df['scan_id'].to_list()
        true_peps = dataset_df['peptide'].to_list()

        results = {'TITLE': [], 'DENOVO': [], 'Positional Score': []}

        if setname == 'nine-species':
            for i, scan_id in enumerate(scans):
                true_pep = convert_(true_peps[i])
                peptide, scores = denovo_search.search(scan_id)
                results['TITLE'].append(true_pep)
                results['DENOVO'].append(peptide)
                results['Positional Score'].append(scores)


        else:
            results = denovo_search.search_blast(true_peps)

        df = pd.DataFrame.from_dict(results)

        target_folder = Path(output_folder) / setname
        target_folder.mkdir(exist_ok=True)
        target_folder = target_folder / subset_name
        target_folder.mkdir(exist_ok=True)
        target_file = target_folder / f'{Path(rec.dataset_desc).stem}.csv'
        df.to_csv(target_file, index=False, header=True)












DATASET_FOLDER = '/home/dp/Data/benchmark/benched/peaks_output'
DATASET_DESC_FOLDER = "/home/dp/Data/benchmark/results/datasets_desc"
PEAKS_META = "/home/dp/Data/benchmark/results/peaks_meta/peaks_meta.pkl"
PEAKS_FOLDER = '/home/dp/Data/benchmark/results/denovo_results/peaks'

if __name__ == '__main__':
    #build_peaks_table(DATASET_FOLDER, DATASET_DESC_FOLDER, PEAKS_META)
    process_peaks(peaks_meta_path=PEAKS_META, output_folder=PEAKS_FOLDER)
