import os
import subprocess
from copy import deepcopy
from pathlib import Path

import npysearch as npy
import numpy as np
import pandas as pd

from powernovo2.utils.utils import to_canonical


def canonical(x):
    y = to_canonical(x)
    return y

def list2str(x):
    x = x.replace(']', '')
    x = x.replace('[', '')
    x = x.split(',')
    x = list(map(float, x))
    x = np.round(x, 2)
    x = list(map(str, x))
    x = ' '.join(x)
    return x

def total_score(x):
    y = x.split(' ')
    y = map(float, y)
    y = np.mean(list(y))
    y = np.round(y, 2)
    return y


def ensurescore(x, y):
    y = y.split(' ')

    if len(x) < len(y):
        y = y[:len(x)]
    elif len(x) > len(y):
        print('>...........')
        raise
    y = ' '.join(y)
    return y

def process_peaks(peaks_folder, output_file):
    denovo_files = []

    for dirpath, dirnames, filenames in os.walk(peaks_folder):
        denovo_files.extend([os.path.join(dirpath, filename) for filename in filenames
                             if filename.endswith('.denovo.csv')])

    agg_df = []

    for file in denovo_files:
        df = pd.read_csv(file)
        df['Spectrum Name'] = np.arange(len(df))
        df['peaks Peptides'] = df['Peptide']
        df['peaks Positional Score'] = df['local confidence (%)']

        df = df[['Spectrum Name', 'peaks Peptides', 'peaks Positional Score']]
        agg_df.append(df)

    out_df = pd.concat(agg_df, axis='rows')

    out_df.to_csv(output_file, index=False)

def aggregate(root_folder):
    groups = {}
    denovo_files  = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        denovo_files.extend([os.path.join(dirpath, filename) for filename in filenames
                             if filename.endswith('.csv')])

    for file in denovo_files:
        service = str(Path(file).parent.parent.stem)
        service = service.replace('output', 'eval')
        file_key = Path(file).stem[-2:]

        if service not in groups:
            groups.update({service:{}})

        if file_key not in groups[service]:
            groups[service].update({file_key:[file]})
        else:
            groups[service][file_key].append(file)

    output_folder = Path(root_folder).parent / 'denovo_results_agg'
    output_folder.mkdir(exist_ok=True)

    for k, v in groups.items():
        service_folder = output_folder / k
        service_folder.mkdir(exist_ok=True)
        for gr in v:
            groups_folder = service_folder / gr
            groups_folder.mkdir(exist_ok=True)

            dfs = []
            service_name = k.split('_')[0]

            for gf in groups[k][gr]:
                df = pd.read_csv(gf, delimiter='\t')
                if service_name == 'powernovo1':
                    df['DENOVO'] = df['TITLE']

                df['TITLE'] = df.index
                df = df.rename(columns={'TITLE': 'Spectrum Name'})
                df = df.dropna(subset=['DENOVO'])
                df['DENOVO'] = df['DENOVO'].apply(canonical)

                if service_name == 'pepnet':
                    df['Positional Score'] = df['Positional Score'].apply(list2str)
                    df = df.drop(columns=['PPM Difference', 'Score'])

                df['Positional Score'] = df.apply(lambda x: ensurescore(x['DENOVO'], x['Positional Score']), axis=1)
                df[f'{service_name} Score'] = df['Positional Score'].apply(total_score)
                df['Area'] = 1
                df = df.rename(columns={'DENOVO': f'{service_name} Peptides',
                                        'Positional Score': f'{service_name} aaScore'})


                dfs.append(df)

            out_df = pd.concat(dfs, axis='rows')
            groups_folder = str(groups_folder)
            out_file = os.path.join(groups_folder,  f'agg_{gr}.csv')


            out_df.to_csv(out_file, index=False, header=True)



def extract_contigs(root_folder, alps_executable, kmer=6, n_contigs=5000):
    denovo_files = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        denovo_files.extend([os.path.join(dirpath, filename) for filename in filenames
                             if filename.endswith('.csv')])

    for file in denovo_files:
        log_filepath = file.replace('csv', 'log')
        print(file)
        subprocess.run(
            ('java', '-jar', f'{alps_executable}', str(file),
             str(kmer), str(n_contigs), '>>', log_filepath))


def score_contigs(root_folder, database):
    denovo_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        denovo_files.extend([os.path.join(dirpath, filename) for filename in filenames
                             if filename.endswith('.fasta')])

    db = npy.read_fasta(database)




    data = {}

    for file in denovo_files:
        service = str(Path(file).parent.parent.stem)
        service = service.split('_')[0]

        if not service in data:
            data.update({service:{'DENOVO': [], 'TRUE': [], 'ORGANISM':[], 'NAME':[]}})



        q1_file = file.replace('.k6.fasta', '')
        assert os.path.exists(q1_file)
        dfq = pd.read_csv(q1_file)
        q_list = dfq[f'{service} Peptides'].to_list()
        q1 = {str(idx):s for idx, s in enumerate(q_list) if len(s) >=5}

        res = npy.blast(query=q1,
                        database=db,
                        maxAccepts=1,
                        maxRejects=16,
                        minIdentity=0.95,
                        alphabet='protein'
                        )

        for i in range(len(res['QueryId'])):
            target_match = res['TargetMatchSeq'][i]
            query_match = res['QueryMatchSeq'][i]
            target_id = res['TargetId'][i]
            organism = 'Escherichia coli' if 'Escherichia coli' in target_id  else 'Homo sapiens'
            full_name = target_id
            data[service]['DENOVO'].append(query_match)
            data[service]['TRUE'].append(target_match)
            data[service]['ORGANISM'].append(organism)
            data[service]['NAME'].append(full_name)



    for srv, v in data.items():
        df = pd.DataFrame.from_dict(v)
        out_file = Path(root_folder) / f'PXD000602_{srv}_results.csv'
        df.to_csv(out_file, index=False)


def score_peptides(root_folder, database):
    denovo_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        denovo_files.extend([os.path.join(dirpath, filename) for filename in filenames
                             if filename.endswith('.csv') and 'k6' not in filename and 'agg' in filename])

    db = npy.read_fasta(database)

    data = {}

    for file in denovo_files:
        service = str(Path(file).parent.parent.stem)
        service = service.split('_')[0]

        if not service in data:
            data.update({service:{'DENOVO': [], 'TRUE': [], 'ORGANISM':[], 'NAME':[]}})



        dfq = pd.read_csv(file)
        print(file)
        q_list = dfq[f'{service} Peptides'].to_list()
        q_list = [q.replace('I', 'L') for q in q_list]

        queries = {str(idx):s for idx, s in enumerate(q_list) if len(s) >= 5}


        res = npy.blast(query=queries,
                        database=db,
                        maxAccepts=1,
                        maxRejects=16,
                        minIdentity=0.95,
                        alphabet='protein'
                        )

        for i in range(len(res['QueryId'])):
            target_match = res['TargetMatchSeq'][i]
            query_match = res['QueryMatchSeq'][i]
            target_id = res['TargetId'][i]
            organism = 'Escherichia coli' if 'Escherichia coli' in target_id  else 'Homo sapiens'
            full_name = target_id
            data[service]['DENOVO'].append(query_match)
            data[service]['TRUE'].append(target_match)
            data[service]['ORGANISM'].append(organism)
            data[service]['NAME'].append(full_name)



    for srv, v in data.items():
        df = pd.DataFrame.from_dict(v)
        out_file = Path(root_folder) / f'PXD000602_{srv}_results.csv'
        df.to_csv(out_file, index=False)


def calc_coverage(root_folder, database):
    denovo_files = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        denovo_files.extend([os.path.join(dirpath, filename) for filename in filenames
                             if filename.endswith('_results.csv')])

    db = npy.read_fasta(database)
    db_cov = {}
    db_out = {}

    for k, v in db.items():
        db_cov.update({k: np.zeros(len(v))})

    tmp_ = deepcopy(db)
    for k, v in tmp_.items():
        db.update({k: v.replace('I', 'L')})

    for file in denovo_files:
        service = file.split('_')[-2]

        if service not in db_out:
            db_out.update({service: deepcopy(db_cov)})

        df = pd.read_csv(file)
        human = df[df["ORGANISM"] == 'Homo sapiens']
        human_contigs = human['TRUE'].to_list()
        human_keys = human['NAME'].to_list()

        for i, key in enumerate(human_keys):
            try:
                fasta = db[key]
                key1 = key
            except KeyError:
                fasta, key1 = search_key(db, key)


            q = human_contigs[i]
            index_start = fasta.find(q)
            index_end = index_start + len(q)
            db_out[service][key1][index_start:index_end] = 1

    proteins = set()

    cov_arr = {'SERVICE': [], 'PROTEIN': [], 'COV': []}

    for service, cov in db_out.items():
        print(service.upper())

        for k, v in cov.items():
            if np.sum(v) > 0:
                coverage = np.round((np.sum(v) / len(v)) * 100, 2)
                print(f'{k}    Coverage: {coverage}%')
                proteins.add(k)
                cov_arr['SERVICE'].append(service.upper())
                cov_arr['PROTEIN'].append(k)
                cov_arr['COV'].append(coverage)


        print()

    cov_df = pd.DataFrame.from_dict(cov_arr)

    services = list(db_out.keys())
    services.sort()

    out_arr = {'PROTEIN_ID': [], 'PROTEIN_NAME':[]}
    out_arr.update({s.upper(): [] for s in  services})

    for protein in proteins:
        protein_id, protein_name = protein.split('|')
        out_arr['PROTEIN_ID'].append(protein_id)
        out_arr['PROTEIN_NAME'].append(protein_name)
        for service in services:
            cov_s = cov_df[cov_df['SERVICE'] == service.upper()]
            cov_ = cov_s[cov_s['PROTEIN'] == protein]
            value = 0 if cov_.empty else cov_['COV'].values[0]
            out_arr[service.upper()].append(value)


    output_file = Path(root_folder) / 'coverage_total.csv'
    df = pd.DataFrame.from_dict(out_arr)
    df.to_csv(output_file, index=False)



def search_key(db, key):
    for k, v in db.items():
        k1 = key.split(';')[0]
        s0, s1 = k1.split('|')

        if s0 in k1 and s1 in k:
            return v, k

    raise







PEAKS_FOLDER = "/Data/benchmark/results_pxd/benched_pxd/peaks_output"
PEAKS_AGG = "/Data/benchmark/results_pxd/denovo_results_agg/peaks_eval/01/agg_01.csv"
RESULT_FOLDER = "/dp/Data/benchmark/results_pxd/denovo_results"
RESULT_AGG_FOLDER = "/Data/benchmark/results_pxd/denovo_results_agg"
ALPS = "/Data/powernovo/assembler/ALPS.jar"
UPS_FASTA = "/Data/benchmark/results_pxd/fasta/UPS1_UPS2_Ecoli_469008_RefSeq.fasta"

if __name__ == '__main__':
    process_peaks(PEAKS_FOLDER, output_file=PEAKS_AGG)
    aggregate(root_folder=RESULT_FOLDER)
    #extract_contigs(RESULT_AGG_FOLDER, alps_executable=ALPS)
    #score_contigs(RESULT_AGG_FOLDER, UPS_FASTA)
    score_peptides(RESULT_AGG_FOLDER, UPS_FASTA)
    calc_coverage(RESULT_AGG_FOLDER, UPS_FASTA)



