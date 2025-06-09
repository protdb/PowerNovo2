from pathlib import Path
import pandas as pd
from Bio import SeqIO
from collections import defaultdict
import re
import ast
from pyteomics import mgf


class DeNovoNSAFAnalyzer:
    def __init__(self, peptide_file, mgf_file, fasta_file, output_file):
        self.peptide_file = Path(peptide_file)
        self.mgf_file = Path(mgf_file)
        self.fasta_file = Path(fasta_file)
        self.output_file = Path(output_file)
        self._validate_inputs()

        self.protein_lengths = {}
        self.protein_sequences = {}
        self.scan_intensities = {}


    def _validate_inputs(self):
        if not self.peptide_file.exists():
            raise FileNotFoundError(f"Peptide file not found: {self.peptide_file}")
        if not self.mgf_file.exists():
            raise FileNotFoundError(f"MGF file not found: {self.mgf_file}")
        if not self.fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_file}")

    def _load_protein_data(self):
        for record in SeqIO.parse(self.fasta_file, "fasta"):
            header_parts = record.description.split('|')
            protein_id = header_parts[1] if len(header_parts) > 2 else record.id
            self.protein_lengths[protein_id] = len(record.seq)
            self.protein_sequences[protein_id] = str(record.seq)

    def _load_mgf_intensities(self):
        with mgf.read(str(self.mgf_file)) as reader:
            for i, spectrum in enumerate(reader):
                scan_id = str(i)
                intensity = spectrum.get('params', {}).get('pepmass', [None, None])[1]
                if intensity is not None:
                    self.scan_intensities[scan_id] = float(intensity)

    @staticmethod
    def _extract_scan_id(ids_str):
        try:
            if isinstance(ids_str, str):
                if ids_str.startswith('[') and ids_str.endswith(']'):
                    return ast.literal_eval(ids_str)[0]
                match = re.search(r'(\d+)', ids_str)
                return int(match.group(1)) if match else None
            return ids_str[0] if isinstance(ids_str, list) else None
        except:
            return None

    @staticmethod
    def _calculate_coverage(protein_seq, peptides):
        covered = set()
        for peptide in peptides:
            start = protein_seq.find(peptide)
            if start != -1:
                covered.update(range(start, start + len(peptide)))
        return len(covered) / len(protein_seq) * 100 if protein_seq else 0

    def process(self):
        self._load_protein_data()
        self._load_mgf_intensities()

        peptide_df = pd.read_csv(self.peptide_file)
        peptide_df['scan_id'] = peptide_df['ids'].apply(self._extract_scan_id)

        peptide_df['intensity'] = peptide_df['scan_id'].map(self.scan_intensities).fillna(0)

        protein_peptides = defaultdict(list)
        for _, row in peptide_df.iterrows():
            if pd.notna(row['major']):
                protein_peptides[row['major']].append(row['sequence'])

        results = []
        total_saf = 0
        saf_values = []

        for (protein_id, protein_name), group in peptide_df.groupby(['major', 'major_name']):
            length = self.protein_lengths.get(protein_id, 1)
            peptides = protein_peptides.get(protein_id, [])
            unique_peptides = list(set(peptides))
            sum_intensity = group['intensity'].sum()
            saf = sum_intensity / length if length > 0 else 0
            coverage = self._calculate_coverage(self.protein_sequences.get(protein_id, ''), peptides)

            results.append({
                'PROTEIN_ID': protein_id,
                'PROTEIN_NAME': protein_name.split()[0] if isinstance(protein_name, str) else protein_name,
                'PEPTIDES': ';'.join(peptides),
                'UNIQUE_PEPTIDES': ';'.join(unique_peptides),
                'TOTAL_PEPTIDES': len(peptides),
                'UNIQUE': len(unique_peptides),
                'SEQUENCE_COVERAGE': coverage,
                'SUM_INTENSITY': sum_intensity,
                'LENGTH': length,
                'SAF': saf
            })
            saf_values.append(saf)
            total_saf += saf

        output_data = []
        for stat, saf in zip(results, saf_values):
            nsa_f = saf / total_saf if total_saf > 0 else 0
            output_data.append({
                'PROTEIN_ID': stat['PROTEIN_ID'],
                'PROTEIN_NAME': stat['PROTEIN_NAME'],
                'PEPTIDES': stat['PEPTIDES'],
                'UNIQUE_PEPTIDES': stat['UNIQUE_PEPTIDES'],
                'TOTAL_PEPTIDES': stat['TOTAL_PEPTIDES'],
                'UNIQUE': stat['UNIQUE'],
                'SEQUENCE_COVERAGE': stat['SEQUENCE_COVERAGE'],
                'SUM_INTENSITY': stat['SUM_INTENSITY'],
                'LENGTH': stat['LENGTH'],
                'SAF': stat['SAF'],
                'NSAF': nsa_f,
                'RELATIVE_ABUNDANCE': nsa_f * 100,
                'FILE': self.peptide_file
            })

        pd.DataFrame(output_data).to_csv(self.output_file, index=False)


"""TEST """

if __name__ == '__main__':
    pep_file = "/home/dp/Data/powernovo2/test/denovo_output/KC_022/KC_022_peptide.csv"
    mgf_file = "/home/dp/Data/powernovo2/test/KC_022.mgf"
    fasta_file = "/home/dp/Data/powernovo2/database/UP000005640_9606.fasta"
    out = "/home/dp/Data/powernovo2/test/denovo_output/KC_022/KC_022_nsaf.csv"

    analyzer = DeNovoNSAFAnalyzer(peptide_file=pep_file,
                                  mgf_file=mgf_file,
                                  fasta_file=fasta_file,
                                  output_file=out
                                  )
    analyzer.process()
