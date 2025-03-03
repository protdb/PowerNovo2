import argparse
import glob
import logging
import os.path
import sys

from powernovo2.config.default_config import setup_run_environment
from powernovo2.inference import PWNInference

logger = logging.getLogger("powernovo2")



def process_folder(input_folder: str, configs:dict, callback_fn: callable = None):
    files = glob.glob(f'{input_folder}/*.mgf')

    if not files:
        logger.error(f"The mgf files in the specified folder were not found: {input_folder}")
        sys.exit(1)
    logger.info(f"Process all files in folder: {input_folder}")

    for file in files:
        file_size = os.path.getsize(file)
        logger.info(f"FILE: {os.path.basename(file)} ({round(file_size / (pow(1024, 2)), 2)} MB)")

    inference = PWNInference(configs=configs, callback_fn=callback_fn)
    inference.load_models()

    for file in files:
        inference.run(input_file=file)


def process_file(input_file: str, configs: dict, callback_fn: callable = None):
    inference = PWNInference(configs=configs, callback_fn=callback_fn)
    inference.load_models()
    inference.run(input_file=input_file)


def run_inference(inputs: str,
                  working_folder: str = 'powernovo2_work',
                  output_folder: str = '',
                  batch_size: int = 16,
                  use_assembler: bool = False,
                  protein_inference: bool = True,
                  fasta_path: str = '',
                  num_contigs: int = 600,
                  contigs_kmer: list = 7,
                  annotated_spectra: bool = False,
                  callback_fn: callable = None  # The function to which the prediction results will be returned
                  ):

    """Setup config"""

    configs = setup_run_environment(
        working_folder=working_folder,
        output_folder=output_folder,
        batch_size=batch_size,
        use_assembler=use_assembler,
        protein_inference=protein_inference,
        fasta_path=fasta_path,
        num_contigs=num_contigs,
        contigs_kmer=contigs_kmer,
        annotated_spectra=annotated_spectra
    )



    if os.path.isfile(inputs):
        process_file(input_file=inputs, configs=configs, callback_fn=callback_fn)
    elif os.path.isdir(inputs):
        process_folder(input_folder=inputs, configs=configs, callback_fn=callback_fn)
    else:
        logger.error('Invalid inputs. Either an mgf file or a directory containing such files must be specified')


def main():
    parser = argparse.ArgumentParser(description="Start PowerNovo pipeline")
    parser.add_argument('inputs', type=str, help="Path to the mgf file or path to the folder containing the list of "
                                                 "*.mgf files")
    parser.add_argument('-w', '--working_folder', type=str, help='Working folder for download models, ALPS, etc.',
                        required=False, default='powernovo2_work')

    parser.add_argument('-o', '--output_folder', type=str, help='Output folder [optional]', required=False, default='')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size', required=False, default=16)
    parser.add_argument('-a', '--annotated_spectra', action='store_true', help='Specify if mgf is annotated',
                        required=False, default=False)
    parser.add_argument('-alps', '--use_assembler',action='store_true', help='Use ALPS assembler [optional]',
                        required=False, default=False)
    parser.add_argument('-c', '--num_contigs', type=int, help='Number of generated contigs',
                        required=False, default=600)
    parser.add_argument('-k', '--contigs_kmer', type=int, help='Contigs kmer size',
                        required=False, default=7)
    parser.add_argument('-p', '--protein_inference', action='store_true', help='Use protein inference algorithm [optional]',
                        required=False, default=False)
    parser.add_argument('-f', '--fasta_path', type=str, help="Path to the fasta file that is used for "
                                                                 "peptide-protein mappings. "
                                                                 "If not specified will be used"
                                                                 "UP000005640_9606.fasta",
                        required=False, default='')

    args = parser.parse_args()

    inputs = args.inputs

    if not os.path.exists(inputs):
        logger.error(f"The specified file or folder was not found: {args.input}")
        sys.exit(1)

    run_inference(
        inputs=args.inputs,
        working_folder=args.working_folder,
        annotated_spectra=args.annotated_spectra,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        use_assembler=args.use_assembler,
        protein_inference=args.protein_inference,
        fasta_path=args.fasta_path,
        num_contigs=args.num_contigs,
        contigs_kmer=args.contigs_kmer
    )


if __name__ == '__main__':
    main()
