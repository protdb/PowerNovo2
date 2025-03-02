import glob
import logging
import os.path
import shutil
import zipfile
from pathlib import Path
from typing import Any

import requests
from requests import HTTPError

FIGSHARE_ID  = 52709036

MIN_PEP_LEN = 5
MAX_PEP_LEN = 48
NUM_TOKENS = 33

logger = logging.getLogger("powernovo2")

DEFAULT_TRAIN_PARAMS ={'environment': {'checkpoint_folder': '',
                                       'device': 'auto',
                                       'pretrained_model': '',
                                       'n_workers': 'auto'},
                       'datasets': {'train_dataset_path': '',
                                    'val_dataset_path': ''},
                       ''
                       'hyper_params': {'scheduler': {'type':'plateau', 'warmup':  100_000,
                                        'max_iters': 600_000},
                                        'precision': 'high',
                                        'epochs': 64,
                                        'patience': 8,
                                        'min_delta': 0.0001,
                                        'batch_size':8,
                                        'kl_warmups': 50_000,
                                        'n_workers': 'auto',
                                        'lr': 1e-5, },
                       }

BASIC_MODEL_PARAMS = {'encoder': {'n_layers': 8,
                                 'n_heads': 8,
                                 'dim': 512,
                                 'dropout': 0.33,
                                 'max_charge': 10.0
                                 },
                      'decoder': {'n_layers': 8,
                                 'n_heads': 8,
                                 'label_smoothing': 0.1,
                                  'dropout': 0.33,
                                 'dim': 512
                                  },
                      'prior_params': {
                        "length_predictor": {
                          "diff_range": MAX_PEP_LEN // 2,
                          "dropout": 0.1,
                          "n_mix": 1,
                          "features": 256,
                          "max_tgt_length": MAX_PEP_LEN,
                          "min_tgt_length": MIN_PEP_LEN
                        },
                        "flow": {
                          "levels": 3,
                          "num_steps": [4, 4, 2],
                          "factors": [2, 2],
                          "hidden_features": 256,
                          "transform": "affine",
                          "coupling_type": "self_attn",
                          "heads": 8,
                          "pos_enc": "attn",
                          "max_length": MAX_PEP_LEN,
                          "dropout": 0.33,
                          "inverse": True,
                          "features": 256,
                          "src_features": 256
                        }
                      },

                    'posterior_params': {
                        "num_layers": 4,
                        "heads": 8,
                        "max_length": MAX_PEP_LEN,
                        "dropout": 0.33,
                        "vocab_size": NUM_TOKENS,
                        "embed_dim":256,
                        "padding_idx": 0,
                        "latent_dim": 256,
                        "hidden_size": 512
                      }

                }

DEFAULT_RUN_PARAMS ={
    'environment': {
        'working_folder': 'powernovo2_work',
        'models_folder': 'models',
        'base_model': 'base_model.ckpt',
        'device': 'auto',
        'num_workers': 'auto',
        'batch_size': 16,
        'annotated_spectra': False
    },
    'inference': {
        'use_alps': False,
        'protein_inference': True,
        'num_contigs': 600,
        'kmers': 7,
        'knapsack_iter': 1,
        'assembler_folder': 'assembler',
        'alps_executable': 'ALPS.jar',
        'fasta_path': 'default',
        'proteins_minIdentity': 0.75,
        'max_peptide_len': MAX_PEP_LEN,
        'validation_mode': False
    },

    'output': {
        'output_folder_name': 'denovo_output',
        'preprocessing_folder': 'index',
    }
}

def setup_train_environment(**kwargs: Any)-> dict:
    train_config = {}
    model_config = {}
    if not kwargs:
        train_config = DEFAULT_TRAIN_PARAMS
        model_config = BASIC_MODEL_PARAMS
    else:
        pass
    return {'train_config': train_config, 'model_config': model_config}



def retrieve_data_from_figshare(working_folder: Path):
    figshare_id = FIGSHARE_ID
    temporary_folder = working_folder / 'tmp_'
    temporary_folder.mkdir(exist_ok=True)

    url = f"https://figshare.com/ndownloader/files/{figshare_id}"
    headers = {'Content-Type': 'application/json'}
    try:
        logger.info(f'Start downloading https://figshare.com/ndownloader/files/{figshare_id}')
        response = requests.request('GET', url, headers=headers, data=None)
        datafile = temporary_folder / 'model_data.zip'
        with open(datafile, 'wb') as fh:
            fh.write(response.content)
        assert os.path.exists(datafile)

        logger.info('Download complete')
        logger.info('Extract data...')
        datafile = temporary_folder / 'model_data.zip'
        with zipfile.ZipFile(datafile, 'r') as zip_fh:
            zip_fh.extractall(temporary_folder)

        sub_folders = next(os.walk(temporary_folder))[1]
        for folder in sub_folders:
            dst_folder = temporary_folder / folder
            if os.path.exists(working_folder / folder):
                shutil.rmtree(working_folder / folder)
            shutil.move(dst_folder, working_folder)

        if os.path.exists(temporary_folder):
            shutil.rmtree(temporary_folder)

        logger.info('Done')

    except (HTTPError, IOError, Exception, AssertionError) as e:
        logger.error("It is not possible to retrieve model data from Figshare. Try downloading them manually:"
                     "10.6084/m9.figshare.25329586")
        raise e



def setup_run_environment(**kwargs: Any)-> dict:

    run_config = DEFAULT_RUN_PARAMS
    model_config = BASIC_MODEL_PARAMS


    if kwargs:
        working_folder = kwargs['working_folder']

        if working_folder == run_config['environment']['working_folder']:
            if not os.path.exists(working_folder):
                logger.info(f'The working directory is not specified in the startup parameters. '
                            f'A default working directory will be created. {working_folder}')

                working_folder = Path(working_folder)
                working_folder.mkdir(exist_ok=True)
            run_config['environment']['working_folder'] = str(working_folder)
        else:
            run_config['environment']['working_folder'] = kwargs['working_folder']

        output_folder = kwargs['output_folder']

        if output_folder:
           run_config['output']['output_folder'] = output_folder

        try:
            run_config['environment']['batch_size'] = kwargs['batch_size']
            run_config['environment']['annotated_spectra'] = kwargs['annotated_spectra']
            run_config['inference']['protein_inference'] = kwargs['protein_inference']
            run_config['inference']['fasta_path'] = kwargs['fasta_path']
            run_config['inference']['use_alps'] = kwargs['use_assembler']
            run_config['inference']['num_contigs'] = kwargs['num_contigs']
            run_config['inference']['kmers'] = kwargs['contigs_kmer']

        except KeyError:
            logging.info('Error while parse config parameters. Use default config')
            run_config = DEFAULT_RUN_PARAMS

    working_folder = Path(str(run_config['environment']['working_folder']))


    if not os.path.exists(working_folder):
        raise FileNotFoundError(f"Working folder not found: {working_folder}")

    test_ = working_folder / str(run_config['environment']['models_folder'])
    test_ =  test_ / str(run_config['environment']['base_model'])

    if not os.path.exists(test_):
        retrieve_data_from_figshare(working_folder)



    if 'base_model_path' not in run_config['environment']:
        models_folder = working_folder / str(run_config['environment']['models_folder'])

        if not os.path.exists(models_folder):
            raise FileNotFoundError(f"Models folder not found: {models_folder}")

        base_model_path = models_folder / str(run_config['environment']['base_model'])

        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Model path not found: {base_model_path}")

        run_config['environment']['base_model_path'] = str(base_model_path)
    else:
        base_model_path = run_config['environment']['base_model_path']
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Model path not found: {base_model_path}")

    if 'alps_executable' not in run_config['inference']:
        if run_config['inference']['use_alps']:
            alps_folder = str(run_config['inference']['assembler_folder'])
            alps_folder = working_folder / alps_folder

            if not os.path.exists(alps_folder):
                raise FileNotFoundError(f"The path to the ALPS assembler folder was not found: {alps_folder}. "
                                        "If you don't want to use ALPS set use_alps = False")

            alps_executable = str(run_config['alps_executable'])

            alps_executable = Path(alps_folder) / alps_executable

            if not os.path.exists(alps_executable):
                raise FileNotFoundError(f"The path to the ALPS assembler executable file was not found: {alps_executable}." 
                                        "If you don't want to use ALPS set use_alps = False")

            run_config['inference']['alps_executable'] = str(alps_executable)

        else:
            run_config['inference']['alps_executable'] = ''

    elif run_config['inference']['use_alps']:
        alps_executable =  run_config['inference']['alps_executable']
        alps_folder = str(run_config['inference']['assembler_folder'])
        alps_folder = working_folder / alps_folder
        alps_executable = Path(alps_folder) / alps_executable

        if not os.path.exists(alps_executable):
            raise FileNotFoundError(f"The path to the ALPS assembler executable file was not found: {alps_executable}."
                                    "If you don't want to use ALPS set use_alps = False")

        run_config['inference']['alps_executable'] = str(alps_executable)
    else:
        run_config['inference']['alps_executable'] = ''

    if  run_config['inference']['protein_inference']:
        fasta_path = run_config['inference']['fasta_path']

        if fasta_path == 'default' or not fasta_path:
            fasta_check = glob.glob(f'{working_folder}/database/*.fasta')

            if fasta_check:
                fasta_path = fasta_check[0]
                logger.info('The FASTA file is specified in the startup parameters. '
                            f'The default file will be used for protein search :{fasta_path}')

        if not os.path.exists(fasta_path):
            raise FileNotFoundError("The path to the FASTA file for protein search is either not "
                                    f"specified or incorrect: {fasta_path}. Please provide the correct path or set "
                                    "the protein_inference option to False.")

        run_config['inference']['fasta_path'] = fasta_path


    return {'run_config': run_config, 'model_config': model_config}



aa_residues = {'G': 57.021463735,
            'A': 71.037113805,
            'S': 87.032028435,
            'P': 97.052763875,
            'V': 99.068413945,
            'T': 101.047678505,
            'C': 103.009184505,
            'L': 113.084064015,
            'N': 114.04292747,
            'D': 115.026943065,
            'Q': 128.05857754,
            'K': 128.09496305,
            'E': 129.042593135,
            'M': 131.040484645,
            'H': 137.058911875,
            'F': 147.068413945,
            'R': 156.10111105,
            'Y': 163.063328575,
            'W': 186.07931298,
            '[Acetyl]-': 42.010565,
            '[Carbamyl]-': 43.005814,
            '[Ammonia-loss]-': -17.026549,
            '[+25.980265]-': 25.980265,
            'M[Oxidation]': 147.03539964499998,
            'N[Deamidated]': 115.02694346999999,
            'Q[Deamidated]': 129.04259353999998,
            'C[Carbamidomethyl]': 160.030648505,
            'S[Phospho]': 166.998359435,
            'T[Phospho]': 181.014009505,
            'Y[Phospho]': 243.029659575}
