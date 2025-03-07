import logging
import os.path
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from powernovo2.config.default_config import setup_run_environment, MIN_PEP_LEN
from powernovo2.knapsack.knapsack_solver import KnapsackSolver
from powernovo2.models.base_model import PWNFlow
from powernovo2.modules.data import preprocessing
from powernovo2.modules.data.spectrum_datasets import AnnotatedSpectrumDataset, SpectrumDataset
from powernovo2.peptides.peptide_aggregator import PeptideAggregator
from powernovo2.utils.utils import to_canonical, PeptideHelper_, calc_ppm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("powernovo2")
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)


class PWNInference(object):
    def __init__(self, **kwargs):
        configs = kwargs['configs']
        self.env_config = configs['run_config']['environment']
        self.output_config = configs['run_config']['output']
        self.infer_config = configs['run_config']['inference']

        try:
            self.callback_fn = kwargs['callback_fn']
        except KeyError:
            self.callback_fn = None

        device_cfg = self.env_config['device']

        if device_cfg == 'auto':
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif device_cfg in ['cpu', 'cuda']:
            self.device = torch.device(device_cfg)
        else:
            raise AttributeError(f'Invalid device parameter in configs: {device_cfg}')

        self.helper = PeptideHelper_(device=self.device)
        self.tokenizer = self.helper.tokenizer

        self.solver = KnapsackSolver(k_iter=self.infer_config['knapsack_iter'])
        self.base_model = PWNFlow(configs=configs).to(self.device)
        self.base_model.eval()


    def load_models(self):
        try:
            loaded_model_path = self.base_model.load_pretrained()
            logger.info(f'Model loaded successfully {loaded_model_path}')
            logger.info(f'Use device: {self.device}')

        except (AssertionError, FileNotFoundError, Exception) as e:
            logger.error(e)
            raise e



    def run(self, input_file: Union[str, os.PathLike, Path]):

        annotated = self.env_config['annotated_spectra']

        loader = self.preprocessing(input_file=input_file, annotated=annotated)
        output_filename = Path(input_file).stem
        output_folder = self.output_config['output_folder_name']

        if not os.path.exists(output_folder):
            output_folder = Path(input_file).parent / output_folder
            output_folder.mkdir(exist_ok=True)
            output_folder = output_folder / output_filename
            output_folder.mkdir(exist_ok=True)

        self.process_data(loader=loader,
                          output_folder=output_folder,
                          output_filename=output_filename,
                          annotated=annotated)

    def preprocessing(self, input_file: os.PathLike,
                      annotated) -> DataLoader:
        try:
            assert os.path.exists(input_file), f"Error: input file not found: {input_file}"
            preprocessing_folder = self.output_config['preprocessing_folder']
            if not os.path.exists(preprocessing_folder):
                preprocessing_folder = Path(input_file).parent / preprocessing_folder
                preprocessing_folder.mkdir(exist_ok=True)

            preprocessing_file = f'{Path(input_file).stem}.hdf5'
            preprocessing_path = Path(preprocessing_folder) / preprocessing_file
            logger.info(f'Preprocessing input file: {input_file}')

            try:
                loader = self._create_loader(
                    input_file=input_file,
                    preprocessing_path=preprocessing_path,
                    overwrite=False,
                    annotated=annotated
                )
            except (ValueError, Exception) as e:
                logger.info('The datafile already exists, but the parameters are incompatible. Attempt to rebuild...')
                loader = self._create_loader(
                    input_file=input_file,
                    preprocessing_path=preprocessing_path,
                    overwrite=True,
                    annotated=annotated
                )

            logger.info('Preprocessing completed successfully')

        except (AssertionError, Exception) as e:
            logger.info(f'Unable to preprocessing input file: {input_file}')
            logger.error(e)
            raise e

        return loader

    def _create_loader(self,
                       input_file: os.PathLike,
                       preprocessing_path: os.PathLike,
                       overwrite: bool = False,
                       annotated: bool = False) -> DataLoader:
        try:
            if annotated:
                dataset = AnnotatedSpectrumDataset(tokenizer=self.tokenizer,
                                                   ms_data_files=input_file,
                                                   overwrite=overwrite,
                                                   index_path=preprocessing_path,
                                                   )
            else:
                dataset = SpectrumDataset(ms_data_files=input_file,
                                          overwrite=overwrite,
                                          preprocessing_fn=[
                                              preprocessing.set_mz_range(min_mz=140),
                                              preprocessing.scale_intensity(scaling="root"),
                                              preprocessing.scale_to_unit_norm,
                                          ],
                                          index_path=preprocessing_path)
        except (ValueError, Exception) as e:
            raise e

        num_workers = self.env_config['num_workers']
        num_workers = int(os.cpu_count() / 2) if num_workers  == 'auto' else int(num_workers)
        batch_size = int(self.env_config['batch_size'])
        loader_params = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True
        }
        loader = dataset.loader(**loader_params)
        return loader

    def process_data(self,
                     loader: DataLoader,
                     output_folder: Union[str, os.PathLike, Path],
                     output_filename: str,
                     annotated: bool = False):
        logger.info(f'Start processing. Total items: {len(loader)} ')

        peptide_aggregator = PeptideAggregator(
            config=self.infer_config,
            output_folder=output_folder,
            output_filename=output_filename)



        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            if None in batch:
                continue

            scan_ids = batch[-1]
            meta_len = 2 - annotated
            batch = batch[:len(batch) - meta_len]
            batch = [b.to(self.device) for b in batch]



            (completed,
             completed_mask,
             tokens,
             probability,
             uncompleted_px,
             _,
             _,
             precursors) = self.base_model.smart_sample(spectra=batch[0].float(), precursors=batch[1])


            if torch.numel(tokens):
                solved = torch.zeros_like(tokens)
                px_solved = torch.zeros_like(solved, dtype=torch.float32)

                bs, _ = tokens.size()
                solved_indices = torch.as_tensor(range(bs), dtype=torch.long, device=self.device)
                out = tokens
                lengths = (out > 0).sum(-1)
                px = None

                for k in range(self.solver.k_iter):
                    out, is_completed, px = self.solver.solve(tokens=out,
                                                              precursors=precursors,
                                                              probability=uncompleted_px,
                                                              lengths=lengths,
                                                              n_iter=k
                                                              )

                    completed_exist = is_completed.any()

                    if completed_exist:
                        solved[solved_indices[is_completed]] = out[is_completed]
                        px_solved[solved_indices[is_completed]] = px[is_completed]
                        out = out[~is_completed]
                        precursors = precursors[~is_completed]
                        solved_indices = solved_indices[~is_completed]
                        uncompleted_px = uncompleted_px[~is_completed]
                        lengths = lengths[~is_completed]
                        px = px[~is_completed]

                    if is_completed.all():
                        break

                if len(solved_indices) > 0:
                    assert px is not None
                    solved[solved_indices] = out
                    px_solved[solved_indices] = px

                completed[~completed_mask] = solved
                probability[~completed_mask] = px_solved


            if self.infer_config['validation_mode']:
                validation_file = Path(output_folder) / f'{output_filename}.csv'
                self.__eval_writer(prediction=completed,
                                   probability=probability,
                                   annotation=batch[2] if annotated else None,
                                   output_file=str(validation_file)
                                   )

                continue


            output_size = completed.size(0)
            precursors = batch[1]
            charges = precursors[:, 1].cpu().numpy()
            mass_sp = precursors[:, -1].cpu().numpy()
            probability = torch.nan_to_num(probability)
            predicted_sequences = self.tokenizer.detokenize(completed, trim_stop_token=True, join=True)
            scores = probability.cpu().numpy()
            if annotated:
                annotation = self.tokenizer.detokenize(batch[2], join=True, trim_stop_token=True)
            else:
                annotation = None
            probability = probability.cpu().numpy()

            for i in range(output_size):
                predicted_seq = predicted_sequences[i]
                predicted_seq = predicted_seq.replace(self.tokenizer.stop_token, '')
                canonical_seq = to_canonical(predicted_seq)
                ppm_diff = calc_ppm(canonical_seq, charges[i], mass_sp[i])
                score = scores[i]
                score = score[score > 0]
                score = np.mean(score) if len(score) > 0 else 0
                scan_id = scan_ids[i]
                record =peptide_aggregator.add_record(scan_id=scan_id,
                                                      annotation=annotation[i] if annotation is not None else '',
                                                      predicted_sequence=predicted_seq,
                                                      canonical_sequence=canonical_seq,
                                                      ppm_diff=ppm_diff,
                                                      aa_scores=probability[i],
                                                      score=score
                                                      )


                if self.callback_fn is not None and record:
                    self.callback_fn(record)


        peptide_aggregator.solve()

    def __eval_writer(self,
                      prediction:torch.Tensor,
                      probability:torch.Tensor,
                      annotation:torch.Tensor,
                      output_file:str
                      ):

        prediction_tokens = self.tokenizer.detokenize(prediction)

        if annotation is not None:
            true_tokens = self.tokenizer.detokenize(annotation)
        else:
            true_tokens = None

        probs = probability.cpu().numpy()
        batch_size = prediction.size(0)

        is_exist = os.path.exists(output_file)

        with open(output_file, 'a+') as fh:
            if not is_exist:
                fh.writelines(['TITLE\tDENOVO\tPositional Score\n'])
            for i in range(batch_size):

                if true_tokens is not None:
                    true_seq = to_canonical(true_tokens[i])
                else:
                    true_seq = ' '
                pred_seq = to_canonical(prediction_tokens[i])
                if len(pred_seq) <= MIN_PEP_LEN - 2:
                    continue
                px = probs[i].tolist()[:len(pred_seq)]
                px = [f'{s:2f}' for s in px]
                p = ' '.join(px)
                p = p.strip()
                output_record = '\t'.join([true_seq, pred_seq, p])
                fh.writelines(output_record + '\n')



if __name__ == '__main__':
   # test_file = "/home/dp/Data/powernovo2/test/m_musculus_Orbitrap_HCD.mgf"
   # test_file = "/home/dp/Data/powernovo2/test/human_Orbitrap_HCD_good.mgf"
    test_file = "/home/dp/Data/powernovo2/test/example.mgf"
   # test_file = "/home/dp/Data/powernovo2/test/vigma.mgf"
   # test_file = "/home/dp/Data/benchmark/speed_test/datasets/human_hcd_speed_test_10000.mgf"
   # test_file = "/home/dp/Data/benchmark/datasets/PXD000602/datasets/UPS2_A/v120627v04.mgf"


    cfgs = setup_run_environment()
    pwn = PWNInference(configs=cfgs)
    pwn.load_models()
    pwn.run(input_file=test_file)
