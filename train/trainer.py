import argparse
import glob
import json
import os.path
import sys
import warnings
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import MulticlassAccuracy

from powernovo2.config.default_config import (DEFAULT_TRAIN_PARAMS,
                                              BASIC_MODEL_PARAMS, setup_train_environment, MAX_PEP_LEN, NUM_TOKENS)
from powernovo2.models.base_model import BaseModelEncoder, BaseModelDecoder, BaseModel
from powernovo2.modules.data.spectrum_datasets import AnnotatedSpectrumDataset
from powernovo2.modules.tokenizers.peptides import PeptideTokenizer
from train.schedulers import WarmupScheduler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



class BasicTrainingWrapper(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        if 'configs' not in kwargs:
            self.train_config = DEFAULT_TRAIN_PARAMS
            self.model_config = BASIC_MODEL_PARAMS
        else:
            self.train_config = kwargs['configs']['train_config']
            self.model_config = kwargs['configs']['model_config']
            del kwargs['configs']

        super().__init__(*args, **kwargs)
        self.tokenizer = PeptideTokenizer.from_massivekb(reverse=False)
        self.encoder = BaseModelEncoder(**self.model_config['encoder'])
        self.decoder = BaseModelDecoder(**self.model_config['decoder'])
        self.model = BaseModel(self.encoder, self.decoder, config=self.model_config)
        self.only_reconstruct = False
        kl_warmups = self.train_config['hyper_params']['kl_warmups']
        self.kl_annealing = lambda step: min(1.0, (step + 1) / float(kl_warmups)) if kl_warmups > 0 else 1.0
        self.accuracy = MulticlassAccuracy(num_classes=NUM_TOKENS)



    def forward(self, batch):
        spectrum = batch[0].float()
        precursors = batch[1]
        tokens_ids = batch[2]

        zeros = ~spectrum.sum(dim=2).bool()
        dummy_ = torch.zeros(spectrum.shape[0], 1, device=self.device, dtype=torch.bool)

        mask = [
            torch.tensor([[False]] * spectrum.shape[0]).type_as(zeros),
            zeros,
        ]

        mask = torch.cat(mask, dim=1).bool()
        mask = torch.cat((mask, dummy_), dim=-1)

        src_masks = ~mask
        src_masks = src_masks.float()
        trt_masks = (tokens_ids > 0).float()
        target_pad = torch.zeros((trt_masks.size(0), MAX_PEP_LEN), device=self.device).float()
        target_pad[:, :trt_masks.shape[1]] = trt_masks
        trt_masks = target_pad

        results = self.model(src_spectra=spectrum,
                             tgt_tokens=tokens_ids,
                             src_masks=src_masks,
                             tgt_masks=trt_masks,
                             precursors=precursors,
                             only_recon_loss=self.only_reconstruct)

        if not self.only_reconstruct:
            recon_err, KL, loss_length = results
            kl_weight = self.kl_annealing(self.global_step)

            recon_err = recon_err.sum()
            loss_length = loss_length.sum()
            KL = KL.sum()
            batch_size = spectrum.size(0)
            loss = (recon_err + loss_length + KL * kl_weight).div(batch_size)

            results = {
                'recon_loss': recon_err,
                'len_loss': loss_length,
                'kl': KL,
                'kl_w': KL * kl_weight,
                'total_loss': loss,
            }

        else:
            recon_err, loss_length = results
            recon_err = recon_err.sum()
            loss_length = loss_length.sum()
            batch_size = spectrum.size(0)
            loss = (recon_err + loss_length).div(batch_size)


            results = {
                'recon_loss': recon_err,
                'len_loss': loss_length,
                'total_loss': loss,
            }

        return results

    def step(self, batch, mode='train'):
        if batch[0] is None:
            return None

        results = self(batch=batch)

        loss = results['total_loss']
        self.log_metrics(mode, results)
        return loss


    def training_step(self, batch, batch_idx):
        self.encoder.train()
        self.decoder.train()
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            loss = self.step(batch, mode='val')
        return loss

    def log_metrics(self, stage_name, metrics):
        for metric_name, metric_value in metrics.items():
            self.log(f'{stage_name}_{metric_name}',
                     metric_value.detach().cpu().item(),
                     batch_size=len(metric_name),
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False)

    def configure_optimizers(self):
        scheduler_cfg = self.train_config['hyper_params']['scheduler']
        optimizer_cfg = self.train_config['hyper_params']
        optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=float(optimizer_cfg['lr']),
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0.01,
                                          amsgrad=False,
                                          maximize=False,
                                          foreach=None,
                                          capturable=False)


        if scheduler_cfg['type'] == 'plateau':
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(optimizer,
                                               monitor="val_total_loss",
                                               patience=5, factor=0.5,
                                               verbose=True),
                "monitor": "val_total_loss",
                "frequency": 1
            }

        elif scheduler_cfg['type'] == 'warmup':
            lr_scheduler = {"scheduler": WarmupScheduler(optimizer,
                                                         scheduler_cfg['warmup'],
                                                         scheduler_cfg['max_iters']),
                            "interval": "step"
            }

        else:
            raise NotImplemented(f"Invalid type of scheduler {scheduler_cfg['type']}")



        return [optimizer], [lr_scheduler]


def train_base_model(configs: dict):

    train_config = configs['train_config']
    dataset_params = train_config['datasets']
    environment_params = train_config['environment']
    hyper_params = train_config['hyper_params']

    torch.set_float32_matmul_precision(hyper_params['precision'])
    trainer_wrapper = BasicTrainingWrapper(configs=configs)
    train_dataset = create_dataset(dataset_path=dataset_params['train_dataset_path'],
                                   tokenizer=trainer_wrapper.tokenizer
                                   )

    val_dataset = create_dataset(dataset_path=dataset_params['val_dataset_path'],
                                 tokenizer=trainer_wrapper.tokenizer
                                 )

    checkpoint_folder = Path(environment_params['checkpoint_folder'])
    if not os.path.exists(checkpoint_folder):
        checkpoint_folder.mkdir(exist_ok=True)

    model_logger = CSVLogger(save_dir=checkpoint_folder,
                             name='pwa_training'
                             )

    early_stop_callback = EarlyStopping(monitor='val_total_loss',
                                        min_delta=hyper_params['min_delta'],
                                        patience=hyper_params['patience'],
                                        verbose=True,
                                        mode="min")

    pretrained_model_path = environment_params['pretrained_model']

    model_data = torch.load(pretrained_model_path)
    model_state = model_data['state_dict']
    trainer_wrapper.load_state_dict(model_state)

    ckpt = None

    if os.path.exists(pretrained_model_path):
        ckpt = pretrained_model_path

    ckpt = None

    trainer = pl.Trainer(default_root_dir=checkpoint_folder,
                         accelerator='auto',
                         logger=model_logger,
                         max_epochs=hyper_params['epochs'],
                         gradient_clip_val=0.5,
                         callbacks=[early_stop_callback],
                         )
    n_workers = os.cpu_count() / 2 if hyper_params['n_workers'] == 'auto' else hyper_params['n_workers']
    num_workers = int(n_workers)
    batch_size = int(hyper_params['batch_size'])
    trainer.fit(
        model=trainer_wrapper,
        train_dataloaders=train_dataset.loader(
            batch_size=batch_size,
            num_workers=num_workers),
        val_dataloaders=val_dataset.loader(
            batch_size=batch_size,
            num_workers=num_workers),
        ckpt_path=ckpt
    )


def create_dataset(dataset_path: str, tokenizer: PeptideTokenizer) -> AnnotatedSpectrumDataset:
    index_folder = Path(dataset_path) / 'index'
    index_folder.mkdir(exist_ok=True)
    dataset_files = glob.glob(f'{dataset_path}/*.mgf')
    index_path = index_folder / f'{Path(dataset_path).stem}.hdf5'
    dataset = AnnotatedSpectrumDataset(tokenizer=tokenizer,
                                       ms_data_files=dataset_files,
                                       overwrite=True,
                                       add_stop_token=True,
                                       index_path=index_path,
                                       )
    return dataset


def parse_nested_args(args, param_dict):

    for arg_key, arg_value in args.items():
        keys = arg_key.split('.')  #
        target = param_dict
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        try:

            if isinstance(target[keys[-1]], int):
                target[keys[-1]] = int(arg_value)
            elif isinstance(target[keys[-1]], float):
                target[keys[-1]] = float(arg_value)
            else:
                target[keys[-1]] = arg_value
        except KeyError:
            target[keys[-1]] = arg_value




def run_trainer_():
    cfg = setup_train_environment()
    parser = argparse.ArgumentParser(description="Update default train parameters via command-line arguments")
    parser.add_argument(
        '--update',
        nargs='*',
        help='Update dictionary values using dot notation, e.g., '
             '--update environment.device=cpu datasets.train_dataset_path=/path/to/train_folder/',
        default=[]
    )
    args = parser.parse_args()
    updates = args.update
    if updates:
        updates_dict = {}
        for update in updates:
            try:
                key, value = update.split('=', maxsplit=1)
                updates_dict[key.strip()] = value.strip()
            except ValueError:
                print(f"Invalid update format: {update}. Use key.subkey=value.", file=sys.stderr)
                sys.exit(1)

        parse_nested_args(updates_dict, cfg['train_config'])

    print(json.dumps(cfg, indent=4))
    train_base_model(configs=cfg)


if __name__ == "__main__":
    """ 
    Update dictionary values of DEFAULT_TRAIN_PARAMS using dot notation, e.g 
    Example: trainer.py --update datasets.train_dataset_path=/path/to/train_folder/ 
    datasets.val_dataset_path=/path/to/val_folder/
    """
    run_trainer_()