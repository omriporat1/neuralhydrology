from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import sys
import torch
import logging
from neuralhydrology.training.basetrainer import BaseTrainer

LOGGER = logging.getLogger(__name__)

from myutils.configs import add_run_config

class MyBaseTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _create_folder_structure(self):
        # create as subdirectory within run directory of base run
        if self.cfg.is_continue_training:
            folder_name = f"continue_training_from_epoch{self._epoch:03d}"

            # store dir of base run for easier access in weight loading
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.run_dir = self.cfg.run_dir / folder_name

        # create as new folder structure
        else:
            now = datetime.now()
            day = f"{now.day}".zfill(2)
            month = f"{now.month}".zfill(2)
            year = f"{now.year}".zfill(4)
            hour = f"{now.hour}".zfill(2)
            minute = f"{now.minute}".zfill(2)
            second = f"{now.second}".zfill(2)
            run_name = f'{day}{month}{year}_{hour}{minute}{second}'

            # if no directory for the runs is specified, a 'runs' folder will be created in the current working dir
            if self.cfg.run_dir is None:
                self.cfg.run_dir = Path().cwd() / "runs" / self.cfg.experiment_name / run_name
            else:
                self.cfg.run_dir = self.cfg.run_dir / run_name

        # create folder + necessary subfolder
        if not self.cfg.run_dir.is_dir():
            self.cfg.train_dir = self.cfg.run_dir / "train_data"
            self.cfg.train_dir.mkdir(parents=True)
        else:
            raise RuntimeError(f"There is already a folder at {self.cfg.run_dir}")
        if self.cfg.log_n_figures is not None:
            self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
            self.cfg.img_log_dir.mkdir(parents=True)
        
        add_run_config(self.cfg)

    def _epoch_0(self):
        self.model.train()
        self.experiment_logger.train()

        # process bar handle
        n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch is not None else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f'# Epoch {0}')

        # Iterate in batches over training set
        nan_count = 0
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            for key in data.keys():
                if not key.startswith('date'):
                    data[key] = data[key].to(self.device)

            # apply possible pre-processing to the batch before the forward pass
            data = self.model.pre_model_hook(data, is_train=True)

            # get predictions
            predictions = self.model(data)

            if self.noise_sampler_y is not None:
                for key in filter(lambda k: 'y' in k, data.keys()):
                    noise = self.noise_sampler_y.sample(data[key].shape)
                    # make sure we add near-zero noise to originally near-zero targets
                    data[key] += (data[key] + self._target_mean / self._target_std) * noise.to(self.device)

            loss, all_losses = self.loss_obj(predictions, data)

            # early stop training if loss is NaN
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError(f"Loss was NaN for {nan_count} times in a row. Stopped training.")
                LOGGER.warning(f"Loss is Nan; ignoring step. (#{nan_count}/{self._allow_subsequent_nan_losses})")

            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

            self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})