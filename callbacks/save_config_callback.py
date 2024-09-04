import os
from typing import Optional

import torch
import numpy as np
import torchvision.utils as vutils
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from evaluation import norm_evaluation

class SaveAndLogConfigCallback(Callback):
    """A simple callback to log the config using the loggers of the trainer.
    """

    def setup(self, trainer: Trainer,
              pl_module: LightningModule,
              stage: Optional[str] = None, ) -> None:
        super().setup(trainer, pl_module, stage)
        for logger in trainer.loggers:
            if isinstance(logger, CometLogger):
                logger.experiment.log_code(folder='callbacks')
                logger.experiment.log_code(folder='models')
                logger.experiment.log_code(folder='Data')
                logger.experiment.log_code(folder='evaluation')
                logger.experiment.log_code(file_name='dataset.py')
                logger.experiment.log_code(folder='experiments')
                logger.experiment.log_code(file_name='run.py')
                logger.experiment.log_code(file_name='utils.py')

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        op = hasattr(pl_module, 'sample_images')
        if op:
                test_batch = next(iter(trainer.datamodule.test_dataloader()))
                output, samples = pl_module.sample_images(test_batch)
                for logger in trainer.loggers:
                    if isinstance(logger, TensorBoardLogger):
                        vutils.save_image(output,
                                          os.path.join(logger.log_dir,
                                                       "Reconstructions",
                                                       f"recons_{logger.name}_Epoch_{trainer.current_epoch}.png"),
                                          normalize=False,
                                          nrow=16,
                                          scale_each=True
                                          )
                        vutils.save_image(samples,
                                          os.path.join(logger.log_dir,
                                                       "Samples",
                                                       f"recons_{logger.name}_Epoch_{trainer.current_epoch}.png"),
                                          normalize=False,
                                          nrow=16,
                                          scale_each=True
                                          )
        ap = hasattr(pl_module, 'log_images')
        if ap:
            test_batch = next(iter(trainer.datamodule.test_dataloader()))
            log_img, samples = pl_module.log_images(test_batch)
            for logger in trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    vutils.save_image(log_img,
                                      os.path.join(logger.log_dir,
                                                   "Reconstructions",
                                                   f"recons_Epoch_{trainer.current_epoch}.png"),
                                      normalize=False,
                                      nrow=12,
                                      scale_each=True
                                      )