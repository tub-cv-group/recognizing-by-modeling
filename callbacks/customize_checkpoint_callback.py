import os

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from evaluation import norm_evaluation


class ModelCheckpointWithArtifactLogging(ModelCheckpoint):

    def __init__(self, save_top_k, **kwargs):
        super().__init__(**kwargs)

        self.save_top_k = save_top_k

    def on_train_end(self,
                     trainer: pl.Trainer,
                     pl_module: pl.LightningModule) -> None:
        return_dict = super().on_train_end(trainer, pl_module)
        # We need to manually log the checkpoint here again since it might be that by
        # setting log_model_every_n_epochs we skipped the best one

        temp_path = '/'.join(self.best_model_path.split('/')[:-1]) + '/best.ckpt'
        os.rename(self.best_model_path, temp_path)
        self.best_model_path = temp_path
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                folder_name = logger.log_dir
        for logger in trainer.loggers:
            if isinstance(logger, CometLogger):
                logger.experiment.log_asset_folder(os.path.join(folder_name, 'checkpoints'), log_file_name=True)
        print(self.best_model_path)
        ckpt_path = self.best_model_path
        if ckpt_path:
            print('=========loading model==========')
            ccc = torch.load(ckpt_path)
            pl_module.load_state_dict(ccc['state_dict'])

        norm_evaluation(pl_module, trainer.datamodule, trainer)
        return return_dict
