import random
import os
import glob
from typing import Optional

import torch
import numpy as np
import imageio
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.utils as vutils
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule
import torchmetrics
import subprocess

class fusion_maxtrix_visual(LightningModule):
    def __init__(self,
                 model,
                 data_set
                 ):
        super().__init__()
        self.model = model
        self.data_set = data_set
        self.test_acc = torchmetrics.Accuracy(num_classes=self.model.params['num_classes'],task='multiclass')

    def on_test_start(self) -> None:
        self.maxtrx = np.zeros((self.model.params['num_classes'], self.model.params['num_classes']))

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        outputs = self.model.test_step(batch, batch_idx)
        pred = outputs['class']
        labels = batch['labels']
        self.test_acc.update(pred, labels)
        for i in range(len(pred)):
            self.maxtrx[pred[i], labels[i]] += 1
        # acc=self.test_acc.compute()
        # self.log_dict({f'{self.data_set}_acc':acc})

    def test_epoch_end(self, outputs):
        self.log_dict({f'{self.data_set}_acc': self.test_acc.compute()})
        self.test_acc.reset()

    def on_test_end(self) -> None:
        self.maxtrx = self.maxtrx / self.maxtrx.sum(axis=0)
        acc=0
        for i in range(self.maxtrx.shape[0]):
            acc+=self.maxtrx[i][i]
        print(acc/7)


def fusion_matrix_evaluation(
        models: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer):
    datamodule.setup()
    models = models.eval()
    temp = {'test_dataset': datamodule.test_dataloader}
    for data_set, data_loader in temp.items():
        new_model = fusion_maxtrix_visual(models, data_set)
        trainer.test(model=new_model, dataloaders=data_loader())
        tables_labels = datamodule.val_dataset.target_label
        df_cm = pd.DataFrame(
            new_model.maxtrx,
            range(len(tables_labels)),
            range(len(tables_labels)))
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1.0)  # for label size
        ax = sns.heatmap(df_cm,
                        annot=True,
                        cmap='Blues',
                        annot_kws={"size": 8},
                        xticklabels=tables_labels,
                        yticklabels=tables_labels, )
        #plt.title(f'{data_set}')
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                eval_file = os.path.join(logger.log_dir, 'eval')
                if not os.path.exists(eval_file):
                    subprocess.run(f'mkdir -p {eval_file}', shell=True)
                plt.savefig(os.path.join(eval_file, f'{data_set} fusion_matrix.png'))
            if isinstance(logger, CometLogger):
                logger.experiment.log_image(os.path.join(eval_file, f'{data_set} fusion_matrix.pdf'))
                df_cm.to_csv(os.path.join(eval_file, f'{data_set} fusion_matrix.csv'))
