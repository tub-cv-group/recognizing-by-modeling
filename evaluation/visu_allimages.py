from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule


def visu_allimages(
        models: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer):
    trainer.limit_test_batches = 40
    datamodule.setup()
    trainer.test(models,datamodule.train_dataloader())
    trainer.test(models, datamodule.test_dataloader())
    trainer.test(models, datamodule.val_dataloader())
