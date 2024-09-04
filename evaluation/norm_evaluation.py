from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule


def norm_evaluation(
        models: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer):
    datamodule.setup()
    # trainer.test(models, datamodule.train_dataloader())
    trainer.test(models, datamodule.val_dataloader())
    trainer.test(models, datamodule.test_dataloader())
