import random
import os
import glob
from typing import Optional

import torch
import numpy as np
import imageio
from PIL import Image
import torchaudio
import torchvision.utils as vutils
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule


def one_call(models: LightningModule,
             datamodule: LightningDataModule,
             trainer: Trainer,
             image,
             audio):
    datamodule.setup()

    models = models.eval()
    # models.classifer.eval()

    trains_fo = torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                                     n_fft=1310,
                                                     n_mels=128)
    toDB = torchaudio.transforms.AmplitudeToDB(top_db=100)
    output_img = torch.zeros(1, 3, 128, 128)
    input_data = {}
    cropped_img1 = imageio.imread(image)
    if len(cropped_img1.shape) == 2:
        cropped_img1 = cropped_img1[:, :, np.newaxis]
        cropped_img1 = np.concatenate(
            [cropped_img1, cropped_img1, cropped_img1], axis=2)
    elif len(cropped_img1.shape) != 3:
        raise Exception('Unsupported shape of image')
    face_img1 = Image.fromarray(cropped_img1.astype(np.uint8))
    face_img1 = datamodule.test_transform(face_img1)
    face_img1 = face_img1.unsqueeze(0)
    input_data['img'] = face_img1
    input_data['labels'] = torch.Tensor([6]).long()
    sig, sr = torchaudio.load(audio)
    new_sig = trains_fo(sig)
    new_sig = toDB(new_sig)
    sig_max = new_sig.max()
    sig_min = new_sig.min()
    new_sig = (new_sig - sig_min) / (sig_max - sig_min)
    new_sig = new_sig.unsqueeze(0)
    input_data['audio'] = new_sig

    m = 0
    output_data = models.test_step(input_data, m)
    c_recons = output_data['c_recons']
    recons=output_data['recons']
    output_img = torch.cat((output_img, recons), dim=0)

    output_img = torch.cat((output_img, c_recons), dim=0)
    output_img = output_img[1:]
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            vutils.save_image(output_img,
                              os.path.join(logger.log_dir,
                                            f'output.png',
                                           ),
                              normalize=False,
                              nrow=2,
                              scale_each=True
                              )
