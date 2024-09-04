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
from pathlib import Path


def interpolation_images(
        models: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer,
        imgs
):
    datamodule.setup()

    models = models.eval()
    # models.classifer.eval()

    trains_fo = torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                                     n_fft=1310,
                                                     n_mels=128)
    toDB = torchaudio.transforms.AmplitudeToDB(top_db=100)
    m = 0
    audio_a = []

    audio_h =[]

    audio_s=[]

    audio_c=[]

    audio_f=[]

    audio_su=[]

    andio_d=[]

    imgdir=[]
    #img_path = os.listdir(imgdir)
    for j,audios in enumerate([audio_c,audio_s, audio_h,audio_a, audio_f,andio_d, audio_su]):
        output_img = torch.zeros(1, 3, 128, 128)
        for i, img_name in enumerate(imgdir):
            input_data = {}
            #cropped_img = imageio.imread(os.path.join(imgdir, img_name))
            cropped_img = imageio.imread(img_name)
            if len(cropped_img.shape) == 2:
                cropped_img = cropped_img[:, :, np.newaxis]
                cropped_img = np.concatenate(
                    [cropped_img, cropped_img, cropped_img], axis=2)
            elif len(cropped_img.shape) != 3:
                raise Exception('Unsupported shape of image.')

            face_img = Image.fromarray(cropped_img.astype(np.uint8))
            img = datamodule.test_transform(face_img)
            face_img1 = img.unsqueeze(0)
            input_data['img'] = face_img1
            audio=audios[i]
            sig, sr = torchaudio.load(audio)
            new_sig = trains_fo(sig)
            new_sig = toDB(new_sig)
            sig_max = new_sig.max()
            sig_min = new_sig.min()
            new_sig = (new_sig - sig_min) / (sig_max - sig_min)
            new_sig = new_sig.unsqueeze(0)
            input_data['audio'] = new_sig
            input_data['labels'] = torch.Tensor([6]).long()
            for t in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                output_data = models.test_step(input_data, m,t)
                c_recons = output_data['c_recons']
                output_img = torch.cat((output_img, c_recons), dim=0)
                m += 1

        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                vutils.save_image(output_img[1:, :, :, :],
                                  os.path.join(logger.log_dir,
                                               "Reconstructions",
                                               f"new_offset_neutral_{str(j)}.png"),
                                  normalize=False,
                                  nrow=11,
                                  scale_each=True
                                  )
