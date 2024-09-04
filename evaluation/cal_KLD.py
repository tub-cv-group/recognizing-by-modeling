from typing import Optional
import os

import torch
import numpy as np
import imageio
from PIL import Image
import PIL
import torchaudio
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule
import torchvision.utils as vutils

from dataset import VAEDataset


def cal_KL(models: LightningModule,
           datamodule: LightningDataModule,
           trainer: Trainer,
           conf):
    datamodule.setup()

    models = models.eval()
    # models.classifer.eval()

    trains_fo = torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                                     n_fft=1310,
                                                     n_mels=128)
    toDB = torchaudio.transforms.AmplitudeToDB(top_db=100)
    output_img=torch.zeros(1,3,128,128)
    img_path = os.listdir(conf['img'])
    m=0
    for i, img_name in enumerate(img_path):
        if i==7:
            print(os.path.join(conf['img'], img_name))
        input_data = {}
        cropped_img = imageio.imread(os.path.join(conf['img'], img_name))
        if len(cropped_img.shape) == 2:
            cropped_img = cropped_img[:, :, np.newaxis]
            cropped_img = np.concatenate(
                [cropped_img, cropped_img, cropped_img], axis=2)
        elif len(cropped_img.shape) != 3:
            raise Exception('Unsupported shape of image.')

        face_img = Image.fromarray(cropped_img.astype(np.uint8))
        img=datamodule.test_transform(face_img)
        face_img1 = img.unsqueeze(0)
        input_data['img'] = face_img1
        audio_1 = conf['audio']
        if isinstance(audio_1, list):
            for j, audios in enumerate(audio_1):
                sig, sr = torchaudio.load(audios)
                new_sig = trains_fo(sig)
                new_sig = toDB(new_sig)
                sig_max = new_sig.max()
                sig_min = new_sig.min()
                new_sig = (new_sig - sig_min) / (sig_max - sig_min)
                new_sig = new_sig.unsqueeze(0)
                input_data['audio'] = new_sig
                input_data['labels']=torch.Tensor([6]).long()
                output_data = models.test_step(input_data, m)
                print(output_data['class'])
                if j == 0:
                    recons = output_data['recons']
                    output_img=torch.cat((output_img,output_data['img']),dim=0)
                    output_img = torch.cat((output_img, recons), dim=0)

                c_recons = output_data['c_recons']
                output_img = torch.cat((output_img, c_recons), dim=0)
                m+=1
    print(output_img.shape)

    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            vutils.save_image(output_img[1:,:,:,:],
                              os.path.join(logger.log_dir,
                                           "Reconstructions",
                                           f"recons.png"),
                              normalize=False,
                              nrow=9,
                              scale_each=True
                              )