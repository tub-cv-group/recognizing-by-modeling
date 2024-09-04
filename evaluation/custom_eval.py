import os
from typing import Optional

import torch
import torchvision.utils as vutils
import numpy as np
import imageio
from PIL import Image
import torchaudio
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities.cli import SaveConfigCallback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule


def custom_eval(
        models: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer,
        conf
):
    datamodule.setup()
    #
    models = models.eval()
    # models.classifer.eval()

    trains_fo = torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                                     n_fft=1310,
                                                     n_mels=128)
    toDB = torchaudio.transforms.AmplitudeToDB(top_db=100)
    output_img = torch.zeros(1, 3, 128, 128)
    img_name = conf['img']
    input_data = {}
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
    
    audios = conf['audio']
    m = 0
    for audio in audios:
        sig, sr = torchaudio.load(audio)
        new_sig = trains_fo(sig)
        new_sig = toDB(new_sig)
        sig_max = new_sig.max()
        sig_min = new_sig.min()
        new_sig = (new_sig - sig_min) / (sig_max - sig_min)
        new_sig = new_sig.unsqueeze(0)
        input_data['audio'] = new_sig
        input_data['labels'] = torch.Tensor([6]).long()
        for t in [0.5,1.0]:
            output_data = models.test_step(input_data, m, t)
            if t == 0.5:
                recons = output_data['recons']
                output_img = torch.cat((output_img, recons), dim=0)
            c_recons = output_data['c_recons']
            output_img = torch.cat((output_img, c_recons), dim=0)
            m += 1
    output_img=output_img[1:, :, :, :].reshape((-1,3,3, 128, 128))
    output_img=torch.transpose(output_img,0,1)
    output_img=output_img.reshape(-1,3,128,128)

    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            vutils.save_image(output_img,
                              os.path.join(logger.log_dir,
                                           "Reconstructions",
                                           f"recons.png"),
                              normalize=False,
                              nrow=6,
                              scale_each=True
                              )
