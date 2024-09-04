import random
import os
import glob
from typing import Optional

import torch
import numpy as np
import imageio
from PIL import Image
import torchaudio
import tsnecuda
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.utils as vutils
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule

Target_Label = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
Target_Conver_Label=['02', '03', '04', '05', '06', '07', '08']

def tsne_eval(models: LightningModule,
            datamodule: LightningDataModule,
            trainer: Trainer,
            conf):
        datamodule.setup()
        device=torch.device("cuda:1")
        models=models.eval().to(device)

        trains_fo=torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                                              n_fft=1310,
                                                              n_mels=128)
        toDB=torchaudio.transforms.AmplitudeToDB(top_db=100)
        tsne_1=tsnecuda.TSNE(n_components=2)
        tsne_2=tsnecuda.TSNE(n_components=2)
        output_o_feature=torch.zeros((1,512))
        output_l_feature = torch.zeros((1, 512))
        output_id=[]
        output_label=[]
        id_paths = conf['id']
        n=0
        for id_path in id_paths:
            emotions=os.listdir(id_path)
            for emotion in emotions:
                if emotion=='01':
                    continue
                emotion_path=os.path.join(id_path,emotion)
                img_files = glob.glob(
                    os.path.join(emotion_path,'test', '**', f'*.jpg'), recursive=True)
                for img_file in img_files:
                    input_data = {}
                    cropped_img1 = imageio.imread(img_file)
                    if len(cropped_img1.shape) == 2:
                        cropped_img1 = cropped_img1[:, :, np.newaxis]
                        cropped_img1 = np.concatenate(
                            [cropped_img1, cropped_img1, cropped_img1], axis=2)
                    elif len(cropped_img1.shape) != 3:
                        raise Exception('Unsupported shape of image')
                    face_img1 = Image.fromarray(cropped_img1.astype(np.uint8))
                    face_img1 = datamodule.test_transform(face_img1)
                    face_img1 = face_img1.unsqueeze(0)
                    input_data['img']=face_img1.to(device)
                    input_data['labels']=torch.Tensor([6]).long().to(device)
                    audio_files=glob.glob(
                    os.path.join(emotion_path,'test','**', f'*.wav'), recursive=True)
                    audio_file=random.choice(audio_files)
                    sig, sr = torchaudio.load(audio_file)
                    new_sig = trains_fo(sig)
                    new_sig = toDB(new_sig)
                    sig_max = new_sig.max()
                    sig_min = new_sig.min()
                    # new_sig = (new_sig - sig_min) / (sig_max - sig_min+0.0001)
                    # new_sig=new_sig-0.5
                    new_sig = (new_sig - sig_min) / (sig_max - sig_min)
                    new_sig = new_sig.unsqueeze(0)
                    input_data['audio'] = new_sig.to(device)

                    output_data = models.test_step(input_data, n)
                    n+=1
                    output_o_feature=torch.vstack((output_o_feature, output_data['ori_feature'].cpu()))
                    output_l_feature = torch.vstack((output_l_feature, output_data['later_feature'].cpu()))
                    output_id.append(id_path.split('/')[-1])
                    #output_label.append(emotion)
                    e_idx=Target_Conver_Label.index(emotion)
                    output_label.append(Target_Label[e_idx])

        tsne_o_feature = tsne_1.fit_transform(output_o_feature[1:,:])
        tsne_l_feature = tsne_2.fit_transform(output_l_feature[1:,:])

        df_o_data = pd.DataFrame(tsne_o_feature, columns=['Dim1', 'Dim2'])
        df_o_data.loc[:, 'class'] = output_label
        df_o_data.loc[:, 'id'] = output_id

        df_l_data = pd.DataFrame(tsne_l_feature, columns=['Dim1', 'Dim2'])
        df_l_data.loc[:,'class']=output_label
        df_l_data.loc[:, 'id'] = output_id
        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')
        sns.set(font_scale=1)
        sns.scatterplot(data=df_o_data, hue='class', x='Dim1', y='Dim2',style='id',s=100)
        plt.legend()
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                plt.savefig(os.path.join(logger.log_dir,'original_features.pdf'))
                plt.savefig(os.path.join(logger.log_dir, 'original_features.png'))
                df_o_data.to_csv(os.path.join(logger.log_dir,'original_features.csv'))
        fig = plt.figure(figsize=(8, 8))
        plt.axis('off')
        sns.scatterplot(data=df_l_data, hue='class', x='Dim1', y='Dim2',style='id',s=100)
        plt.legend([], [], frameon=False)
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                plt.savefig(os.path.join(logger.log_dir, 'shifted_features.pdf'))
                plt.savefig(os.path.join(logger.log_dir, 'shifted_features.png'))
                df_l_data.to_csv(os.path.join(logger.log_dir, 'shifted_features.csv'))
