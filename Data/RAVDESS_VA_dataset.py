import os
import glob
import yaml
import random
from typing import List, Optional, Sequence, Union, Any, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
import imageio
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torchaudio


class RAVDESS_VA_dataset(Dataset):
    def __init__(self, data_dir, stage, patch_size, transform, feature_setting):
        self.target_label = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.target_conver_label = ['02', '03', '04', '05','06','07','08']

        t_table = {'Wav': None, 'Spectrum': torchaudio.transforms.Spectrogram(n_fft=feature_setting['n_fft']),
                   'Mel_frequence': torchaudio.transforms.MelSpectrogram(sample_rate=feature_setting['sr'],
                                                                         n_fft=feature_setting['n_fft'],
                                                                         n_mels=feature_setting['n_mels']),
                   'MFCC': torchaudio.transforms.MFCC(sample_rate=feature_setting['sr'],
                                                      n_mfcc=feature_setting['n_mfcc'],
                                                      melkwargs={'n_fft': feature_setting['n_fft'],
                                                                 'n_mels': feature_setting['n_mels']})}
        self.sr = feature_setting['sr']
        self.audio_transform = t_table[feature_setting['type']]
        self.to_DB = feature_setting['to_DB']
        self.trans_to_DB = torchaudio.transforms.AmplitudeToDB(top_db=100)

        self.transform = transform
        with open(data_dir, 'r') as file:
            files = yaml.load(file, Loader=yaml.FullLoader)

        self.face_sets = files[stage]
        self.stage = stage
        self.data_dir = data_dir
        self.dataset_length = len(self.face_sets)
        self.outdir = '/'.join(data_dir.split('/')[:-1])
        self.patch_size = patch_size
    def __getitem__(self, i):
        imgs_path = os.path.join(self.outdir, self.face_sets[i])

        cropped_img = imageio.imread(imgs_path)

        if len(cropped_img.shape) == 2:
            cropped_img = cropped_img[:, :, np.newaxis]
            cropped_img = np.concatenate(
                [cropped_img, cropped_img, cropped_img], axis=2)
        elif len(cropped_img.shape) != 3:
            raise Exception('Unsupported shape of image.')

        face_img = Image.fromarray(cropped_img.astype(np.uint8))

        if self.transform is not None:
            img_data = self.transform(face_img)
        else:
            img_data = face_img
        # if self.stage=='train':
        #     emotion= self.face_sets[i].split('/')[1]
        #     audios=self.audio_files[emotion]
        #     audio_p = random.choice(audios)
        #     sig, sr = torchaudio.load(audio_p)
        # else:
        audio_files = glob.glob(
                os.path.join(self.outdir, self.face_sets[i].split('/')[0], self.face_sets[i].split('/')[1],
                             '**',f"{self.face_sets[i].split('/')[3]}.wav"), recursive=True)
        audio_p = random.choice(audio_files)
        #audio_p=audio_files[0]
        sig, sr = torchaudio.load(os.path.join(self.outdir, audio_p))


        if sr != self.sr or sig.shape[0] != 1:
            assert NotImplementedError
        if self.audio_transform:
            new_sig = self.audio_transform(sig)
        else:
            new_sig = sig
        if self.to_DB:
            new_sig = self.trans_to_DB(new_sig)
        sig_max = new_sig.max()
        sig_min = new_sig.min()
        # new_sig = (new_sig - sig_min) / (sig_max - sig_min+0.0001)
        # new_sig=new_sig-0.5
        new_sig = (new_sig - sig_min) / (sig_max - sig_min)
        new_sig = new_sig

        if self.face_sets[i].split('/')[1] == '01':
            label = len(self.target_conver_label)
            raise NotImplementedError
        else:
            emotion = self.face_sets[i].split('/')[1]
            sur = self.face_sets[i].split('/')[3].split('-')[2]
            if emotion != sur:
                raise NotImplementedError
            label = self.target_conver_label.index(self.face_sets[i].split('/')[1])

        return {'img': img_data, 'labels': label, 'audio': new_sig}

    def __len__(self):
        return self.dataset_length

