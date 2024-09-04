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
from torchvision import transforms


class VIDEO_dataset(Dataset):
    def __init__(self, data_dir, stage, transform):
        self.target_label = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
        self.target_conver_label = ['02', '03', '04', '05', '06', '07', '08']

        self.transform = transform
        self.audio_transform = transforms.Compose([
            # transforms.CenterCrop(148),
            transforms.ToTensor(),
            transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR)])

        with open(data_dir, 'r') as file:
            files = yaml.load(file, Loader=yaml.FullLoader)

        self.face_sets = files[stage]
        self.stage = stage
        self.data_dir = data_dir
        self.dataset_length = len(self.face_sets)
        self.outdir = '/'.join(data_dir.split('/')[:-1])

    def __getitem__(self, i):

        cropped_img = imageio.imread(self.face_sets[i])

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
        ID, _, label, level, image_name = self.face_sets[i].split('/')[6:]

        audio_name = image_name.split('.')[0][-3:] + '.npy'

        # audio_p=audio_files[0]
        audio_p = os.path.join(self.outdir, ID, 'spectrograms', label, level, audio_name)

        new_sig = np.load(audio_p)
        new_sig = self.audio_transform(new_sig)

        label = self.target_label.index(label)

        return {'img': img_data, 'labels': label, 'audio': new_sig}

    def __len__(self):
        return self.dataset_length