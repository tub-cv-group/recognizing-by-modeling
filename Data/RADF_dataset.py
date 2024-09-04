import os
import glob
import yaml
from typing import List, Optional, Sequence, Union, Any, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
import imageio
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF


class RADF_dataset(Dataset):
    def __init__(self, data_dir, stage, transform):
        self.target_label = ['happy', 'sad', 'contemptuous', 'angry', 'neutral', 'disgusted',
                             'surprised', 'fearful']
        self.transform = transform
        with open(data_dir, 'r') as file:
            files = yaml.load(file, Loader=yaml.FullLoader)

        self.face_sets = files[stage]

        self.data_dir = data_dir
        self.dataset_length = len(self.face_sets)
        self.outdir = '/'.join(data_dir.split('/')[:-1])

    def __getitem__(self, i):
        cropped_img = imageio.imread(os.path.join(self.outdir, self.face_sets[i]))
        label = self.target_label.index(self.face_sets[i].split('/')[0])
        if len(cropped_img.shape) == 2:
            cropped_img = cropped_img[:, :, np.newaxis]
            cropped_img = np.concatenate(
                [cropped_img, cropped_img, cropped_img], axis=2)
        elif len(cropped_img.shape) != 3:
            raise Exception('Unsupported shape of image.')
        # imgsize = cropped_img.shape[:2]

        face_img = Image.fromarray(cropped_img.astype(np.uint8))
        # self.aspect_ratio_transform.aspect_ratio = imgsize[1] / float(
        #     imgsize[0])
        # face_img = self.aspect_ratio_transform(face_img)

        if self.transform is not None:
            face_img = self.transform(face_img)
        return {'img': face_img, 'labels': label}

    def __len__(self):
        return self.dataset_length
