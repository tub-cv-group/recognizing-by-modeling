import os

import torch
from torch import Tensor
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import create_transforms
from Data import RADF_dataset, RAVDESS_VA_dataset, CELEB_dataset, VIDEO_dataset


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            test_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            mean=None,
            std=None,
            crop=None,
            data_file=None,
            audio_file=None,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.std = std
        self.mean = mean
        self.crop = crop
        self.data_file = data_file

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    # #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


class RADFDataset(VAEDataset):
    def __init__(self, **kwargs, ):
        super().__init__(**kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        available_dataset = {
            'sortedRaFD': RADF_dataset,
            'RaFD_cropped': RADF_dataset,
        }
        data_set = self.data_dir.split('/')[-1]
        self.train_transform = create_transforms(self.patch_size, True, self.mean, self.std, self.crop)
        self.test_transform = create_transforms(self.patch_size, False, self.mean, self.std, self.crop)

        if data_set in available_dataset:
            self.train_dataset = available_dataset[data_set](
                os.path.join(self.data_dir, self.data_file), 'train',
                transform=self.train_transform,
            )
            # length = int(len(train_dataset) * 0.9)
            # splits = [length, len(train_dataset) - length]
            # self.train_dataset,self.val_dataset=torch.utils.data.random_split(train_dataset,splits)
            self.val_dataset = available_dataset[data_set](
                os.path.join(self.data_dir, self.data_file), 'test',
                transform=self.test_transform, )

            self.test_dataset = available_dataset[data_set](
                os.path.join(self.data_dir, self.data_file), 'test',
                transform=self.test_transform,
            )
            print(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))
        else:
            raise NotImplementedError("Please give an availabile Dataset")


class RADVIDEODataset(VAEDataset):
    def __init__(self, **kwargs, ):
        super().__init__(**kwargs)
        self.features = kwargs['feature']

    def setup(self, stage: Optional[str] = None) -> None:
        available_dataset = {
            'RAVDESS_CROPP_VA': RAVDESS_VA_dataset,
        }
        data_set = self.data_dir.split('/')[-1]
        self.train_transform = create_transforms(self.patch_size, True, self.mean, self.std, self.crop)
        self.validation_transform = create_transforms(self.patch_size, False, self.mean, self.std, self.crop)
        self.test_transform = create_transforms(self.patch_size, False, self.mean, self.std, self.crop)
        if data_set in available_dataset:
            self.train_dataset = RAVDESS_VA_dataset(
                os.path.join(self.data_dir, self.data_file), 'train', self.patch_size,
                transform=self.train_transform, feature_setting=self.features
            )

            self.val_dataset = RAVDESS_VA_dataset(
                os.path.join(self.data_dir, self.data_file), 'val', self.patch_size,
                transform=self.validation_transform, feature_setting=self.features
            )

            self.test_dataset = RAVDESS_VA_dataset(
                os.path.join(self.data_dir, self.data_file), 'test', self.patch_size,
                transform=self.test_transform, feature_setting=self.features
            )
            print(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))
        else:
            raise NotImplementedError("Please give an availabile Dataset")

class CELEBADataset(VAEDataset):
    def __init__(self, **kwargs, ):
        super().__init__(**kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        available_dataset = {
            'cropped_celeba': CELEB_dataset,
            'RAVDESS_cropp': CELEB_dataset,
            'mead': VIDEO_dataset,
        }
        data_set = self.data_dir.split('/')[-1]
        self.train_transform = create_transforms(self.patch_size, True, self.mean, self.std, self.crop)
        self.test_transform = create_transforms(self.patch_size, False, self.mean, self.std, self.crop)

        if data_set in available_dataset:
            self.train_dataset = available_dataset[data_set](
                os.path.join(self.data_dir, self.data_file), 'train',
                transform=self.train_transform,
            )
    
            self.val_dataset = available_dataset[data_set](
                os.path.join(self.data_dir, self.data_file), 'val',
                transform=self.test_transform, )

            self.test_dataset = available_dataset[data_set](
                os.path.join(self.data_dir, self.data_file), 'val',
                transform=self.test_transform,
            )
            print(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))
        else:
            raise NotImplementedError("Please give an availabile Dataset")

