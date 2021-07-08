import os
import pickle

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import torchvision.transforms.functional as TVF

from .fast_dataloader import FastDataLoader


class SampleImagesDataset(VisionDataset):
    def __init__(self, *args, num_samples=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_filenames = list_files_recursive(self.root, IMG_EXTENSIONS)
        self._image_cache = {}
        self.num_samples = num_samples

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img = pil_loader(filename)
        img_global = self.transform(img)
        # take samples
        w, h = img.size
        min_scale, max_scale = 0.2, 1.
        scales = torch.rand(self.num_samples) * (max_scale - min_scale) + min_scale
        # tops = torch.rand(self.num_samples) * scales
        # lefts = torch.rand(self.num_samples) * scales
        tops = torch.rand(self.num_samples)
        lefts = torch.rand(self.num_samples)
        img_samples = []
        for top, left, scale in zip(tops, lefts, scales):
            width = int(w * scale)
            height = int(h * scale)
            top = int((h - height) * top)
            left = int((w - width) * left)
            img_sample = TVF.crop(img, top, left, height, width)
            img_sample = self.transform(img_sample)
            img_samples.append(img_sample)
        img_samples = torch.stack(img_samples)
        return img_global, img_samples, tops, lefts, scales

    def __len__(self):
        return len(self.image_filenames)


def list_files_recursive(root, extensions):
    all_files = []
    for (path, dirs, files) in os.walk(root):
        with_correct_extensions = filter(
            lambda n: os.path.splitext(n)[1] in extensions, files
        )
        all_files.extend([
            os.path.join(path, name) for name in with_correct_extensions
        ])
    return all_files


class SampleImagesDataModule(pl.LightningDataModule):
    def __init__(self, root_path, img_size, batch_size=64, num_workers=1):
        super().__init__()
        self.root_path = root_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage):
        # transform
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor()
        ])
        dataset = SampleImagesDataset(self.root_path, transform=transform)

        n_train = int(len(dataset) * 0.85)

        # train/val split
        train, val = random_split(dataset, [n_train, len(dataset) - n_train])

        # assign to use in dataloaders
        self.train_dataset = train
        self.val_dataset = val

    def train_dataloader(self):
        return FastDataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return FastDataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            drop_last=True
        )
