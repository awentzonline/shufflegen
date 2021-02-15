import os
import pickle

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from .fast_dataloader import FastDataLoader


class JustImagesDataset(VisionDataset):
    """Just batches of some images from a folder. No classes or anything."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_filenames = list_files_recursive(self.root, IMG_EXTENSIONS)
        self._image_cache = {}

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        if filename not in self._image_cache:
            img = pil_loader(filename)
            xformed = self.transform(img)
            self._image_cache[filename] = xformed
        return self._image_cache[filename]

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


class JustImagesDataModule(pl.LightningDataModule):
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
        dataset = JustImagesDataset(self.root_path, transform=transform)

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
