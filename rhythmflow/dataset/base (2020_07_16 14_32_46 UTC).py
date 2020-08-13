import os
import glob
import copy
import torch
import numpy as np
import note_seq

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.util.transforms import *
from utils.plot import plot_batch
from .constants import *


class RhythmDataset(Dataset):

    def __init(self, datadir, splits=[0.8, 0.1, 0.1], shuffle=True):

        with open(glob.glob(f'{datadir}/raw/*.json'), 'r') as file_in:
            json_data = json.load(file_in)

        self.data = self._extract_json(json_data)
        self._format_data()
        self._create_splits(splits, shuffle)

    def _extract_json(json: str) -> list:

        return

    def _crop_sequence(self):
        pass

    def _create_splits(self, splits, shuffle):
        length = self.__len__
        if shuffle:
            idx = np.random.permutation(length)
            self.data = [self.data[i] for i in idx]
        idx = np.linspace(0, length - 1, length).astype('int')
        train_idx = idx[:int(splits[0]*length)]
        valid_idx = idx[int(splits[0]*nb_files):int((splits[0]+splits[1])*nb_files)]
        test_idx = idx[int((splits[0]+splits[1])*nb_files):]

        self.valid = (
            [self.data[i, :] for i in valid_idx],
            [self.metadata[i, :] for i in valid_idx]
        )
        self.test = (
            [self.data[i, :] for i in test_idx],
            [self.metadata[i, :] for i in test_idx]
        )
        self.train = (
            [self.data[i, :] for i in train_idx],
            [self.metadata[i, :] for i in train_idx]
        )

    def __getitem__(self, idx):
        dat = self.data[idx]
        meta =

    def __len__(self):
        return len(self.data)


class SemanticRhythmDataset(RhythmDataset):

    def __init(self, *args, **kwargs):
        super(SemanticRhythmDataset, self).__init__(*args, **kwargs)
        with open(glob.glob(f'{datadir}/raw/*.json'), 'r') as file_in:
            json_data = json.load(file_in)

        self.data = self._extract_json(json_data)
        self._format_data()
        self._create_splits(splits, shuffle)

    def _extract_json(json: str) -> list:

        return

    def _crop_sequence(self):
        pass

    def _create_splits(self, splits, shuffle):
        length = self.__len__
        if shuffle:
            idx = np.random.permutation(length)
            self.data = [self.data[i] for i in idx]
        idx = np.linspace(0, length - 1, length).astype('int')
        train_idx = idx[:int(splits[0]*length)]
        valid_idx = idx[int(splits[0]*nb_files):int((splits[0]+splits[1])*nb_files)]
        test_idx = idx[int((splits[0]+splits[1])*nb_files):]

        self.valid = (
            [self.data[i, :] for i in valid_idx],
            [self.metadata[i, :] for i in valid_idx]
        )
        self.test = (
            [self.data[i, :] for i in test_idx],
            [self.metadata[i, :] for i in test_idx]
        )
        self.train = (
            [self.data[i, :] for i in train_idx],
            [self.metadata[i, :] for i in train_idx]
        )

    def __getitem__(self, idx):
        dat = self.data[idx]
        meta =

    def __len__(self):
        return len(self.data)