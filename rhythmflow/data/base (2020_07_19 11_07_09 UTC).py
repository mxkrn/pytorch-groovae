import glob
import numpy as np
import note_seq
import os
import torch


from torch.utils.data import Dataset
from .converters import drums_lib
from data.constants import DATADIRS, NUM_DRUM_PITCH_CLASSES, DRUM_PITCH_IDX


class BaseSequenceDataset(Dataset):
    def __init__(
        self, dataset_name="drumlab", splits=[0.8, 0.1, 0.1], shuffle=True, split="train"
    ):
        self.files = self._get_files(dataset_name)
        data = self._create_splits(splits, shuffle)
        self.loaded_files = data[split]
        self._extract_params()

    def _extract_params(self):
        """Extract parameters based on first file"""
        self.params = dict()
        sample = self._track_from_midi(self.loaded_files[0])
        self.input_size = sample.shape

    def _get_files(self, dataset_name):
        try:
            files = glob.glob(f"{DATADIRS[dataset_name]}*.mid")
        except KeyError as e:
            print("please check dataset/constants.py that the dataset name exists")
            raise e
        if len(files) > 0:
            return files
        else:
            print(f"no files found at {DATADIRS[dataset_name]}*.mid")
            raise FileNotFoundError

    def _track_from_midi(self, fname: str) -> list:
        with open(fname, "rb") as f:
            sequence = note_seq.midi_io.midi_to_note_sequence(f.read())
        quantized_sequence = note_seq.sequences_lib.quantize_note_sequence(
            sequence, steps_per_quarter=4
        )
        drum_track = drums_lib.DrumTrack()
        drum_track.from_quantized_sequence(quantized_sequence)
        return self._one_hot_track(drum_track._events)

    def _one_hot_track(self, events) -> np.array:
        data = torch.zeros((len(events), 1, NUM_DRUM_PITCH_CLASSES), dtype=int)
        for step, event in enumerate(events):
            for pitch in event:
                index = DRUM_PITCH_IDX[pitch]
                data[step][0][index] = 1
        return data

    def _format_track(self, track, trim=True, split=False):
        return track

    def _create_splits(self, splits, shuffle):
        length = len(self.files)
        if shuffle:
            idx = np.random.permutation(length)
            self.files = [self.files[i] for i in idx]
        idx = np.linspace(0, length - 1, length).astype("int")

        train_idx = idx[: int(splits[0] * length)]
        train = [self.files[i] for i in train_idx]

        valid_idx = idx[int(splits[0] * length): int((splits[0] + splits[1]) * length)]
        valid = [self.files[i] for i in valid_idx]

        test_idx = idx[int((splits[0] + splits[1]) * length):]
        test = [self.files[i] for i in test_idx]

        return {"train": train, "valid": valid, "test": test}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sequence = self._track_from_midi(self.loaded_files[idx])
        sample = torch.as_tensor(sequence)
        return sample

    def __len__(self):
        return len(self.loaded_files)
