import glob
import numpy as np
import note_seq
import torch

from torch.utils.data import Dataset
from data.converters import groove
from data.constants import DATADIRS, DRUM_PITCH_CLASSES, SEQUENCE_LENGTH


class RhythmDataset(Dataset):
    def __init__(
        self,
        dataset_name="gmd",
        splits=[0.8, 0.1, 0.1],
        shuffle=True,
        split="train",
        humanize=True
    ):
        try:
            files = glob.glob(f"{DATADIRS[dataset_name]}*.mid")
        except KeyError as e:
            print(f"please check data/constants.py that a directory for {dataset_name} exists")
            raise e
        if len(files) == 0:
            print(f"no files found in {DATADIRS[dataset_name]}")
            raise FileNotFoundError

        data = self._create_splits(files, splits, shuffle)
        self.files = data[split]

        try:
            self.pitch_classes = DRUM_PITCH_CLASSES[dataset_name]
        except KeyError:
            print(f"check data/constants that a pitch class for {dataset_name} exists")
            raise
        self.humanize = humanize
        self.input_size = self._get_input_size()

    def midi_to_tensor(self, fname: str):
        with open(fname, "rb") as f:
            sequence = note_seq.midi_io.midi_to_note_sequence(f.read())
            quantized_sequence = note_seq.sequences_lib.quantize_note_sequence(
                sequence, steps_per_quarter=4
            )
        converter = groove.GrooveConverter(
            split_bars=2,
            steps_per_quarter=4,
            quarters_per_bar=4,
            pitch_classes=self.pitch_classes,
            humanize=self.humanize
        )
        tensor = converter.to_tensors(quantized_sequence)
        return tensor

    def _get_input_size(self):
        """Extract input size"""
        self.params = dict()
        i = 0
        while True:
            sample = self.midi_to_tensor(self.files[i])
            try:
                return sample.inputs[0].shape
            except IndexError:
                i += 1
                continue

    @staticmethod
    def _create_splits(files, splits, shuffle):
        if shuffle:
            idx = np.random.permutation(len(files))
            files = [files[i] for i in idx]
        idx = np.linspace(0, len(files) - 1, len(files)).astype("int")

        train_idx = idx[: int(splits[0] * len(files))]
        valid_idx = idx[int(splits[0] * len(files)): int((splits[0] + splits[1]) * len(files))]
        test_idx = idx[int((splits[0] + splits[1]) * len(files)):]

        train = [files[i] for i in train_idx]
        valid = [files[i] for i in valid_idx]
        test = [files[i] for i in test_idx]

        return {"train": train, "valid": valid, "test": test}

    def __getitem__(self, idx):
        # TODO: Better error handling thru preprocessing
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tensor = self.midi_to_tensor(self.files[idx])
        if len(tensor.inputs) != 0:
            if len(tensor.inputs[0]) == SEQUENCE_LENGTH:
                return tensor
        return torch.zeros(self.input_size, dtype=torch.float)

    def __len__(self):
        return len(self.files)


# DEBUG
# if __name__ == "__main__":
#     dataset = RhythmDataset()
#     sample = dataset.midi_to_tensor(dataset.files[0])
#     print(sample)
