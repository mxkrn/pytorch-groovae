from torch.utils.data import DataLoader

from .base import GrooveDataset

# TODO: Add DataLoader caching, see -> https://github.com/pytorch/pytorch/pull/39274/files


def load_dataset(config, **kwargs):
    dataset = {"train": [], "valid": [], "test": []}
    loader = {"train": [], "valid": [], "test": []}
    for key in dataset.keys():
        dataset[key] = GrooveDataset(config.dataset_name, split=key)
        if key == "train":
            loader[key] = DataLoader(
                dataset[key],
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.nbworkers,
                pin_memory=False,
                **kwargs
            )
        else:
            loader[key] = DataLoader(
                dataset[key],
                batch_size=config.batch_size,
                shuffle=(config.train_type == "random"),
                num_workers=config.nbworkers,
                pin_memory=False,
                **kwargs
            )
    config.input_size = dataset["train"].input_size[2]
    config.output_size = config.input_size
    return loader, config
