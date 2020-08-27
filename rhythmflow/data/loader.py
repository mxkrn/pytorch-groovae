from torch.utils.data import DataLoader

from data.base import RhythmDataset

# TODO: Add DataLoader caching, see -> https://github.com/pytorch/pytorch/pull/39274/files


def load_dataset(config, **kwargs):
    dataset = {"train": [], "valid": [], "test": []}
    loader = {"train": [], "valid": [], "test": []}
    for key in dataset.keys():
        dataset[key] = RhythmDataset(config.dataset, split=key)
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
    config.input_size = dataset["train"].input_size[-1]
    config.output_size = config.input_size
    return loader, config

# DEBUG
# if __name__ == "__main__":
#     from util.config import Config
#     config = Config("train")
#     loader = load_dataset(config)
#     error = 0
#     total = 0
#     for i, x in enumerate(loader[0]['test']):
#         error += sum(x[1]).item()
#         total += 16 - sum(x[1]).item()
#     print(error)
