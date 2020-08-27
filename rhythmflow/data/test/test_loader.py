from data.loader import load_dataset
from util.config import Config


def test_data_loader():
    config = Config("train")
    loader = load_dataset(config)
    i = 0
    print(loader)
    for x in loader[0]['test']:
        if i < 1:
            print(x)
            i += 1
    assert(x == 1)
