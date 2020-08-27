from models.base import ModelConstructor
from util.config import Config
from data.constants import NUM_DRUM_PITCH_CLASSES


def test_initialization():
    config = Config('train')
    config.input_size = NUM_DRUM_PITCH_CLASSES
    model_constructor = ModelConstructor(config)
