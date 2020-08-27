from util.config import Config


def test_parser():
    config = Config()
    assert len(config.__dict__) > 0
