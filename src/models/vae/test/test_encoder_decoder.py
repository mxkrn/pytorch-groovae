import pytest
import torch

from models.vae.encoder import BaseRNNEncoder
from models.vae.decoder import BaseRNNDecoder
from data.loader import load_dataset
from data.constants import SEQUENCE_LENGTH
from util.config import Config


@pytest.fixture
def config():
    return Config(
        n_layers=2,
        batch_size=2
    )


@pytest.fixture
def data(config):
    loader = load_dataset(config)
    i = 0
    for x, y in loader[0]["train"]:
        if i < 1:
            sample = x.to(config.device)
            target = y.to(config.device)
            return sample, target
        i += 1


def test_encoder(config, data):
    encoder = BaseRNNEncoder(
        config.input_size,
        config.hidden_size,
        config.latent_size,
        config.batch_size,
        n_layers=config.n_layers,
        encoder_type=config.encoder_type
    )
    encoder.to(config.device)
    encoder.train()
    hidden = encoder.init_hidden().to(config.device)
    z, hidden = encoder(data[0], hidden)
    assert z.size() == torch.Size([config.batch_size, SEQUENCE_LENGTH, config.hidden_size])
    assert hidden.size() == torch.Size([config.n_layers, config.batch_size, config.hidden_size])
    return z, hidden, data[1]


@pytest.fixture
def encoder_output_fixture(config, data):
    encoder = BaseRNNEncoder(
        config.input_size,
        config.hidden_size,
        config.latent_size,
        config.batch_size,
        n_layers=config.n_layers
    )
    encoder.to(config.device)
    encoder.train()
    hidden = encoder.init_hidden().to(config.device)
    z, hidden = encoder(data[0], hidden)
    return z, hidden, data[1]


def test_decoder(config, encoder_output_fixture):
    z = encoder_output_fixture[0]
    hidden = encoder_output_fixture[1]
    target = encoder_output_fixture[2]

    decoder = BaseRNNDecoder(
        config.input_size,
        config.hidden_size,
        config.latent_size,
        config.batch_size,
        n_layers=config.n_layers,
        encoder_type=config.decoder_type
    )
    decoder.to(config.device)
    decoder.train()
    hidden = decoder.init_hidden().to(config.device)
    output = decoder(z, hidden, target)
    assert output[0].size() == torch.Size([config.batch_size, SEQUENCE_LENGTH, config.input_size])
    assert type(output[1] == float)


# debug_config = Config(n_layers=2, batch_size=1)
# loader = load_dataset(debug_config)
# i = 0
# for x, y in loader[0]['valid']:
#     if i < 1:
#         sample = x.to(debug_config.device)
#         target = y.to(debug_config.device)
#     i += 1

# encoder_output = test_encoder(debug_config, (sample, target))
# test_decoder(debug_config, encoder_output)
