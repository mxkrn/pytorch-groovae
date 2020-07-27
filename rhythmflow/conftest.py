import pytest
from util.config import Config
from models.vae import BaseRNNEncoder, BaseRNNDecoder, VAE
from data.loader import load_dataset


"""
Global fixtures for testing

Check different model parameters using different global constants

"""
N_LAYERS = 2
BATCH_SIZE = 8
HIDDEN_SIZE = 256
LATENT_SIZE = 5


@pytest.fixture
def config():
    return Config(
        n_layers=N_LAYERS,
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE
    )


@pytest.fixture
def batch(config):
    loader = load_dataset(config)
    i = 0
    for x, y in loader[0]["train"]:
        if i < 1:
            i += 1
            sample = x.to(config.device)
            target = y.to(config.device)
            return sample, target
        else:
            break


@pytest.fixture
def encoder(config, batch):
    encoder = BaseRNNEncoder(
        config.input_size,
        config.hidden_size,
        config.latent_size,
        config.batch_size,
        n_layers=config.n_layers,
        encoder_type=config.encoder_type
    )
    return encoder


@pytest.fixture
def decoder(config, batch):
    decoder = BaseRNNDecoder(
        config.input_size,
        config.hidden_size,
        config.latent_size,
        config.batch_size,
        n_layers=config.n_layers,
        decoder_type=config.decoder_type
    )
    return decoder


@pytest.fixture
def vae(encoder, decoder, config):
    vae = VAE(
        encoder,
        decoder,
        config.input_size,
        config.hidden_size,
        config.latent_size,
        config.batch_size
    )
    return vae.to(config.device)
