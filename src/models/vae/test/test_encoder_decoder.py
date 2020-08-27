import torch

from models.vae.encoder import BaseRNNEncoder
from models.vae.decoder import BaseRNNDecoder
from data.constants import SEQUENCE_LENGTH

# See conftest.py for fixtures


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
