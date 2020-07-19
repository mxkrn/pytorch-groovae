from models.vae import VAE
from models.nnet.rnn import EncoderRNN, DecoderRNN
from data.constants import NUM_DRUM_PITCH_CLASSES
from data.base import BaseSequenceDataset


def test_vae():
    # initialize autoencoder
    n_hidden = 128
    latent_dims = 4
    encoder = EncoderRNN(input_size=NUM_DRUM_PITCH_CLASSES, hidden_size=n_hidden, latent_size=latent_dims)
    decoder = DecoderRNN(hidden_size=n_hidden, output_size=NUM_DRUM_PITCH_CLASSES)
    autoencoder = VAE(encoder=encoder, decoder=decoder, encoder_dims=n_hidden, latent_dims=latent_dims)

    # get sample
    dataset = BaseSequenceDataset()
    sample = dataset._track_from_midi(dataset.loaded_files[0]).float()
    hidden = encoder.initHidden()

    recon_batch, z_loss, mu, log_var = autoencoder(sample, hidden)
    print(recon_batch)