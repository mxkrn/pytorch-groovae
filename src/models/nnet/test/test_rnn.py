from models.nnet.rnn import EncoderRNN, DecoderRNN
from data.constants import NUM_DRUM_PITCH_CLASSES
from data.base import BaseSequenceDataset


def test_encoder_deocder():
    # initialize RNN
    n_hidden = 128
    latent_size = 2
    encoder = EncoderRNN(NUM_DRUM_PITCH_CLASSES, n_hidden, latent_size)
    decoder = DecoderRNN(n_hidden, NUM_DRUM_PITCH_CLASSES)
    # get sample
    dataset = BaseSequenceDataset()
    sample = dataset._track_from_midi(dataset.loaded_files[0]).float()
    hidden = encoder.initHidden()
    output, next_hidden = encoder(sample, hidden)
    x_tilde, _ = decoder(output, next_hidden)
    assert(x_tilde.size(2) == NUM_DRUM_PITCH_CLASSES)
