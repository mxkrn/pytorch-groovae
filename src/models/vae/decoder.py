import torch
import torch.nn as nn
import torch.nn.functional as F

from data.constants import NUM_DRUM_PITCH_CLASSES


class BaseRNNDecoder(nn.Module):

    def __init__(
        self,
        output_size,
        hidden_size,
        latent_size,
        batch_size,
        decoder_type='rnn',
        n_layers=1,
        bidirectional=False
    ):
        super(BaseRNNDecoder, self).__init__()
        self._output_size = output_size
        self._num_pitch_classes = output_size / 3
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self._batch_size = batch_size
        self._decoder_type = decoder_type
        self._n_layers = n_layers
        self._bidirectional = bidirectional
        self._build()

    def forward(self, z, hidden, target):
        output = self._decode(z, hidden)
        r_loss = self._reconstruction_loss(output, target)
        return output, r_loss

    def sample(self, z):
        hidden = self.init_hidden()
        output, _ = self._decoder_layer(z, hidden)
        return self._output_layer(output)

    def _build(self):
        types = {
            "rnn": nn.RNN(
                self._hidden_size,
                self._hidden_size,
                self._n_layers,
                nonlinearity="relu",
                batch_first=True,
                bidirectional=self._bidirectional),
            "gru": nn.GRU(
                self._hidden_size,
                self._hidden_size,
                self._n_layers,
                batch_first=True,
                bidirectional=self._bidirectional
            )
        }
        try:
            self._decoder_layer = types[self._decoder_type]
        except KeyError:
            print(f'invalid decoder type: {self._decoder_type}')
            raise
        self._decoder_layer.flatten_parameters()

        self._output_layer = nn.Linear(self._hidden_size, self._output_size)

    def _decode(self, z, hidden):
        output, _ = self._decoder_layer(z, hidden)
        output = self._output_layer(output)
        return output

    @staticmethod
    def _reconstruction_loss(output, target):
        onsets, velocities, offsets = torch.split(
            output, NUM_DRUM_PITCH_CLASSES, len(output.size()) - 1)
        target_onsets, target_velocities, target_offsets = torch.split(
            target, NUM_DRUM_PITCH_CLASSES, len(target.size()) - 1)

        onset_loss = F.binary_cross_entropy(torch.sigmoid(onsets), target_onsets)
        velocity_loss = F.mse_loss(F.relu(velocities), target_velocities)
        offset_loss = F.mse_loss(F.hardtanh(offsets), offsets)
        loss = onset_loss + velocity_loss + offset_loss
        return loss

    def init_hidden(self):
        return torch.zeros(self._n_layers, self._batch_size, self._hidden_size, dtype=torch.float)
