import torch
import torch.nn as nn


class BaseRNNEncoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        latent_size,
        batch_size,
        encoder_type='rnn',
        n_layers=1,
        dropout=0,
        bidirectional=False
    ):
        super(BaseRNNEncoder, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self._batch_size = batch_size
        self._encoder_type = encoder_type
        self._n_layers = n_layers
        self._dropout = dropout
        self._bidirectional = bidirectional
        self._build()

    def _build(self):
        self._input_layer = nn.Linear(self._input_size, self._hidden_size)
        types = {
            "rnn": nn.RNN(
                self._hidden_size,
                self._hidden_size,
                self._n_layers,
                batch_first=True,
                dropout=self._dropout,
                bidirectional=self._bidirectional),
            "lstm": nn.LSTM(
                self._hidden_size,
                self._hidden_size,
                self._n_layers,
                batch_first=True,
                dropout=self._dropout,
                bidirectional=self._bidirectional
            ),
            "gru": nn.GRU(
                self._hidden_size,
                self._hidden_size,
                self._n_layers,
                batch_first=True,
                dropout=self._dropout,
                bidirectional=self._bidirectional
            )
        }
        try:
            self._encoder_layer = types[self._encoder_type]
        except KeyError:
            print(f'invalid encoder type: {self._encoder_type}')
            raise
        self._encoder_layer.flatten_parameters()

    def forward(self, x):
        hidden = self.init_hidden().to(x.device)
        output = self._input_layer(x)
        output, hidden = self._encoder_layer(output, hidden)
        # output = torch.transpose(0, 1)  # [batch, time, sample] -> [time, batch, sample]
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self._n_layers, self._batch_size, self._hidden_size, dtype=torch.float)
