import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, latent_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.input = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout)

    def forward(self, input, hidden):
        output = self.input(input)
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        return output, hidden
