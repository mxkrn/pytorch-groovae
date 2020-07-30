import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.constants import SEQUENCE_LENGTH, NUM_DRUM_PITCH_CLASSES


class BaseVAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        input_size,
        hidden_size,
        latent_size,
        batch_size,
        z_loss='kl'
    ):
        super(BaseVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self._z_loss = z_loss
        self._batch_size = batch_size
        self._build()

    def _build(self):
        self.encoder._build()
        self.decoder._build()

        # latent encoder
        self.mu = nn.Linear(self._hidden_size, self._latent_size)
        self.log_var = nn.Linear(self._hidden_size, self._latent_size)

        # latent decoder
        self.from_latent = nn.Linear(self._latent_size, self._hidden_size)
        gd_dims = np.prod(self._input_size)
        self.mu_decoder = nn.Linear(gd_dims, gd_dims)
        self.log_var_decoder = nn.Linear(gd_dims, gd_dims)

        # self.apply(self._init_parameters)

    def sample(self, x, hidden, target, z=None):
        if x is not None:
            output, _, r_loss = self.forward(x, hidden, target)  # Pass x thru the autoencoder network
        elif z is not None:
            hidden = self.encoder.init_hidden().to(z.device)
            output = self.decoder.sample(z, hidden)
        else:
            raise ValueError(
                f'you must pass one of x: [{SEQUENCE_LENGTH, self._input_size}] or'
                'z [{SEQUENCE_LENGTH, self._latent_size}]')

        output = self._sample(output)
        return output, r_loss

    def forward(self, x, hidden, target):
        mu, log_var, hidden = self._encode(x, hidden)  # Gaussian encoding
        z, z_loss = self._reparametrize(mu, log_var)
        output, r_loss = self._decode(z, hidden, target)  # Decode the samples
        return output, z_loss, r_loss

    def _encode(self, x, hidden):
        encoder_output, _ = self.encoder(x, hidden)  # h = input for latent variables
        mu = self.mu(encoder_output)
        log_var = self.log_var(encoder_output)
        return mu, log_var, hidden

    def _decode(self, z, hidden, target):
        output = self.from_latent(z)
        output, r_loss = self.decoder(output, hidden, target)
        return output, r_loss

    def _reparametrize(self, mu, log_var):
        """
        Latent samples by reparametrization technique
        KL divergence from https://arxiv.org/pdf/1312.6114.pdf
        """
        eps = torch.randn_like(mu).detach().to(mu.device)  # re-parametrize
        z = (log_var.exp().sqrt() * eps) + mu  # get latent vector

        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return z, (kl_div / self._batch_size)

    def _sample(self, output):
        raise NotImplementedError


class VAE(BaseVAE):

    def __init__(
        self,
        encoder,
        decoder,
        input_size,
        hidden_size,
        latent_size,
        batch_size,
        z_loss='kl'
    ):
        super(VAE, self).__init__(
            encoder,
            decoder,
            input_size,
            hidden_size,
            latent_size,
            batch_size,
            z_loss
        )

    def _sample(self, output):
        onsets, velocities, offsets = torch.split(
            output, NUM_DRUM_PITCH_CLASSES, len(output.size()) - 1)
        offsets = F.tanh(offsets)

        onsets_distrib = torch.distributions.bernoulli.Bernoulli(logits=onsets)
        onsets_sample = onsets_distrib.sample()

        velocities_distrib = torch.distributions.continuous_bernoulli.ContinuousBernoulli(logits=velocities)
        velocities_sample = velocities_distrib.sample()

        return torch.cat([onsets_sample, velocities_sample, offsets], dim=2)


class VAEFlow(BaseVAE):
    def __init__(self, encoder, decoder, flow, input_dims, encoder_dims, latent_dims):
        super(VAEFlow, self).__init__(
            encoder, decoder, input_dims, encoder_dims, latent_dims
        )
        self.flow_enc = nn.Linear(encoder_dims, flow.n_parameters())
        self.flow = flow
        self.apply(self.init_parameters)

    def encode(self, x):
        h, _ = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        flow_params = self.flow_enc(x)
        return mu, log_var, flow_params

    def latent(self, x, mu, log_var, flow_params):
        eps = torch.randn_like(mu).detach().to(x.device)
        z_0 = (log_var.exp().sqrt() * eps) + mu

        self.flow.set_parameters(flow_params)  # Update flows parameters
        z_k, logs = self._flow_reparametrize(z_0, mu, log_var)
        return z_k, (logs / self._batch_size)

    def _flow_reparametrize(self, z, mu, log_var):
        """
        Complexify posterior with flows
        """
        z_k, list_ladj = self.flow(z)
        log_p_zk = torch.sum(-0.5 * z_k * z_k, dim=1)
        log_q_z0 = torch.sum(
            -0.5 * (log_var + (z - mu) * (z - mu) * log_var.exp().reciprocal()),
            dim=1,
        )
        logs = (log_q_z0 - log_p_zk).sum()
        ladj = torch.cat(list_ladj, dim=1)
        logs -= torch.sum(ladj)
        return z_k, logs / self._batch_size
