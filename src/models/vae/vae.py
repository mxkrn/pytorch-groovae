import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.mu = nn.Sequential(
            nn.Linear(self._hidden_size, self._latent_size)
        )
        self.log_var = nn.Sequential(
            nn.Linear(self._hidden_size, self._latent_size),
            nn.Softplus()
        )

        # we are decompressing the latent space to generate a new sequence
        # TODO: Autoregressive sampler
        self.decompress_latent = nn.Linear(self._latent_size, self._latent_size*SEQUENCE_LENGTH)

        # latent decoder
        self.from_latent = nn.Linear(self._latent_size, self._hidden_size)

    def sample(self, x, hidden, target):
        output, _, r_loss = self.forward(x, hidden, target)  # Pass x thru the autoencoder network
        output = self._sample(output)
        return output

    def z_sample(self, mu, log_var, hidden, target):
        output = self._z_sample(mu, log_var, hidden, target)
        return output

    def forward(self, x, hidden, target):
        mu, log_var = self._encode(x)  # Gaussian encoding
        z, z_loss = self._reparametrize(mu, log_var)
        output, r_loss = self._decode(z, target)  # Decode the samples
        return output, z_loss, r_loss

    def _encode(self, x):
        encoder_output, _ = self.encoder(x)  # h = input for latent variables
        mu = self.mu(encoder_output)
        log_var = self.log_var(encoder_output)
        return mu, log_var

    def _decode(self, z, target):
        z = torch.reshape(z, (self._batch_size, 1, self._latent_size))  # we need to accomodate the sequence dimension
        output = self.decompress_latent(z)
        output = torch.reshape(output, (self._batch_size, SEQUENCE_LENGTH, self._latent_size))
        output = self.from_latent(output)
        output, r_loss = self.decoder(output, target)
        return output, r_loss

    def _reparametrize(self, mu, log_var):
        """
        Latent samples by reparametrization technique
        KL divergence from https://arxiv.org/pdf/1312.6114.pdf
        """
        eps = torch.randn_like(mu)  # standard normal distribution
        z = (log_var.exp().sqrt() * eps) + mu  # reparametrize
        kl_div = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / mu.size(0)
        return z, kl_div

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

        onsets_distrib = torch.distributions.bernoulli.Bernoulli(logits=onsets)
        onsets_sample = onsets_distrib.sample()

        velocities_sample = torch.sigmoid(velocities)
        offsets_sample = F.tanh(offsets)

        velocities_sample = velocities_sample
        offsets_sample = offsets_sample

        return torch.cat([onsets_sample, velocities_sample, offsets_sample], dim=2)

    def _z_sample(self, mu, log_var, hidden, target):
        eps = torch.randn_like(mu)
        z = (log_var.exp().sqrt() * eps) + mu
        output, _ = self._decode(z, hidden, target)
        output = self._sample(output)
        return output


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
