import torch
import numpy as np
import torch.nn as nn
import torch.distributions as distrib
import torch.nn.functional as F

from .ae import AE
from ..util.divergence import kl_divergence, mmd_divergence


class VAE(AE):
    def __init__(
        self,
        encoder,
        decoder,
        encoder_dims,
        latent_dims,
        divergence='kl'
    ):
        super(VAE, self).__init__(encoder, decoder, encoder_dims, latent_dims)
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_dims = encoder_dims
        self.latent_dims = latent_dims
        self.divergence = divergence

        # Latent gaussians
        self.mu = nn.Linear(encoder_dims, latent_dims)
        self.log_var = nn.Linear(encoder_dims, latent_dims)
        self.apply(self.init_parameters)

    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x, hidden):
        h, _ = self.encoder(x, hidden)  # h = input for latent variables
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var, hidden

    # def decode(self, z, hidden):
    #     if self.gaussian_dec:
    #         x_vals, hidden = self.decoder(y, hidden)
    #         x_vals = x_vals.view(-1, np.prod(self.input_dims))
    #         mu = self.mu_dec(x_vals)
    #         log_var = self.log_var_dec(x_vals)
    #         q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(log_var.shape[1]))
    #         eps = q.sample((y.size(0),)).detach().to(y.device)
    #         x_tilde = (log_var.exp().sqrt() * eps) + mu
    #         x_tilde = x_tilde.view(-1, *self.input_dims)
    #     else:
    #         x_tilde = self.decoder(y)
    #     return x_tilde

    def decode(self, z, hidden):
        y = self.from_latent(z)
        return self.decoder(y, hidden)

    def forward(self, x, hidden):
        mu, log_var, hidden = self.encode(x, hidden)
        z, z_loss = self.reparametrize(x, mu, log_var)
        y = self.decode(z, hidden)  # Decode the samples
        return y, z_loss, mu, log_var

    def reparametrize(self, x, mu, log_var):
        """
        Latent samples by reparametrization technique,
        latent loss is one of ['mmd', 'kl'] divergence
        """
        eps = torch.randn_like(mu).detach().to(x.device)
        z = (log_var.exp().sqrt() * eps) + mu
        if self.divergence == "kl":
            return z, kl_divergence(mu, log_var, x.size(0))
        elif self.divergence == "mmd":
            q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(log_var.shape[1]))
            z_prior = q.sample((x.size(0),)).to(x.device)
            mmd_dist = mmd_divergence(z, z_prior)
            return z, mmd_dist


class VAEFlow(VAE):
    def __init__(self, encoder, decoder, flow, input_dims, encoder_dims, latent_dims):
        super(VAEFlow, self).__init__(
            encoder, decoder, input_dims, encoder_dims, latent_dims
        )
        self.flow_enc = nn.Linear(encoder_dims, flow.n_parameters())
        self.flow = flow
        self.apply(self.init_parameters)

    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            m.weight.data.uniform_(-0.01, 0.01)
            m.bias.data.fill_(0.0)

    def encode(self, x):
        h, _ = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        flow_params = self.flow_enc(x)
        return mu, log_var, flow_params

    def latent(self, x, mu, log_var, flow_params):
        # Obtain our first set of latent points
        eps = torch.randn_like(mu).detach().to(x.device)
        z_0 = (log_var.exp().sqrt() * eps) + mu

        self.flow.set_parameters(flow_params)  # Update flows parameters

        z_k, list_ladj = self.flow(z_0)  # Complexify posterior with flows
        log_p_zk = torch.sum(-0.5 * z_k * z_k, dim=1)
        log_q_z0 = torch.sum(
            -0.5 * (log_var + (z_0 - mu) * (z_0 - mu) * log_var.exp().reciprocal()),
            dim=1,
        )
        logs = (log_q_z0 - log_p_zk).sum()
        ladj = torch.cat(list_ladj, dim=1)
        logs -= torch.sum(ladj)
        return z_k, (logs / float(x.size(0)))
