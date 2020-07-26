import torch
import torch.nn as nn
import numpy as np


class VAE(nn.Module):
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
        super(VAE, self).__init__()
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

        self.apply(self._init_parameters)

    def sample(self, z):
        return self.decoder.sample(z)

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

    # def _gaussian_decode(self, z, hidden, target):
    #     output, r_loss = self.decoder(z)
    #     output = output.view(-1, np.prod(self._input_size))
    #     mu = self.mu_decoder(output)
    #     log_var = self.log_var_decoder(output)
    #     q = distrib.Normal(torch.zeros(mu.size(1)), torch.ones(log_var.size(1)))
    #     eps = q.sample((mu.size(0), )).detach().to(self.device)
    #     output = (log_var.exp().sqrt() * eps) + mu
    #     return output.view(-1, * self.input_dims)

    def _reparametrize(self, mu, log_var):
        """
        Latent samples by reparametrization technique
        KL divergence from https://arxiv.org/pdf/1312.6114.pdf
        """
        eps = torch.randn_like(mu).detach().to(mu.device)  # re-parametrize
        z = (log_var.exp().sqrt() * eps) + mu  # get latent vector

        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_div = kl_div / mu.size(0)
        return z, kl_div

    @staticmethod
    def _init_parameters(module):
        if type(module) == nn.Linear:
            torch.nn.init.uniform_(module.weight)
            module.bias.data.fill_(0.1)


# class VAEFlow(VAE):
#     def __init__(self, encoder, decoder, flow, input_dims, encoder_dims, latent_dims):
#         super(VAEFlow, self).__init__(
#             encoder, decoder, input_dims, encoder_dims, latent_dims
#         )
#         self.flow_enc = nn.Linear(encoder_dims, flow.n_parameters())
#         self.flow = flow
#         self.apply(self.init_parameters)

#     def init_parameters(self, m):
#         if type(m) == nn.Linear or type(m) == nn.Conv2d:
#             m.weight.data.uniform_(-0.01, 0.01)
#             m.bias.data.fill_(0.0)

#     def encode(self, x):
#         h, _ = self.encoder(x)
#         mu = self.mu(h)
#         log_var = self.log_var(h)
#         flow_params = self.flow_enc(x)
#         return mu, log_var, flow_params

#     def latent(self, x, mu, log_var, flow_params):
#         # Obtain our first set of latent points
#         eps = torch.randn_like(mu).detach().to(x.device)
#         z_0 = (log_var.exp().sqrt() * eps) + mu

#         self.flow.set_parameters(flow_params)  # Update flows parameters

#         z_k, list_ladj = self.flow(z_0)  # Complexify posterior with flows
#         log_p_zk = torch.sum(-0.5 * z_k * z_k, dim=1)
#         log_q_z0 = torch.sum(
#             -0.5 * (log_var + (z_0 - mu) * (z_0 - mu) * log_var.exp().reciprocal()),
#             dim=1,
#         )
#         logs = (log_q_z0 - log_p_zk).sum()
#         ladj = torch.cat(list_ladj, dim=1)
#         logs -= torch.sum(ladj)
#         return z_k, (logs / float(x.size(0)))
