import torch.nn as nn
import torch.optim as optim

from models.vae import BaseRNNEncoder, BaseRNNDecoder
# from .constructors import (
#     # construct_encoder_decoder,
#     construct_flow,
#     # construct_disentangle,
#     # construct_regressor,
# )
from .vae import VAE
from .util import multinomial_loss, multinomial_mse_loss


class ModelConstructor:
    def __init__(self, config):
        """This is a class wrapper for constructing a model from the parsed arguments"""
        self.config = config
        self.model_type = config.model

        self._model = self._get_model()
        self.optimizer = optim.Adam(self._model.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            verbose=True,
            threshold=1e-7,
        )
        self.rec_loss = self._get_rec_loss()
        self.loss = self._get_loss()

    def _get_model(self):
        encoder = BaseRNNEncoder(
            self.config.input_size,
            self.config.hidden_size,
            self.config.latent_size,
            self.config.batch_size,
            self.config.encoder_type,
            self.config.n_layers
        )
        decoder = BaseRNNDecoder(
            self.config.input_size,
            self.config.hidden_size,
            self.config.latent_size,
            self.config.batch_size,
            self.config.decoder_type,
            self.config.n_layers
        )

        models = {
            "vae": VAE(
                encoder,
                decoder,
                self.config.input_size,
                self.config.hidden_size,
                self.config.latent_size,
                self.config.hidden_size
            )
        }
        return models[self.model_type]

    def _get_rec_loss(self, default="mse"):
        losses = {
            "mse": nn.MSELoss(reduction="sum").to(self.config.device)
        }
        try:
            return losses[self.config.rec_loss_type]
        except KeyError:
            print(
                f"Unknown reconstruction loss {self.args.rec_loss}, using default reconstruction loss_type: {default}"
            )
            return losses[default]

    def _get_loss(self, default="mse"):
        losses = {
            "mse": nn.MSELoss(reduction="mean").to(self.config.device),
            "l1": nn.SmoothL1Loss(reduction="mean").to(self.config.device),
            "bce": nn.BCELoss(reduction="mean").to(self.config.device),
            "multinomial": multinomial_loss,
            "multi_mse": multinomial_mse_loss,
        }
        try:
            return losses[self.config.loss_type]
        except KeyError:
            print(f"Unknown loss {self.args.loss}, using default loss_type: {default}")
            return losses[default]

    def __str__(self):
        return f"Model: {self.model_type.__str__}; Loss: {self.loss.__str__}"
