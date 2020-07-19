import traceback
import torch.nn as nn
import torch.optim as optim

from .nnet.rnn import EncoderRNN, DecoderRNN
from .constructors import (
    # construct_encoder_decoder,
    construct_flow,
    # construct_disentangle,
    # construct_regressor,
)
from .vae import AE, VAE, VAEFlow  # RegressionAE, DisentanglingAE
from .util import multinomial_loss, multinomial_mse_loss


class ModelConstructor:
    def __init__(self, config):
        """This is a class wrapper for constructing a model from the parsed arguments"""
        self.config = config
        self.model_type = config.model

        self._format()
        self._model = self._get_model()
        # if len(args.ref_model > 0):
        #     self._load_ref_model()
        self.optimizer = optim.Adam(self._model.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            verbose=True,
            threshold=1e-7,
        )
        self.loss = self._get_loss()

    def _format(self):
        if self.config.loss in ["multinomial"]:
            self.config.output_size *= self.config.n_classes
        elif self.config.loss in ["multi_mse"]:
            self.config.output_size *= self.config.args.n_classes + 1

    def _get_model(self, default="vae"):
        models = {
            # TODO: Benchmarks
            # "mlp": GatedMLP(
            #     np.prod(self.config.input_size),
            #     self.config.output_size,
            #     hidden_size=self.config.n_hidden,
            #     n_layers=self.config.n_layers,
            #     type_mod="normal",
            # ),
            # "gated_mlp": GatedMLP(
            #     np.prod(self.config.input_size),
            #     self.config.output_size,
            #     hidden_size=self.config.n_hidden,
            #     n_layers=self.config.n_layers,
            #     type_mod="gated",
            # ),
            "ae": self._get_flow_model(),
            "vae": self._get_flow_model(),
            "wae": self._get_flow_model(),
            "vae_flow": self._get_flow_model(),
        }
        try:
            return models[self.model_type]
        except KeyError:
            print(
                f"invalid model type {self.model_type}; using default model_type {default}"
            )
            return models[default]
        except Exception as e:
            traceback.format_exc(e)
            print("something went wrong initializing the model")

    def _get_flow_model(self):
        # rec_loss, encoder / decoder, and flow (if appropriate)
        # rec_loss = self._get_rec_loss()
        encoder = EncoderRNN(self.config.input_size, self.config.encoder_dims, self.config.latent_dims)
        decoder = DecoderRNN(self.config.encoder_dims, self.config.input_size)
        flow, blocks = construct_flow(
            self.config.latent_dims,
            flow_type=self.config.flow,
            flow_length=self.config.flow_length,
            amortization="input",
        )

        # AE model
        models = {
            "ae": AE(
                encoder, decoder, self.config.encoder_dims, self.config.latent_dims
            ),
            "vae": VAE(
                encoder,
                decoder,
                self.config.encoder_dims,
                self.config.latent_dims
            ),
            "vae_flow": VAEFlow(
                encoder,
                decoder,
                flow,
                self.config.input_size,
                self.config.encoder_dims,
                self.config.latent_dims,
            ),
        }
        model = models[self.model_type]

        # # flow-type
        # regression_model = construct_regressor(
        #     self.config.latent_dims,
        #     self.config.output_size,
        #     model=self.config.regressor,
        #     hidden_dims=self.config.reg_hiddens,
        #     n_layers=self.config.reg_layers,
        #     flow_type=self.config.reg_flow,
        # )
        # if self.config.semantic_dim == -1:  # regression flow
        #     model = RegressionAE(
        #         model,
        #         self.config.latent_dims,
        #         self.config.output_size,
        #         rec_loss,
        #         regressor=regression_model,
        #         regressor_name=self.config.regressor,
        #     )  # regression model
        # else:  # disentangling flow
        #     disentangling = construct_disentangle(
        #         self.config.latent_dims,
        #         model=self.config.disentangling,
        #         semantic_dim=self.config.semantic_dim,
        #         n_layers=self.config.dis_layers,
        #         flow_type=self.config.reg_flow,
        #     )
        #     model = DisentanglingAE(
        #         model,
        #         self.config.latent_dims,
        #         self.config.output_size,
        #         rec_loss,
        #         regressor=regression_model,
        #         regressor_name=self.config.regressor,
        #         disentangling=disentangling,
        #         semantic_dim=self.config.semantic_dim,
        #     )
        return model

    # def _load_ref_model(self):
    #     print(f"Loading reference {self.args.ref_model}")
    #     ref_model = torch.load(self.config.ref_model)
    #     if self.config.regressor != "mlp":
    #         ref_model_ae = ref_model.ae_model.to(self.config.device)
    #         self.model.ae_model = None
    #         self.model.ae_model = ref_model_ae
    #         ref_model = None
    #     else:
    #         self.model = None
    #         self.model = ref_model.to(self.config.device)

    def _get_rec_loss(self, default="mse"):
        losses = {
            "mse": nn.MSELoss(reduction="sum").to(self.config.device),
            "l1": nn.SmoothL1Loss(reduction="sum").to(self.config.device),
            "multinomial": multinomial_loss,
            "multi_mse": multinomial_mse_loss,
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
