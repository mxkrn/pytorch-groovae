from __future__ import print_function
import argparse
import torch

from util.logger import get_logger


class Config:
    def __init__(self, parse=True, **kwargs):
        """
        Object containing the configurations for model construction and training parameters
        To change a default value simply pass a key:value combo and it will be loaded from kwargs
        """
        self.params = self._load_params()
        self.logger = get_logger(self.params['logging'])
        for key, value in kwargs.items():
            if key in self.params.keys():
                self.logger.info(f'new config: {key}={value}')
                self.params[key] = value
            else:
                self.logger.info(f'invalid configuration item: {key}')
        if parse:
            self._parse()
        self._cuda()
        self.__dict__ = self.params

    def add_argument(self, key, value, check_exists=True):
        if key in self.params.keys():
            self.params[key] = value
        elif not check_exists:
            self.params[key] = value
        else:
            print(f'invalid configuration argument: {key}')
            raise KeyError

    def _load_params(self):
        params = {
            "train": True,
            "logging": 0,
            "dataset": "gmd",
            "datadir": "/mnt/c/Users/maxkr/data/gmd_drumlab_merge/",
            "dataset_type": "groove",
            "output": "outputs",
            "train_type": "random",
            "nbworkers": 0,
            "encoder_type": "rnn",
            "decoder_type": "rnn",
            "model": "vae",
            "loss": "mse",
            "rec_loss_type": "mse",
            "n_hidden": 512,
            "n_layers": 1,
            "hidden_size": 256,
            "latent_size": 8,
            "note_dropout": 0.05,
            "start_regress": 15,
            "beta_factor": 1e3,  # latent loss weight
            "gamma_factor": 1,
            "warm_latent": 20,  # warm-up epochs for latent
            "flow": "iaf",  # flow
            "flow_length": 16,
            "regressor": "",  # regressor
            "reg_layers": 3,
            "reg_hiddens": 128,
            "reg_flow": "maf",
            "reg_factor": 1e3,
            "loss_type": "mse",  # optimization
            "early_stop": 40,
            "plot_interval": 100,
            "batch_size": 16,
            "epochs": 500,
            "eval": 100,
            "lr": 1e-4,
            "semantic_dim": -1,  # semantic
            "dise_layers": 8,
            "dise_approach": "density",
            "start_disentangle": 100,
            "warm_disentangle": 25,
            "batch_evals": 8,
            "batch_out": 8,
            "check_exists": False,
            "time_limit": -1,
            "device": "cpu"
        }
        return params

    def model_name(self):
        model_name = f"{self.model}_{self.encoder_type}_{self.loss}_" \
                    f"{self.n_layers}_{self.hidden_size}_beta{self.beta_factor}" \
                    f"{self.latent_size}_{self.batch_size}"
        return model_name

    def _cuda(self):
        if self.params["device"] == "cpu":
            self.params["device"] = torch.device("cpu")
            print("optimization will be on cpu")
        else:
            self.params["device"] = torch.device(torch.cuda.current_device())
            print(f"optimization will be on {torch.cuda.get_device_name()}")
            torch.backends.cudnn.benchmark = True  # Enable CuDNN optimization

    def _parse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.params.items():
            parser.add_argument(f'--{key}', type=type(value), default=value, required=False)
        args = parser.parse_args()
        for key, value in args.__dict__.items():
            self.add_argument(key, value)

    def __str__(self):
        return f"{self.model}_{self.encoder_type}_{self.dataset}_" \
            f"{self.loss}_{self.latent_size}"
