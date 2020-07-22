from __future__ import print_function
import torch


class Config:
    def __init__(self, **kwargs):
        """
        Object containing the configurations for model construction and training parameters
        To change a default value simply pass a key:value combo and it will be loaded from kwargs
        """
        self.params = self._load_params()
        for key, value in kwargs.items():
            if key in self.params.keys():
                self.params[key] = value
            else:
                print(f'invalid configuration item: {key}')
        self.__dict__ = self.params
        self._check_args()
        self._cuda()

    def _load_params(self):
        return {
            "dataset": "drumlab",
            "dataset_type": "groove",
            "output": "outputs",
            "train_type": "random",
            "nbworkers": 0,
            "model": "vae",
            "loss": "mse",
            "rec_loss_type": "mse",
            "n_hidden": 256,
            "n_layers": 3,
            "hidden_size": 256,
            "latent_size": 5,
            "note_dropout": 0.1,
            "flow": "iaf",  # flow
            "flow_length": 16,
            "regressor": "",  # regressor
            "reg_layers": 3,
            "reg_hiddens": 128,
            "reg_flow": "maf",
            "reg_factor": 1e3,
            "loss_type": "mse",  # optimization
            "early_stop": 60,
            "plot_interval": 100,
            "batch_size": 8,
            "epochs": 5,
            "eval": 100,
            "lr": 2e-4,
            "semantic_dim": -1,  # semantic
            "dise_layers": 8,
            "dise_approach": "density",
            "start_disentangle": 100,
            "warm_disentangle": 25,
            "batch_evals": 8,
            "batch_out": 8,
            "check_exists": False,
            "time_limit": -1,
            "device": ""
        }

    def _check_args(self):
        if self.device != "cpu":
            torch.backends.cudnn.benchmark = True  # Enable CuDNN optimization

    def model_name(self):
        model_name = f"{self.model}_{self.dataset}_{self.loss}_{self.latent_dims}"
        return model_name

    def _cuda(self):
        self.cuda = torch.cuda.is_available()
        if self.device == "cpu":
            pass
        if self.cuda:
            self.device = torch.device(torch.cuda.current_device())
            print(f"optimization will be on {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("optimization will be on cpu")

    def __str__(self):
        return f"{self.model}_{self.dataset}_{self.loss}_{self.latent_dims}"