from __future__ import print_function
import re
import torch


class Config:
    def __init__(self, routine="train", check=True, *args, **kwargs):
        # initialize
        self.args = {}
        self.routine = routine
        self._routine_fn(routine)()
        self.__dict__ = self.args  # allow attribute access

        # parse and check
        self._check_args()
        self._cuda()

    def _routine_fn(self, check):
        routines = {"train": self._train}
        try:
            return routines[check]
        except KeyError:
            print(f"unknown parser configuration {check}")
            exit(1)

    def add_argument(self, name: str, type, default=None, help=None):
        if default is not None:
            self.args[re.split("--", name)[-1]] = default
        else:
            print(f"{ValueError}: {name} cannot be None.")
            exit(1)

    def _train(self):
        self.add_argument(
            "--dataset_name", type=str, default="gmd", help="Name of dataset to be used"
        )
        self.add_argument(
            "--output", type=str, default="outputs", help="Path to output directory"
        )
        self.add_argument(
            "--dataset", type=str, default="32par", help="Name of the dataset"
        )
        self.add_argument(
            "--data", type=str, default="mel", help="Type of data to train on"
        )
        self.add_argument(
            "--train_type", type=str, default="fixed", help="Fixed or random data split"
        )
        self.add_argument(
            "--nbworkers", type=int, default=0, help="Number of workers for parallel"
        )

        # Model arguments
        self.add_argument(
            "--model", type=str, default="vae", help="Type of model (MLP, VAE, ...)"
        )
        self.add_argument(
            "--loss", type=str, default="mse", help="Loss for parameter regression"
        )
        self.add_argument(
            "--rec_loss", type=str, default="mse", help="Reconstruction loss"
        )
        self.add_argument(
            "--n_classes", type=int, default=61, help="Classes for multinoulli loss"
        )
        self.add_argument(
            "--n_hidden", type=int, default=128, help="Number of hidden units"
        )
        self.add_argument(
            "--n_layers", type=int, default=1, help="Number of computing layers"
        )

        # CNN parameters
        # self.add_argument('--channels',       type=int,   default=64, help='Number of channels in convolution')
        # self.add_argument('--kernel',         type=int,   default=5, help='Size of convolution kernel')
        # self.add_argument('--dilation',       type=int,   default=3, help='Dilation factor of convolution')

        # AE-specific parameters
        self.add_argument(
            "--layers", type=str, default="rnn", help="Type of layers in the model",
        )
        self.add_argument(
            "--encoder_dims", type=int, default=128, help="Encoder output dimensions"
        )
        self.add_argument(
            "--latent_dims", type=int, default=4, help="Number of latent dimensions"
        )
        self.add_argument(
            "--warm_latent", type=int, default=50, help="Warmup epochs for latent"
        )
        self.add_argument(
            "--start_regress", type=int, default=100, help="Epoch to start regression"
        )
        self.add_argument(
            "--warm_regress", type=int, default=100, help="Warmup epochs for regression"
        )
        self.add_argument(
            "--beta_factor", type=int, default=1, help="Beta factor in VAE"
        )

        # Two-step training parameters
        self.add_argument("--ref_model", type=str, default="", help="Reference model")
        # Flow specific parameters
        self.add_argument("--flow", type=str, default="iaf", help="Type of flow to use")
        self.add_argument(
            "--flow_length", type=int, default=16, help="Number of flow transforms"
        )
        # Regression parameters
        self.add_argument("--regressor", type=str, default="", help="Type of regressor")
        self.add_argument(
            "--reg_layers", type=int, default=3, help="Number of regression layers"
        )
        self.add_argument(
            "--reg_hiddens", type=int, default=256, help="Number of units in regressor"
        )
        self.add_argument(
            "--reg_flow", type=str, default="maf", help="Type of flow in regressor"
        )
        self.add_argument(
            "--reg_factor", type=float, default=1e3, help="Regression loss weight"
        )

        # Optimization arguments
        self.add_argument(
            "--loss_type", type=str, default="multi_mse", help="Loss type"
        )
        self.add_argument("--k_run", type=int, default=0, help="ID of runs (k-folds)")
        self.add_argument("--early_stop", type=int, default=60, help="Early stopping")
        self.add_argument(
            "--plot_interval",
            type=int,
            default=100,
            help="Interval of plotting frequency",
        )
        self.add_argument(
            "--batch_size", type=int, default=64, help="Size of the batch"
        )
        self.add_argument(
            "--epochs", type=int, default=1, help="Number of epochs to train on"
        )
        self.add_argument(
            "--eval", type=int, default=100, help="Frequency of full evalution"
        )
        self.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

        # Semantic arguments
        self.add_argument(
            "--semantic_dim", type=int, default=-1, help="Using semantic dimension"
        )
        self.add_argument(
            "--dis_layers", type=int, default=8, help="Number of disentangling layers"
        )
        self.add_argument(
            "--disentangling",
            type=str,
            default="density",
            help="Type of disentangling approach",
        )
        self.add_argument(
            "--start_disentangle",
            type=int,
            default=100,
            help="Epoch to start disentangling",
        )
        self.add_argument(
            "--warm_disentangle", type=int, default=25, help="Warmup on disentanglement"
        )

        # Evaluation parameters
        self.add_argument(
            "--batch_evals", type=int, default=16, help="Number of batch to evaluate"
        )
        self.add_argument(
            "--batch_out", type=int, default=3, help="Number of batch to synthesize"
        )
        self.add_argument(
            "--check_exists", type=int, default=0, help="Check if model exists"
        )
        self.add_argument(
            "--time_limit", type=int, default=0, help="Maximum training time in mins"
        )

        # CUDA arguments
        self.add_argument("--device", type=str, default="cpu", help="Device for CUDA")

    def _check_args(self):
        if self.device != "cpu":
            torch.backends.cudnn.benchmark = True  # Enable CuDNN optimization

    def model_name(self):
        model_name = f"{self.model}_{self.data}_{self.loss}_{self.latent_dims}"
        if not (self.model in ["mlp", "gated_mlp", "cnn", "gated_cnn", "res_cnn"]):
            model_name += "_" + self.layers
            if self.model == "vae_flow":
                model_name += "_" + self.flow
            model_name += "_" + self.regressor
            if self.regressor != "mlp":
                model_name += "_" + self.reg_flow + "_" + str(self.reg_layers)
            if self.semantic_dim > -1:
                model_name += "_" + str(self.semantic_dim) + "_" + self.disentangling
        if self.k_run > 0:
            model_name += "_" + str(self.k_run)
        return model_name

    def _cuda(self):
        # Handling cuda
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device(torch.cuda.current_device())
            print(f"optimization will be on {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("optimization will be on cpu")

    def __str__(self):
        return f"Configuration: {self.__dict__}"
