import torch
import traceback

import torch.nn as nn
from torch.distributions import MultivariateNormal

from .flows import (
    PlanarFlow,
    IAFlow,
    BatchNormFlow,
    ShuffleFlow,
    MAFlow,
    ReverseFlow,
    NormalizingFlow,
    DisentanglingFlow,
    TriangularSylvesterFlow,
    MaskedCouplingFlow,
    DeepSigmoidFlow,
    DeepDenseSigmoidFlow,
    ContextIAFlow,
    ContextMAFlow,
    DDSF_IAFlow,
)
from .util import (
    BayesianRegressor,
    FlowTransform,
    FlowKL,
    FlowKLFull,
    FlowCDE,
    FlowDecoder,
    FlowPosterior,
    FlowExternal,
)
# from .nnet.mlp import GatedMLP, DecodeMLP


# def construct_encoder_decoder(
#     input_size,
#     output_size,
#     latent_size,
#     hidden_size=512,
#     channels=32,
#     n_layers=6,
#     n_mlp=2,
#     type_ae="ae",
#     type_mod="gated_mlp",
#     config=None,
# ):
#     """ Construct encoder and decoder layers for AE models """
#     if type_mod in ["mlp", "gated_mlp"]:
#         type_ed = (type_mod == "mlp") and "normal" or "gated"
#         encoder = GatedMLP(np.prod(input_size), output_size, hidden_size, n_layers, type_ed)
#         decoder = DecodeMLP(latent_size, input_size, hidden_size, n_layers, type_ed)
#     else:
#         # TODO: LSTM, RTRBM, ESN
#         pass
#     return encoder, decoder


def construct_flow(flow_dim, flow_type="maf", flow_length=16, amortization="input"):
    """ Construct normalizing flow """
    flow_constructor = {
        "planar": [PlanarFlow],
        "sylvester": [TriangularSylvesterFlow, BatchNormFlow, ShuffleFlow],
        "real_nvp": [MaskedCouplingFlow, BatchNormFlow, ShuffleFlow],
        "maf": [MAFlow, BatchNormFlow, ReverseFlow],
        "iaf": [IAFlow, BatchNormFlow, ShuffleFlow],
        "dsf": [DeepSigmoidFlow, BatchNormFlow, ReverseFlow],
        "ddsf": [DeepDenseSigmoidFlow, BatchNormFlow, ReverseFlow],
        "ddsf_iaf": [DDSF_IAFlow, BatchNormFlow, ShuffleFlow],
        "iaf_ctx": [ContextIAFlow, BatchNormFlow, ShuffleFlow],
        "maf_ctx": [ContextMAFlow, BatchNormFlow, ReverseFlow],
    }
    try:
        blocks = flow_constructor[flow_type]
    except KeyError:
        default = "maf"
        print(f"invalid flow type {flow_type}; using default: {default}")
        blocks = flow_constructor[default]
    flow = NormalizingFlow(
        dim=flow_dim,
        blocks=blocks,
        flow_length=flow_length,
        density=MultivariateNormal(torch.zeros(flow_dim), torch.eye(flow_dim)),
        amortized="self",
    )
    return flow, blocks


def construct_disentangle(
    in_dims, model_type="density", semantic_dim=0, n_layers=4, flow_type="maf"
):
    """ Construct DisentanglingFlow """
    _, blocks = construct_flow(
        in_dims, flow_type=flow_type, flow_length=1, amortization="self"
    )
    eps_var_dict = {"density": 1, "base": 1, "full": 1e-1}
    return DisentanglingFlow(
        in_dims,
        blocks=blocks,
        flow_length=n_layers,
        amortize="self",
        eps_var=eps_var_dict[model_type],
        var_type="dims_rand",
    )


def construct_regressor(
    in_dims,
    out_dims,
    model_type="mlp",
    hidden_dims=0,
    n_layers=16,
    flow_type="maf",
    amortize="self",
    eps_var=1e-2,
    var_type="dims",
):
    """ Construct Regressor Model """
    if hidden_dims == 0:
        hidden_dims = in_dims * 4
    if model_type == "mlp":  # MLP Regressor
        regression_model = nn.Sequential()
        for layer in range(n_layers):
            in_s = (layer == 0) and in_dims or hidden_dims
            out_s = (layer == (n_layers - 1)) and out_dims or hidden_dims
            regression_model.add_module("l%d" % layer, nn.Linear(in_s, out_s))
            if layer < (n_layers - 1):
                regression_model.add_module("b%d" % layer, nn.BatchNorm1d(out_s))
                regression_model.add_module("r%d" % layer, nn.ReLU())
                regression_model.add_module("d%d" % layer, nn.Dropout(p=0.3))
            else:
                regression_model.add_module("h", nn.Sigmoid())
    elif model_type == "bnn":  # Bayesian regressor
        _, blocks = construct_flow(
            in_dims, flow_type=flow_type, flow_length=1, amortization=amortize
        )
        regression_model = BayesianRegressor(
            in_dims,
            out_dims,
            hidden_size=hidden_dims,
            n_layers=n_layers,
            blocks=blocks,
            flow_length=n_layers,
        )
    elif model_type[:4] == "flow":  # Flow mixture prediction
        _, blocks = construct_flow(
            in_dims, flow_type=flow_type, flow_length=1, amortization=amortize
        )
        flows_constructor = {
            "flow_p": FlowTransform(
                in_dims,
                blocks=blocks,
                flow_length=n_layers,
                amortize=amortize,
                eps_var=eps_var,
                var_type=var_type,
            ),
            "flow_trans": FlowTransform(
                in_dims,
                blocks=blocks,
                flow_length=n_layers,
                amortize=amortize,
                eps_var=eps_var,
                var_type=var_type,
            ),
            "flow_m": FlowKL(
                in_dims,
                blocks=blocks,
                flow_length=n_layers,
                amortize=amortize,
                eps_var=eps_var,
                var_type=var_type,
            ),
            "flow_kl": FlowKL(
                in_dims,
                blocks=blocks,
                flow_length=n_layers,
                amortize=amortize,
                eps_var=eps_var,
                var_type=var_type,
            ),
            "flow_kl_f": FlowKLFull(
                in_dims,
                blocks=blocks,
                flow_length=n_layers,
                amortize=amortize,
                eps_var=eps_var,
                var_type=var_type,
            ),
            "flow_cde": FlowCDE(
                in_dims,
                blocks=blocks,
                flow_length=n_layers,
                amortize=amortize,
                eps_var=eps_var,
                var_type=var_type,
            ),
            "flow_ext": FlowExternal(
                in_dims,
                blocks=blocks,
                flow_length=n_layers,
                amortize=amortize,
                eps_var=eps_var,
                var_type=var_type,
            ),
            "flow_post": FlowPosterior(
                in_dims,
                blocks=blocks,
                flow_length=n_layers,
                amortize=amortize,
                eps_var=eps_var,
                var_type=var_type,
            ),
            "flow_dec": FlowDecoder(
                in_dims,
                blocks=blocks,
                flow_length=n_layers,
                amortize=amortize,
                eps_var=eps_var,
                var_type=var_type,
            ),
        }
        try:
            regression_model = flows_constructor[model_type]
        except KeyError as e:
            traceback.format_exc(f"invalid model type {model_type}")
            raise e
    else:
        raise ValueError(f"Invalid regressor choice: {model_type}")
    return regression_model
