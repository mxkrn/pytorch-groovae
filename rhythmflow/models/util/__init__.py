from .layers import GatedDense
from .loss import ArgMax, multinomial_loss, multinomial_mse_loss
from .regression import (
    BayesianRegressor,
    VariationalLambda,
    FlowPredictor,
    FlowTransform,
    FlowKL,
    FlowKLFull,
    FlowCDE,
    FlowAmortizedPredictor,
    FlowDecoder,
    FlowExternal,
    FlowPosterior,
)

__all__ = [
    BayesianRegressor,
    GatedDense,
    ArgMax,
    multinomial_loss,
    multinomial_mse_loss,
    VariationalLambda,
    FlowPredictor,
    FlowTransform,
    FlowKL,
    FlowKLFull,
    FlowCDE,
    FlowAmortizedPredictor,
    FlowDecoder,
    FlowPosterior,
    FlowExternal,
]
