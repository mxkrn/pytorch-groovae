import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data.loader import load_dataset
from models.base import ModelConstructor
from util import Config, Train, Evaluate, get_logger

# parse and initialize configuration
config = Config()
model_name = config.model_name()
logger = get_logger(config.logging)
logger.debug(f"configuration: {config}")

# intitialize data loader
loader, config = load_dataset(config)

# construct model
model_constructor = ModelConstructor(config)
model = model_constructor._model.to(config.device)
optimizer = model_constructor.optimizer
scheduler = model_constructor.scheduler

# run
start_time = time.time()
losses = torch.zeros(config.epochs, 3)
best_loss = np.inf
early = 0
writer = SummaryWriter('/home/max/repos/rhythmflow/logs/tensorboard',)

logger.info("Starting training...")
for e in range(config.epochs):

    # One epoch of training
    train = Train(config, e)
    train_losses = train.epoch(model, loader['train'], optimizer)
    writer.add_scalars('train loss', train_losses, e)

    evaluate = Evaluate(config)
    valid_losses = evaluate.epoch(model, loader["valid"])
    writer.add_scalars('valid loss', valid_losses, e)

    if (config.model not in ["ae", "vae", "wae", "vae_flow"]) or (
        e >= config.start_regress
    ):
        scheduler.step(valid_losses['full'])  # schedule learning rate

    test_losses = evaluate.epoch(model, loader["test"])  # test evaluation
    writer.add_scalars('test loss', test_losses, e)

    if config.start_regress == 1000:
        losses[e, 1] = losses[e, 0]
        losses[e, 2] = losses[e, 0]

    if valid_losses['full'] < best_loss:  # save model
        if e > config.warm_latent:
            best_loss = valid_losses['full']
            logger.info(f"New best model: {config.output}/models/{model_name}.model")
            torch.save(model, f"/home/max/repos/rhythmflow/{config.output}/models/{model_name}.model")
            early = 0
    elif (
        config.early_stop > 0 and e >= config.start_regress
    ):  # check for early stopping
        early += 1
        if early > config.early_stop:
            logger.info("stopping early")
            break
    logger.debug(f"Epoch {str(e)}")
    logger.info(f"Epoch {str(e)}")
    logger.info(f"Valid loss: {valid_losses}")
    torch.cuda.empty_cache()
writer.flush()
writer.close()
