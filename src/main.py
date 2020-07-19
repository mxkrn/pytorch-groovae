import time
import numpy as np
import torch

from data.loader import load_dataset
from models.base import ModelConstructor
from util.config import Config
from util.train import train_epoch
from util.evaluate import evaluate_epoch

# parse and initialize configuration
config = Config("train")
model_name = config.model_name()
base_dir = f"{config.output}/"
base_audio = f"{config.output}/audio/{model_name}"

# intitialize data loader
loader, config = load_dataset(config)

# construct model
model_constructor = ModelConstructor(config)
model = model_constructor._model.to(config.device)
optimizer = model_constructor.optimizer
scheduler = model_constructor.scheduler
loss = model_constructor.loss

"""
-------------------------------------
TRAINING
-------------------------------------
"""
start_time = time.time()
losses = torch.zeros(config.epochs, 3)
if config.epochs == 0:
    losses = torch.zeros(200, 3)
best_loss = np.inf
early = 0

print("Starting training...")
for i in range(config.epochs):
    # if config.start_regress == 0:  # print a summary of all objects
    #     from pympler import muppy, summary

    #     all_objects = muppy.get_objects()
    #     sum1 = summary.summarize(all_objects)
    #     print(f"************ Summary (Epoch {str(i)}) ************")
    #     summary.print_(sum1)

    # Set warm-up values
    # config.beta = config.beta_factor * (float(i) / float(max(config.warm_latent, i)))
    # if i >= config.start_regress:
    #     config.gamma = (float(i - config.start_regress) * config.reg_factor) / float(
    #         max(config.warm_regress, i - config.start_regress)
    #     )
    #     if config.regressor != "mlp":
    #         config.gamma *= 1e-1
    # else:
    #     config.gamma = 0
    # if i >= config.start_disentangle:
    #     config.delta = (float(i - config.start_disentangle)) / float(
    #         max(config.warm_disentangle, i - config.start_disentangle)
    #     )
    # else:
    #     config.delta = 0
    # print(f"{config.beta} - {config.gamma}")

    # One epoch of training
    losses[i, 0] = train_epoch(model, loader["train"], loss, optimizer, config)
    losses[i, 1] = evaluate_epoch(model, loader["valid"], loss, config)
    if (config.model not in ["ae", "vae", "wae", "vae_flow"]) or (
        i >= config.start_regress
    ):
        scheduler.step(losses[i, 1])  # schedule learning rate

    losses[i, 2] = model.eval_epoch(loader["test"], loss, config)  # test evaluation
    if config.start_regress == 1000:
        losses[i, 1] = losses[i, 0]
        losses[i, 2] = losses[i, 0]

    if losses[i, 1] < best_loss:  # save model
        best_loss = losses[i, 1]
        print(f"new best model: {config.output}/models/{model_name}.model")
        torch.save(model, f"{config.output}/models/{model_name}.model")
        early = 0
    elif (
        config.early_stop > 0 and i >= config.start_regress
    ):  # check for early stopping
        early += 1
        if early > config.early_stop:
            print("stopping early")
            break

    if (i + 1) % config.plot_interval == 0 or (
        config.epochs == 1
    ):  # periodic evaluation (or debug model)
        config.plot = "train"
        with torch.no_grad():
            model.eval()
            # TODO: Write model evaluation function
            # evaluate_model(model, fixed_batch, loader['test'], config, train=True, name=base_img + '_batch_' + str(i))

    if (config.time_limit > 0) and (
        ((time.time() - start_time) / 60.0) > config.time_limit
    ):  # time limit for HPC
        print(
            "hit time limit after "
            + str((time.time() - start_time) / 60.0)
            + " minutes."
        )
        print("entering evaluation mode")
        break
    if config.regressor == "flow_kl_f":
        print(torch.cuda.memory_allocated(config.device))
    print(f"Epoch {str(i)}")
    print(f"Loss: {losses[i]}")
    torch.cuda.empty_cache()


# """
# -------------------------------------
# EVALUATE
# -------------------------------------
# """
# args.plot = 'final'
# args.model_name, args.base_img, args.base_audio = model_name, base_img, base_audio
# args.base_model = f'{args.output}/models/{model_name}

# print(f'loading {args.output}/models/{model_name}.model for evaluation...')
# model = torch.load(f'{args.output}/models/{model_name}.model')
# model = model.to(args.device)

# with torch.no_grad(): # memory saver

#     evaluate_params(model, test_loader, args, losses=losses)

#     if (args.model not in ['mlp', 'gated_mlp']):
#         evaluate_reconstruction(model, test_loader, args, train=False) # reconstruction evaluation
#         args = evaluate_latent_space(model, test_loader, args, train=False) # evaluate latent space
#         evaluate_meta_parameters(model, test_loader, args, train=False) # meta-parameter analysis
#         evaluate_latent_neighborhood(model, test_loader, args, train=False) # latent neighborhood analysis
#         evaluate_semantic_parameters(model, test_loader, args, train=False) # semantic parameter analysis

#     # if (parser.args.synthesize): # Synthesis engine (on GPU)
#     #     from synth.synthesize import create_synth
#     #     print('[Synthesis evaluation]')
#     #     args.engine, args.generator, args.param_defaults, args.rev_idx = create_synth(args.dataset)
#     #     evaluate_synthesis(model, test_loader, parser.args, train=False)

#     #     print('[Load set of testing sound (outside Diva)]')
#     #     test_sounds = get_external_sounds(parser.args.test_sounds, test_loader.dataset, parser.args)
#     #     evaluate_projection(model, test_sounds, parser.args, train=False)

#     #     print('[Evaluate vocal sketching dataset]')
#     #     test_sounds = get_external_sounds(parser.args.vocal_sounds, test_loader.dataset, parser.args)
#     #     evaluate_projection(model, test_sounds, parser.args, train=False, type_val='vocal')
