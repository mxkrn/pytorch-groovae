import time
import numpy as np
import torch

from data.loader import load_dataset
from models.base import ModelConstructor
from util.config import Config
from util.train import Train
from util.evaluate import Evaluate

# parse and initialize configuration
config = Config()
model_name = config.model_name()

# intitialize data loader
loader, config = load_dataset(config)


# construct model
model_constructor = ModelConstructor(config)
model = model_constructor._model.to(config.device)
optimizer = model_constructor.optimizer
scheduler = model_constructor.scheduler
reconstruction_loss = model_constructor.rec_loss

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
for e in range(config.epochs):
    # if config.start_regress == 0:  # print a summary of all objects
    #     from pympler import muppy, summary

    #     all_objects = muppy.get_objects()
    #     sum1 = summary.summarize(all_objects)
    #     print(f"************ Summary (Epoch {str(i)}) ************")
    #     summary.print_(sum1)

    # One epoch of training
    train = Train(config, e)
    losses[e, 0] = train.epoch(model, loader['train'], reconstruction_loss, optimizer)

    evaluate = Evaluate(config)
    losses[e, 1] = evaluate.epoch(model, loader["valid"], reconstruction_loss)

    if (config.model not in ["ae", "vae", "wae", "vae_flow"]) or (
        e >= config.start_regress
    ):
        scheduler.step(losses[e, 1])  # schedule learning rate

    losses[e, 2] = evaluate.epoch(model, loader["test"], reconstruction_loss)  # test evaluation

    if config.start_regress == 1000:
        losses[e, 1] = losses[e, 0]
        losses[e, 2] = losses[e, 0]

    if losses[e, 1] < best_loss:  # save model
        best_loss = losses[e, 1]
        print(f"new best model: {config.output}/models/{model_name}.model")
        torch.save(model, f"/home/max/repos/rhythmflow/{config.output}/models/{model_name}.model")
        early = 0
    elif (
        config.early_stop > 0 and e >= config.start_regress
    ):  # check for early stopping
        early += 1
        if early > config.early_stop:
            print("stopping early")
            break
    print(f"Epoch {str(e)}")
    print(f"Loss: {losses[e]}")
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
