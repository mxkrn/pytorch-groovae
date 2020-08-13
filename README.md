# RhythmFlow

Generate expressive musical rhythms with real-time stylistic control. 

This repository is split up into two component, `syncopate` and `groove`, which are responsible for generating rhythm variations and expressivity, 
respectively. PyTorch is used for neural network design. This only contains the python code for training and testing the network, to see an application
of the networks exported to ONNX please go to my other repository `M4L-RhythmFlow`. 

## Groove

`Groove` is a basically a PyTorch implementation of GrooVAE (https://github.com/magenta/magenta/tree/master/magenta/models/music_vae) which is a variational autoencoder (VAE) network that expects a "naive" drum sequence and returns to us the most likely groove mapping. The groove mapping consists of microtiming and velocity profiles for each instrument at each timestep.

## Syncopate

`Syncopate` is an autoregressive flow network that learns to map binary drum patterns to a low-dimensional latent space. 
The flow network can be reversed to reconstruct the drum patterns based on the latent variables used. 

