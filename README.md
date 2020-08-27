# RhythmFlow

Generate musical rhythms with real-time semantic control. To be specific, RhythmFlow a network of normalizing flows that learns to map binary drum patterns to a hidden latent space. The flow network can be reversed to conditionally reconstruct a drum pattern that depends on the latent variables used.

PyTorch is the framework of choice. This only contains the python code for training and testing the network, to see an application of the networks exported to ONNX please go to my other repository `M4L-RhythmFlow`.