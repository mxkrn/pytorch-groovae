import json
import numpy as np
import torch
import time

from data.base import GrooveDataset


MODEL = 'vae_rnn_mse_1_256_beta1000.08_16.model'
DATADIR = "/mnt/c/Users/maxkr/data/gmd_drumlab_merge/"

dataset = GrooveDataset(datadir=DATADIR)
model = torch.load(f'/home/max/repos/rhythmflow/outputs/models/{MODEL}')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def sample():
    r = np.random.randint(0, len(dataset.loaded_files))
    file = dataset.loaded_files[r]
    return dataset.to_tensor(file)


def format_output(tensor):
    output = tensor.cpu().detach().numpy()[0]
    velocities = json.dumps(output[:, 8:17].tolist())

    def format_offsets(output):
        offsets = output[:, 18:27]
        offsets = np.where(offsets == 0.0, 64.0, offsets)
        return json.dumps(offsets.tolist())
    offsets = format_offsets(output)
    output_dict = {
        'velocities': velocities,
        'offsets': offsets
    }
    return output_dict


s = sample()
st = time.time()
model = model.to(device)
model.eval()
with torch.no_grad():
    hidden = torch.zeros((1, 1, 256), dtype=torch.float).to(device)
    x = torch.zeros((1, 32, 27), dtype=torch.float).to(device)
    target = torch.zeros((1, 32, 27), dtype=torch.float).to(device)
    x[0, :] = torch.tensor(s[0])
    target[0, :] = torch.tensor(s[1])
    _, z_loss, r_loss = model(x, hidden, target)
    _, hidden = model.encoder(x, hidden)
    y, _ = model.sample(x, hidden, target)

output = format_output(y)
print(output)
# print('time taken', time.time() - st)
# print('z_loss:', z_loss)
# print('r_loss:', r_loss)
# print(y.shape)
