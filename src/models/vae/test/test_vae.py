import torch
# See conftest.py for fixtures


def test_initialize_vae(config, vae, batch):
    x = batch[0]
    target = batch[1]
    hidden = vae.encoder.init_hidden().to(config.device)
    result = vae(x, hidden, target)
    assert result[0].size() == torch.Size([8, 32, 27])
    assert result[1] > 0
    assert result[2] > 0
