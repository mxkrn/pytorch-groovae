from torch import autograd
from torch.nn.functional import mse_loss


def train_epoch(model, loader, rec_loss, loss_params, optimizer, args):
    full_loss = 0
    for batch in loader:
        # with autograd.detect_anomaly():
        x = batch[0]
        x = x.to(args.device, non_blocking=True)  # Send to device

        hidden = model.encoder.initHidden().to(args.device, non_blocking=True)
        recon_x, z_loss, mu, log_var = model(x, hidden)

        # Reconstruction loss
        rec_loss = mse_loss(recon_x, x)

        # TODO: Regression loss

        # Final loss
        b_loss = rec_loss + z_loss
        #  + (args.gamma * reg_loss)).mean(dim=0)

        # Perform backward
        optimizer.zero_grad()
        b_loss.backward()
        optimizer.step()
        full_loss += b_loss
    full_loss /= len(loader)
    return full_loss
