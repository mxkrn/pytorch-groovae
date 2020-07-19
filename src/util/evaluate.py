import torch


def evaluate_epoch(model, loader, loss_params, args):
    model.eval()
    full_loss = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(args.device, non_blocking=True)
            hidden = model.initHidden()
            recon_x, z_loss, mu, log_var = model(x, hidden)

            # Reconstruction loss
            rec_loss = model.recons_loss(recon_x, x)

            # TODO: Regression loss

            # Final loss
            b_loss = rec_loss + (
                args.beta * z_loss
            )
            #  + (args.gamma * reg_loss)).mean(dim=0)

            full_loss += b_loss
        full_loss /= len(loader)
    return full_loss
