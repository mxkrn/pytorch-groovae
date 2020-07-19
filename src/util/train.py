def train_epoch(model, loader, loss_params, optimizer, args):
    model.train()
    full_loss = 0
    for x in loader:
        x = x.to(args.device, non_blocking=True)  # Send to device

        hidden = model.encoder.initHidden()
        recon_x, z_loss, mu, log_var = model(x, hidden)

        # Reconstruction loss
        rec_loss = model.recons_loss(recon_x, x)

        # TODO: Regression loss

        # Final loss
        b_loss = rec_loss + (
            args.beta * z_loss
        )
        #  + (args.gamma * reg_loss)).mean(dim=0)

        # Perform backward
        optimizer.zero_grad()
        b_loss.backward()
        optimizer.step()
        full_loss += b_loss
    full_loss /= len(loader)
    return full_loss
