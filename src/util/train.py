class Train:

    def __init__(self, config, epoch):
        """
        This class is a wrapper that generates that appropriate
        training scheme needed for each model_type
        """
        self.config = self.loss_params(config, epoch)
        self.epoch = self._generator(config.model)

    def _generator(self, model_type):
        models = {
            'vae': self.vae
        }
        return models[model_type]

    def vae(self, model, loader, loss, optimizer):
        full_loss = 0
        model = model.to(self.config.device)
        model.train()

        for batch in loader:
            # Prepare input
            x = batch[0].to(self.config.device, non_blocking=True)
            target = batch[1].to(self.config.device, non_blocking=True)
            hidden = model.encoder.init_hidden().to(self.config.device, non_blocking=True)

            # Forward pass
            y, z_loss, r_loss = model(x, hidden, target)

            # Reconstruction loss
            rec_loss = loss(y, x)

            # Final loss
            b_loss = (rec_loss + self.config.gamma*r_loss + z_loss*self.config.beta).mean(dim=0)

            # Perform backward
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()
            full_loss += b_loss
        full_loss /= len(loader)
        return full_loss

        # Set warm-up values
    def loss_params(self, config, epoch):
        config.beta = config.beta_factor * (float(epoch) / float(max(config.warm_latent, epoch)))
        if epoch >= config.start_regress:
            config.gamma = (float(epoch - config.start_regress) * config.reg_factor) / float(
                max(config.warm_regress, epoch - config.start_regress)
            )
            if config.regressor != "mlp":
                config.gamma *= 1e-1
        else:
            config.gamma = 0
        if epoch >= config.start_disentangle:
            config.delta = (float(epoch - config.start_disentangle)) / float(
                max(config.warm_disentangle, epoch - config.start_disentangle)
            )
        else:
            config.delta = 0
        print(f"{config.beta} - {config.gamma}")
        return config
