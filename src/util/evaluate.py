import torch


class Evaluate:

    def __init__(self, config):
        """
        This class is a wrapper that generates that appropriate
        training scheme needed for each model_type
        """
        self.config = config
        self.epoch = self._generator(config.model)

    def _generator(self, model_type):
        models = {
            'vae': self.vae
        }
        return models[model_type]

    def vae(self, model, loader, loss):
        full_loss = 0
        model.eval()

        with torch.no_grad():
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
                full_loss += b_loss
            full_loss /= len(loader)
        return full_loss
