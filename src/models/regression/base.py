import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)

    def train_epoch(self, loader, loss, optimizer, args):
        self.train()
        full_loss = 0
        for x in loader:
            # Send to device
            x = x.to(args.device)
            optimizer.zero_grad()
            out = self(x)
            b_loss = loss(out)
            b_loss.backward()
            optimizer.step()
            full_loss += b_loss
        full_loss /= len(loader)
        return full_loss

    def eval_epoch(self, loader, loss, args):
        self.eval()
        full_loss = 0
        with torch.no_grad():
            for x in loader:
                x = x.to(args.device)
                out = self(x).data
                full_loss += loss(out)
            full_loss /= len(loader)
        return full_loss
