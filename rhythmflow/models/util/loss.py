import torch
import torch.nn.functional as F


class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, 1)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        ctx.save_for_backward(idx)
        return idx.float()

    @staticmethod
    def backward(ctx, grad_output):
        (idx,) = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx._input_shape, device=ctx._input_device, dtype=ctx._input_dtype
        )
        grad_input.scatter_(1, idx[:, None], grad_output.sum())
        return grad_input


def multinomial_loss(x_logit, x):
    batch_size = x.shape[0]
    # Reshape input
    x_logit = x_logit.view(batch_size, -1, x.shape[1])
    # Take softmax
    x_logit = F.log_softmax(x_logit, 1)
    # make integer class labels
    target = (x * (x_logit.shape[1] - 1)).long()
    # computes cross entropy over all dimensions separately:
    ce = F.nll_loss(x_logit, target, weight=None, reduction="mean")
    # Return summed loss
    return ce.sum()


def multinomial_mse_loss(x_logit, x):
    b_size = x.shape[0]
    n_out = x.shape[1]
    # Reshape our logits
    x_rep = x_logit.view(b_size, -1, n_out)
    x_multi = x_rep[:, :-1, :].view(b_size, -1)
    # Take the multinomial loss
    multi_loss = multinomial_loss(x_multi, x)
    # Retrieve MSE part of loss
    x_mse = x_rep[:, -1, :]
    # Compute values for MSE
    mse_loss = F.mse_loss(x_mse, x)
    return mse_loss + multi_loss
