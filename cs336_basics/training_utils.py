import torch
import numpy as np
import os
from typing import BinaryIO, IO
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
        for p in group["params"]:
            if p.grad is None:
                continue
        state = self.state[p] # Get state associated with p.
        t = state.get("t", 0) # Get iteration number from the state, or initial value.
        grad = p.grad.data # Get the gradient of loss with respect to p.
        p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
        state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        AdamW optimizer.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                if "step" not in state or "m" not in state or "v" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                m = state["m"]
                v = state["v"]

                # Exponential moving averages
                ## First moment
                new_m = beta1 * m + (1.0 - beta1) * grad
                ## Second moment
                new_v = beta2 * v + (1.0 - beta2) * (grad * grad)

                # Compute adjusted learning rate for iteration t
                state["step"] = state["step"] + 1
                step = state["step"]
                adjusted_lr = lr * math.sqrt(1.0 - (beta2 ** step)) / (1.0 - (beta1 ** step))

                # Update parameters
                denom = torch.sqrt(new_v) + eps
                updated_param = param.data - adjusted_lr * (new_m / denom)

                # Weight decay
                if weight_decay != 0:
                    updated_param = updated_param - lr * weight_decay * updated_param

                param.data = updated_param
                state["m"] = new_m
                state["v"] = new_v

        return loss


def cross_entropy(predicted_logits: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for logits and target indices in a numerically stable way.

    Let o be the logits and y be target indices. For each example,
    loss = -log_softmax(o)[y] = logsumexp(o) - o_y.

    Requirements handled:
    - Subtract the largest element for numerical stability (max-shift).
    - Cancel out log/exp in the numerator by using o_y directly (no exp/log for numerator).
    - Support arbitrary batch-like leading dimensions; reduce by mean over all of them.

    Args:
        predicted_logits: Tensor with shape (..., vocab_size), batch-like dims first.
        target_indices: Tensor with shape (...,), matching the batch-like dims of logits.

    Returns:
        Scalar tensor: average cross-entropy over all batch-like elements.
    """
    vocab_dim = -1

    # Max-shift for numerical stability; keep vocab dim for clearer broadcasting
    m = predicted_logits.max(dim=vocab_dim, keepdim=True).values
    shifted_logits = predicted_logits - m  # (..., V)

    # Denominator via stable log-sum-exp
    # logsumexp(o) = m + log(sum(exp(o - m)))
    sum_exp = torch.exp(shifted_logits).sum(dim=vocab_dim, keepdim=True)
    logsumexp = m + torch.log(sum_exp)

    o_y = predicted_logits.gather(dim=vocab_dim, index=target_indices.unsqueeze(vocab_dim))

    # Final loss per example: m + log(sum exp(o - m)) - o_y
    loss = logsumexp - o_y
    return loss.mean()


def cosine_annealing_schedule(t, lr_max, lr_min, Tw, Tc):
    """
    Cosine annealing schedule.
    t: current iteration
    lr_max: maximum learning rate
    lr_min: minimum learning rate
    Tw: warmup period
    Tc: cosine annealing period
    """
    if t < Tw:
        lr_t = t / Tw * lr_max
    elif t <= Tc:
        lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - Tw) / (Tc - Tw)))
    else:
        lr_t = lr_min
    return lr_t

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """
    Gradient clipping.
    parameters: collection of trainable parameters
    max_l2_norm: maximum l2-norm of the gradient
    """
    all_grads = [param.grad.flatten() for param in parameters if param.grad is not None]
    all_grads = torch.cat(all_grads, dim=0)
    grad_norm = all_grads.norm(p=2)
        
    for param in parameters:
        if param.grad is None:
            continue
        if grad_norm > max_l2_norm:
            clip_coef = max_l2_norm / (grad_norm + 1e-6)
            param.grad = param.grad * clip_coef



def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling batches from a 1D numpy array of token ids.

    Returns two LongTensors (inputs, targets) of shape (batch_size, context_length)
    placed on the requested device. Targets are inputs shifted by one token.
    """
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D numpy array of token ids")
    n = dataset.shape[0]
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n <= context_length:
        raise ValueError("dataset too short for given context_length")

    # Valid starting indices: [0, n - context_length)
    starts_t = torch.randint(0, n - context_length, (batch_size,), dtype=torch.long)
    starts = starts_t.cpu().numpy().astype(np.int64)

    # Vectorized slicing using broadcasting
    arange_ctx = np.arange(context_length, dtype=np.int64)[None, :]  # (1, m)
    x_np = dataset[starts[:, None] + arange_ctx]  # (B, m)
    y_np = dataset[starts[:, None] + arange_ctx + 1]  # (B, m)

    # Move to device as LongTensors (raises on invalid device)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    """
    Serialize model/optimizer state and iteration to a file path or file-like.
    """
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load checkpoint, restore model/optimizer state, and return iteration.
    """
    obj = torch.load(src)
    model.load_state_dict(obj["model"]) 
    optimizer.load_state_dict(obj["optimizer"]) 
    return int(obj["iteration"]) 


