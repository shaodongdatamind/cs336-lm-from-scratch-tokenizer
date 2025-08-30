import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
import wandb
import math

from cs336_basics.bpe import BPETokenizer, tokenize_file_to_npy
from cs336_basics.basic_building_blocks import TransformerLM
from cs336_basics.training_utils import (
    AdamW,
    cross_entropy,
    cosine_annealing_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)

torch.set_float32_matmul_precision('high')

def _load_token_array(path: str, dtype: str = "uint16") -> np.ndarray:
    """
    Load a 1D token array efficiently using memory mapping.
    - If path ends with .npy, uses np.load(..., mmap_mode='r')
    - Otherwise, assumes a raw binary file of the given dtype and infers length
    """
    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 1:
            raise ValueError("Expected 1D token array in .npy file")
        return arr
    # raw binary with known dtype, 1D
    dt = np.dtype(dtype)
    num_bytes = os.path.getsize(path)
    if num_bytes % dt.itemsize != 0:
        raise ValueError("File size not divisible by dtype size; cannot infer length")
    length = num_bytes // dt.itemsize
    return np.memmap(path, dtype=dt, mode="r", shape=(length,))


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    # Data (token files)
    parser.add_argument("--train_tokens", type=str, required=True, help="Path to training tokens (.npy or raw tokens bin)")
    parser.add_argument("--val_tokens", type=str, required=True, help="Path to validation tokens (.npy or raw tokens bin)")
    parser.add_argument("--data_dtype", type=str, default="uint16", help="dtype for raw binary if not .npy")
    parser.add_argument("--bpe_vocab", type=str, default=None, help="Path to BPE vocab .pkl (for regeneration)")
    parser.add_argument("--bpe_merges", type=str, default=None, help="Path to BPE merges .pkl (for regeneration)")
    parser.add_argument("--train_text", type=str, default=None, help="Optional raw text file to tokenize into --train_tokens")
    parser.add_argument("--val_text", type=str, default=None, help="Optional raw text file to tokenize into --val_tokens")
    parser.add_argument("--force_regen", action="store_true", help="Regenerate .npy even if it exists")

    # Model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1365, help="~8/3 * d_model")
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimizer & schedule
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000)
    parser.add_argument("--lr_min", type=float, default=1e-5)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=20000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="lm-from-scratch")
    parser.add_argument("--wandb_run", type=str, default=None)


    # Checkpointing
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=0, help="0 disables periodic saves; still saves on exit if path set")
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # W&B setup (optional)
    use_wandb = False
    if args.wandb:
        init_kwargs = {"project": args.wandb_project}
        if args.wandb_run:
            init_kwargs["name"] = args.wandb_run
        cfg = vars(args).copy()
        wandb.init(**init_kwargs, config=cfg)
        use_wandb = True

    # Data
    # Optionally regenerate token arrays from raw text using BPE
    if args.train_text is not None:
        if args.bpe_vocab is None or args.bpe_merges is None:
            raise ValueError("--bpe_vocab and --bpe_merges are required when using --train_text")
        
        if args.force_regen or (not os.path.exists(args.train_tokens)):
            os.makedirs(os.path.dirname(args.train_tokens) or ".", exist_ok=True)
            tok = BPETokenizer.from_files(args.bpe_vocab, args.bpe_merges, ["<|endoftext|>"])
            print(f"[data] generating tokens: {args.train_text} -> {args.train_tokens}")
            tokenize_file_to_npy(tok, args.train_text, args.train_tokens, dtype=args.data_dtype)

    if args.val_text is not None and args.val_tokens is not None:
        if args.bpe_vocab is None or args.bpe_merges is None:
            raise ValueError("--bpe_vocab and --bpe_merges are required when using --val_text")
        if args.force_regen or (not os.path.exists(args.val_tokens)):
            os.makedirs(os.path.dirname(args.val_tokens) or ".", exist_ok=True)
            tok = BPETokenizer.from_files(args.bpe_vocab, args.bpe_merges, ["<|endoftext|>"])
            print(f"[data] generating val tokens: {args.val_text} -> {args.val_tokens}")
            tokenize_file_to_npy(tok, args.val_text, args.val_tokens, dtype=args.data_dtype)

    train_tokens = _load_token_array(args.train_tokens, dtype=args.data_dtype)
    val_tokens = _load_token_array(args.val_tokens, dtype=args.data_dtype) if args.val_tokens else None

    # Model
    device = args.device
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(device),
        dtype=None,
    )
    model = torch.compile(model)
    model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    # Training loop
    model.train()
    t0 = time.time()
    try:
        for it in range(start_iter, args.max_iters):
            # LR schedule
            lr_t = cosine_annealing_schedule(
                t=it,
                lr_max=args.lr,
                lr_min=args.lr_min,
                Tw=args.warmup_iters,
                Tc=args.cosine_cycle_iters,
            )
            _set_optimizer_lr(optimizer, lr_t)

            # Batch
            x, y = get_batch(train_tokens, args.batch_size, args.context_length, device)

            # Forward
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()

            # Logging
            if (it + 1) % args.log_interval == 0:
                elapsed = time.time() - t0
                print(f"iter {it+1} | lr {lr_t:.6g} | loss {float(loss.item()):.4f} | {elapsed:.2f}s")
                if use_wandb:
                    wandb.log({
                        "iter": it + 1,
                        "lr": float(lr_t),
                        "train_loss": float(loss.item()),
                    }, step=it + 1)
                t0 = time.time()

            # Eval
            if val_tokens is not None and (it + 1) % args.eval_interval == 0:
                val_loss = _evaluate(
                    model=model,
                    dataset=val_tokens,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    device=device,
                    num_batches=args.eval_batches,
                )
                val_ppl = math.exp(val_loss)
                print(f"eval @ iter {it+1}: val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
                if use_wandb:
                    wandb.log({
                        "iter": it + 1,
                        "val_loss": float(val_loss),
                        "val_ppl": float(val_ppl),
                    }, step=it + 1)

            # Checkpoint
            if args.checkpoint_path and args.save_every and (it + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, it + 1, args.checkpoint_path)

    finally:
        # Final checkpoint on exit if path is set
        if args.checkpoint_path is not None:
            save_checkpoint(model, optimizer, min(args.max_iters, it + 1), args.checkpoint_path)
        if 'use_wandb' in locals() and use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()


