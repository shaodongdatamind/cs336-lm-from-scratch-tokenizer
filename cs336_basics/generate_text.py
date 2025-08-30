import argparse
from typing import List

import torch

from cs336_basics.bpe import BPETokenizer
from cs336_basics.basic_building_blocks import TransformerLM


@torch.no_grad()
def temperature_softmax(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Numerically stable temperature-scaled softmax over the last dimension.
    """
    if tau is None or tau <= 0:
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    scaled = logits / tau
    m = scaled.max(dim=-1, keepdim=True).values
    exp = torch.exp(scaled - m)
    z = exp.sum(dim=-1, keepdim=True)
    return exp / z


@torch.no_grad()
def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) filtering to a probability distribution over vocabulary.
    Assumes probs is 1D or batched with last dim = vocab size. Returns renormalized probs.
    """

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    if sorted_probs.ndim == 1:
        cutoff = torch.searchsorted(cumulative, torch.tensor(p, device=probs.device), right=True)
        num_keep = int(cutoff.item())
        num_keep = max(1, num_keep)
        keep_idx = sorted_idx[..., :num_keep]
        mask = torch.zeros_like(probs).bool()
        mask[keep_idx] = True
    else:
        batch = probs.shape[0]
        mask = torch.zeros_like(probs).bool()
        p_tensor = torch.full((batch,), p, device=probs.device)
        cutoffs = torch.searchsorted(cumulative, p_tensor[:, None], right=True).squeeze(-1)
        cutoffs = torch.clamp(cutoffs, min=1)
        for b in range(batch):
            keep_idx_b = sorted_idx[b, : cutoffs[b].item()]
            mask[b].scatter_(0, keep_idx_b, True)

    filtered = torch.where(mask, probs, torch.zeros_like(probs))
    denom = filtered.sum(dim=-1, keepdim=True)
    denom = torch.clamp(denom, min=1e-12)
    return filtered / denom


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
) -> List[int]:
    model.eval()
    ids: List[int] = tokenizer.encode(prompt)

    eot_id = tokenizer.bytes_to_id.get(b"<|endoftext|>", None)

    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.context_length :]
        logits = model(x_cond)
        next_logits = logits[:, -1, :]

        probs = temperature_softmax(next_logits, temperature)
        probs = top_p_filter(probs, top_p)

        next_token = torch.multinomial(probs, num_samples=1)
        next_id = int(next_token.item())

        x = torch.cat([x, next_token], dim=1)

        if eot_id is not None and next_id == eot_id:
            break

    return x[0].tolist()


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained Transformer LM")
    parser.add_argument("--bpe_vocab", type=str, required=True, help="Path to BPE vocab .pkl")
    parser.add_argument("--bpe_merges", type=str, required=True, help="Path to BPE merges .pkl")
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--prompt_file", type=str, default=None, help="Optional file containing prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    tokenizer = BPETokenizer.from_files(args.bpe_vocab, args.bpe_merges, ["<|endoftext|>"])

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

    obj = torch.load(args.checkpoint, map_location=device)
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    model.to(device)

    prompt_text = args.prompt
    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8", errors="ignore") as f:
            prompt_text = f.read()

    token_ids = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    text = tokenizer.decode(token_ids)
    print(text)


if __name__ == "__main__":
    main()