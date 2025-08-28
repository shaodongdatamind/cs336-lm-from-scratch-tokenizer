from regex import M
from torch import nn
from einops import einsum, rearrange, repeat
import torch
import logging


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        # no bias, following the most modern LLMs
        super().__init__()
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... input, output input -> ... output")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.device = device
        self.dtype = dtype

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup the embedding vectors for the given token IDs.
        return self.embedding[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS_a = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = x * RMS_a * self.g
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Position-Wise FFN in the transformer.
        Canonically, d_ff = 8 / 3 * d_model.
        """
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        # initialize weights with truncated normal distribution
        std = (2.0 / (d_ff + d_model)) ** 0.5
        nn.init.trunc_normal_(self.W1, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.W2, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.W3, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        silu_w1_x = torch.sigmoid(w1_x) * w1_x
        w3_x = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        glu_output = silu_w1_x * w3_x
        swiglu_output = einsum(
            glu_output, self.W2, "... d_ff, d_model d_ff -> ... d_model"
        )
        return swiglu_output

class Silu(nn.Module):
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires even head dimension (pairs of features)"
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        self.device = device

        # Precompute frequencies and their sin/cos up to max_seq_len
        dim_index = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = self.theta ** (-dim_index / d_k)  # shape: (d_k/2,)
        positions = torch.arange(
            self.max_seq_len, dtype=torch.float32, device=device
        )  # (max_seq_len,)
        freqs = einsum(
            positions, inv_freq, "max_seq_len, d_k_2 -> max_seq_len d_k_2"
        )  # (max_seq_len, d_k/2)
        cos_cached = freqs.cos()  # (max_seq_len, d_k/2)
        sin_cached = freqs.sin()  # (max_seq_len, d_k/2)

        # Register as non-persistent buffers so state_dict stays compact
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len) with values in [0, max_seq_len)
        assert x.shape[-1] == self.d_k, "Input last dim must equal RoPE d_k"
        assert (
            token_positions.shape[-1] == x.shape[-2]
        ), "token_positions length must match sequence length"

        # Gather cos/sin for each token position across the batch/sequence dims
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2)

        # Ensure dtype/device alignment for mixed-precision
        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)

        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k/2)

        out_even = x_even * cos - x_odd * sin  # (..., seq_len, d_k/2)
        out_odd = x_odd * cos + x_even * sin  # (..., seq_len, d_k/2)

        out = torch.empty_like(x)  # (..., seq_len, d_k)
        out[..., 0::2] = out_even  # (..., seq_len, d_k/2)
        out[..., 1::2] = out_odd  # (..., seq_len, d_k/2)
        return out


def scaled_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Scale softmax to avoid overflow.

    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    x_exp = torch.exp(x)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Given a query, key, and value, return the output of the scaled dot-product attention.

    Args:
        Q (Float[Tensor, "..."]): Query tensor. (batch_size, ..., seq_len, d_k)
        K (Float[Tensor, "..."]): Key tensor. (batch_size, ..., seq_len, d_k)
        V (Float[Tensor, "..."]): Value tensor. (batch_size, ..., seq_len, d_v)
        Mask (Boolean[Tensor, "..."]): Mask tensor. (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    qk_dot = einsum(
        Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k"
    )  # (batch_size, ..., seq_len_q, seq_len_k)
    qk_dot = qk_dot / (d_k**0.5)
    if Mask is not None:
        # adding a -inf in any entry of the mask matrix that is False
        qk_dot = qk_dot.masked_fill(
            ~Mask, -torch.inf
        )  # (batch_size, ..., seq_len_q, seq_len_k)

    qk_dot = scaled_softmax(qk_dot, dim=-1)  # (batch_size, ..., seq_len_q, seq_len_k)
    return einsum(
        qk_dot, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v"
    )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float | None = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        # use self defined Linear layer. no bias. W is initialized with truncated normal distribution.
        self.W_q = Linear(
            d_model, d_model, device=device, dtype=dtype
        )  # query projection
        self.W_k = Linear(
            d_model, d_model, device=device, dtype=dtype
        )  # key projection
        self.W_v = Linear(
            d_model, d_model, device=device, dtype=dtype
        )  # value projection
        self.W_o = Linear(
            d_model, d_model, device=device, dtype=dtype
        )  # output projection
        self.rope = None
        if rope_theta is not None:
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            head_dim = d_model // num_heads
            self.rope = RoPE(
                theta=rope_theta, d_k=head_dim, max_seq_len=max_seq_len, device=device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        # Note Q K V here are of all heads.
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # split into heads: (..., num_heads, seq_len, head_dim)
        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        # Apply RoPE to Q and K if enabled
        if self.rope is not None:
            seq_len = x.shape[-2]
            positions = torch.arange(seq_len, device=x.device)
            # Broadcast positions across batch and heads using einops.repeat
            token_positions = repeat(
                positions, "s -> b h s", b=Q.shape[0], h=self.num_heads
            )
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # causal mask: lower-triangular True keeps
        seq_len = x.shape[-2]
        Mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril(
            diagonal=0
        )

        # attention per head (einsum inside)
        attn_out = scaled_dot_product_attention(
            Q, K, V, Mask=Mask
        )  # (..., H, seq, d_k)

        # merge heads back: (..., seq, d_model)
        attn_out = rearrange(attn_out, "... h seq d -> ... seq (h d)")

        # output projection
        return self.W_o(attn_out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 10000,
        rope_theta: float | None = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm1 = self.norm1(x)
        x = self.mha(x_norm1) + x
        x_norm2 = self.norm2(x)
        x = self.ffn(x_norm2) + x
        return x
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float | None = 10000.0, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length # i.e. max_seq_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta # rope_theta
        self.device = device
        self.dtype = dtype
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, rope_theta=rope_theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Note the output of the transformer is the logits, not the probabilities.
        """
        # x: (batch_size, seq_len)
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        # x: (batch_size, seq_len, d_model)
        x = self.norm(x)
        output_logits = self.lm_head(x)
        # output_logits: (batch_size, seq_len, vocab_size)
        return output_logits