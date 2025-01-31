from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.model_memory_size import model_memory_size


@dataclass
class LlamaConfig:
    batch_size: int = 32
    vocab_size: int = 128_256
    context_length: int = (
        131_072  # make smaller if not enough ram but need to rescale theta
    )
    embed_dim: int = 2048  # hidden size in huggingface
    intermediate_dim: int = 8192  # hidden size in huggingface
    num_layers: int = 16
    norm_eps: float = 1e-5
    num_attention_heads: int = 32
    num_key_value_groups: int = 8  # for group query attention
    rope_theta: float = 500_000.0
    rope_scaling: dict[str, float | int] = field(
        default_factory=lambda: {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,  # TODO: irrelevant cause have context_length
        }
    )
    device = None  # TODO: bad practice to have this here


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_groups = config.num_key_value_groups
        self.embed_dim = config.embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.group_size = (
            self.num_heads // self.num_kv_groups
        )  # number of heads per group

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(
            self.embed_dim, self.num_kv_groups * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.num_kv_groups * self.head_dim, bias=False
        )

        self.cache_k = torch.zeros(
            (
                config.batch_size,
                self.num_kv_groups,
                config.context_length,
                self.head_dim,
            )
        ).cuda(device=config.device)

        self.cache_v = torch.zeros(
            (
                config.batch_size,
                self.num_kv_groups,
                config.context_length,
                self.head_dim,
            )
        ).cuda(device=config.device)
        self.o_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

    def forward(self, x, pos_emb, mask, start_pos=0):
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len)
        batch_size, seq_len, embed_dim = x.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        # -1 means num_heads or num_kv_groups => (batch_size, seq_len, num_heads, head_dim) or (batch_size, seq_len, num_kv_groups, head_dim)

        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        queries = self.q_proj(x).view(hidden_shape).transpose(1, 2)

        # (batch_size, seq_len, num_kv_groups * head_dim) -> (batch_size, seq_len, num_kv_groups, head_dim) -> (batch_size, num_kv_groups, seq_len, head_dim)
        keys = self.k_proj(x).view(hidden_shape).transpose(1, 2)
        values = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        # apply rope
        queries = apply_rope(queries, pos_emb)
        keys = apply_rope(keys, pos_emb)

        self.cache_k = self.cache_k.to(keys)
        self.cache_v = self.cache_v.to(values)

        # Update cache with new keys and values
        # (batch_size, num_kv_groups, seq_len, head_dim)
        self.cache_k[:batch_size, :, start_pos : start_pos + seq_len] = keys
        self.cache_v[:batch_size, :, start_pos : start_pos + seq_len] = values

        # Retrieve cached keys and values
        # (batch_size, num_kv_groups, cached_len + seq_len, head_dim)
        keys = self.cache_k[:batch_size, :, : start_pos + seq_len]
        values = self.cache_v[:batch_size, :, : start_pos + seq_len]

        # repeat k/v heads if num_kv_heads < num_heads
        # (batch_size, num_heads, cached_len + seq_len, head_dim)
        keys = repeat_kv(keys, self.group_size)
        values = repeat_kv(values, self.group_size)

        # self attn
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, cached_len + seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, cached_len + seq_len) -> (batch_size, num_heads, seq_len, cached_len + seq_len)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(x)

        # (batch_size, num_heads, seq_len, cached_len + seq_len) @ (batch_size, num_heads, cached_len + seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(scores, values)
        attn_output = (
            attn_output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        attn_output = self.o_proj(attn_output)  # (batch_size, seq_len, embed_dim)
        return attn_output


def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=num_repeats). The hidden states go from (batch,
    num_key_value_heads, seq_len, head_dim) to (batch, num_attention_heads, seq_len, head_dim)
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if num_repeats == 1:
        return hidden_states

    # expand returns a new view(shared mem) not a new tensor
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, num_repeats, seq_len, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * num_repeats, seq_len, head_dim
    )


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.embed_dim, config.intermediate_dim, bias=False
        )  # gated linear unit(GLU)
        self.up_proj = nn.Linear(
            config.embed_dim, config.intermediate_dim, bias=False
        )  # learn interactions between features before reducing the dimensionality back
        self.down_proj = nn.Linear(
            config.intermediate_dim, config.embed_dim, bias=False
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))  # SwiGLU


class LlamaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self._norm(x.float()) * self.weight).type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.embed_dim, eps=config.norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.embed_dim, eps=config.norm_eps
        )

    def forward(self, x, pos_emb, mask, start_pos=0):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, pos_emb, mask, start_pos)
        x = residual + x  # residual connection

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x  # residual connection

        return x


# default rope params
def compute_default_params(config):
    base = config.rope_theta
    head_dim = config.embed_dim // config.num_attention_heads

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(config.device)
            / head_dim
        )
    )  # (head_dim // 2)

    return inv_freq


# uses llama3's rope method
def compute_llama_params(config):
    inv_freq = compute_default_params(config)

    factor = config.rope_scaling["factor"]
    low_freq_factor = config.rope_scaling["low_freq_factor"]
    high_freq_factor = config.rope_scaling["high_freq_factor"]
    old_context_len = config.rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq

    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (1 - smooth_factor) * (
        inv_freq_llama / factor
    ) + smooth_factor * inv_freq_llama
    is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


# rescale theta for different context lengths
def rescale_theta(new_context_len, config: LlamaConfig):
    scaling_factor = new_context_len / config.context_length
    config.context_length = new_context_len
    theta_new = config.rope_theta * scaling_factor
    config.rope_theta = theta_new
    print("new rope theta", config.rope_theta)


# precompute the sin and cos values for the rotary embeddings
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        inv_freq = compute_llama_params(config)  # (head_dim // 2)

        # can recompute hence persistent=False
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # returns the sin and cos values for the rotary embeddings
    @torch.no_grad()
    def forward(
        self, x, position_ids
    ):  # we don't init position here because easier for caching
        batch_size, seq_len = position_ids.shape

        expanded_inv_freq = (
            self.inv_freq[None, :, None].float().expand(batch_size, -1, 1)
        )  # (head_dim // 2) -> (1, head_dim // 2, 1) -> (batch_size, head_dim // 2, 1)

        expanded_position_ids = position_ids[:, None, :].float()
        # (batch_size, seq_len) -> (batch_size, 1, seq_len)

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # disable autocast
            freqs = (
                expanded_inv_freq.float()  # (batch_size, head_dim // 2, 1)
                @ expanded_position_ids.float()  # (batch_size, 1, seq_len) -> (batch_size, head_dim // 2, seq_len)
            ).transpose(
                1, 2
            )  # (batch_size, seq_len, head_dim // 2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (batch_size, seq_len, head_dim)
            cos = emb.cos()  # (batch_size, seq_len, head_dim)
            sin = emb.sin()  # (batch_size, seq_len, head_dim)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]  # (batch_size, num_heads, seq_len, head_dim // 2)
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)  # (batch_size, num_heads, seq_len, head_dim)


def apply_rope(x, position_emb):
    cos, sin = position_emb  # (batch_size, seq_len, head_dim)
    cos = cos.unsqueeze(1)  # (batch_size, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)  # (batch_size, 1, seq_len, head_dim)

    x_embed = (x * cos) + (
        rotate_half(x) * sin
    )  # element-wise multiplication (batch_size, num_heads, seq_len, head_dim)
    return x_embed


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = LlamaRMSNorm(config.embed_dim, eps=config.norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    @torch.inference_mode()
    def forward(self, x, start_pos=0):
        batch_size, seq_len = x.shape
        x = self.embed_tokens(x)  # (batch_size, seq_len, embed_dim)
        position_ids = torch.arange(
            start_pos, start_pos + x.shape[1], device=x.device
        )  # (start_pos, ..., start_pos + seq_len - 1)
        position_ids = position_ids.expand(x.shape[0], -1)  # (batch_size, seq_len)
        position_emb = self.rotary_emb(
            x, position_ids
        )  # get the sin and cos values (batch_size, seq_len, head_dim)

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)

            mask = torch.triu(mask, diagonal=1)
            """ 
            the main diagonal becomes 0 if diagonal=1 otherwise -inf
            e.g. for seq_len = 5
            [[  0., -inf, -inf, -inf, -inf],
            [  0.,   0., -inf, -inf, -inf],
            [  0.,   0.,   0., -inf, -inf],
            [  0.,   0.,   0.,   0., -inf],
            [  0.,   0.,   0.,   0.,   0.]], shape=(5, 5)
            """

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seq_len, cache_len + seq_len), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seq_len, start_pos), device=x.device), mask]
            ).type_as(x)

        for layer in self.layers:
            # (batch_size, seq_len, embed_dim)
            x = layer(x, position_emb, mask, start_pos)

        x = self.norm(x)  # (batch_size, seq_len, embed_dim)
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.params = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # tie weights
        self.model.embed_tokens.weight = self.lm_head.weight

    @torch.inference_mode()
    def forward(self, x, start_pos=0):
        x = self.model(x, start_pos)
        x = self.lm_head(x).float()  # (batch_size, seq_len, vocab_size) logits
        return x

    @classmethod
    def load_pretrained_weights(
        cls,
        model_type="meta-llama/Llama-3.2-1B-Instruct",
        new_context_len=8192,
        token=None,
    ):
        from transformers import LlamaForCausalLM as HuggingfaceLlamaForCausalLM

        # pretrained weights / pretrained model
        pretrained_model = HuggingfaceLlamaForCausalLM.from_pretrained(
            model_type, token=token
        )

        # current model
        config = LlamaConfig()
        rescale_theta(new_context_len, config=config)
        my_model = LlamaForCausalLM(config)

        # check if the model and loaded weights have the same number of keys
        assert len(my_model.state_dict().keys()) == len(
            pretrained_model.state_dict().keys()
        ), "Model and loaded weights have different number of keys"

        for k, v in my_model.state_dict().items():
            # check if the model and loaded weights have the same name
            assert (
                k in pretrained_model.state_dict()
            ), f"Key {k} not found in pretrained model"

            # check if the model and loaded weights have the same shape
            assert (
                v.shape == pretrained_model.state_dict()[k].shape
            ), f"Shape mismatch for {k}"

        # check total number of parameters
        my_total_params = sum(p.numel() for p in my_model.parameters())
        pretrained_total_params = sum(p.numel() for p in pretrained_model.parameters())
        assert (
            my_total_params == pretrained_total_params
        ), f"Total number of parameters mismatch: My model: {my_total_params}, Pretrained model: {pretrained_total_params}"

        # check have the same memory size
        # float32
        assert model_memory_size(
            my_model, input_dtype=torch.float32
        ) == model_memory_size(
            pretrained_model, input_dtype=torch.float32
        ), "Memory size mismatch for float32"

        # bfloat16
        assert model_memory_size(
            my_model, input_dtype=torch.bfloat16
        ) == model_memory_size(
            pretrained_model, input_dtype=torch.bfloat16
        ), "Memory size mismatch for bfloat16"

        # load weights
        for k, v in my_model.state_dict().items():
            with torch.no_grad():
                my_model.state_dict()[k].copy_(pretrained_model.state_dict()[k])
        return my_model

    def get_info(self):
        for k, v in self.model.state_dict().items():
            print(f"Key: {k}, Shape: {v.shape}, dtype: {v.dtype}, Device: {v.device}")
