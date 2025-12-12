# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-OSS Model Implementation in JAX/Flax NNX.

This implementation matches the PyTorch reference in gpt-oss/gpt_oss/torch/model.py.
The model processes sequences as [n_tokens, hidden_size] internally (unbatched).
"""

import dataclasses
import math
from functools import partial
from typing import Tuple

import jax
from flax import nnx
from jax import P
from jax import numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh, reshard
from jaxtyping import Array, ArrayLike

ShardingSpec = PartitionSpec


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingCfg:
    emb_vd: ShardingSpec
    emb_dv: ShardingSpec
    qkv_weight_dqkv: ShardingSpec
    o_weight_dhd: ShardingSpec
    gate_weight_de: ShardingSpec
    mlp1_weight_edh: ShardingSpec
    mlp2_weight_edh: ShardingSpec
    rms_norm: ShardingSpec
    act_btd: ShardingSpec
    act_btnh: ShardingSpec
    sinks: ShardingSpec

    @staticmethod
    def no_sharding():
        """Configuration with no sharding (all None)."""
        return ShardingCfg(
            emb_vd=P(None, None),
            emb_dv=P(None, None),
            qkv_weight_dqkv=P(None, None),
            o_weight_dhd=P(None, None, None),
            gate_weight_de=P(None, None),
            mlp1_weight_edh=P(None, None, None),
            mlp2_weight_edh=P(None, None, None),
            rms_norm=P(None),
            act_btd=P(None, None, None),
            act_btnh=P(None, None, None, None),
            sinks=P(None),
        )

    @staticmethod
    def default():
        return ShardingCfg(
            emb_vd=P("tp", "fsdp"),
            emb_dv=P("fsdp", "tp"),
            qkv_weight_dqkv=P("fsdp", "tp"),
            o_weight_dhd=P("tp", None, "fsdp"),
            gate_weight_de=P("fsdp", "tp"),
            mlp1_weight_edh=P("fsdp", "tp", None),
            mlp2_weight_edh=P("fsdp", "tp", None),
            rms_norm=P("tp"),
            act_btd=P("fsdp", None, "tp"),
            act_btnh=P("fsdp", None, "tp", None),
            sinks=P("tp"),
        )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    num_hidden_layers: int
    num_experts: int
    experts_per_token: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    swiglu_limit: float
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    sliding_window: int
    initial_context_length: int
    rope_theta: float
    rope_scaling_factor: float
    rope_ntk_alpha: float
    rope_ntk_beta: float
    shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

    @classmethod
    def _from_param(cls, use_sharding: bool, **kwargs):
        if use_sharding:
            kwargs["shd_cfg"] = ShardingCfg.default()
        return cls(**kwargs)

    @classmethod
    def default(cls, use_sharding: bool = False):
        """Default OSS model configuration."""
        return cls._from_param(
            use_sharding,
            num_hidden_layers=36,
            num_experts=128,
            experts_per_token=4,
            vocab_size=201088,
            hidden_size=2880,
            intermediate_size=2880,
            swiglu_limit=7.0,
            head_dim=64,
            num_attention_heads=64,
            num_key_value_heads=8,
            sliding_window=128,
            initial_context_length=4096,
            rope_theta=150000.0,
            rope_scaling_factor=32.0,
            rope_ntk_alpha=1.0,
            rope_ntk_beta=32.0,
        )


def shard(x: jnp.ndarray, s):
    # Simplified: no sharding or resharding in the minimal version
    return x


class RMSNorm(nnx.Module):
    """RMS Normalization layer matching PyTorch implementation."""
    
    def __init__(self, num_features: int, cfg: ModelConfig, *, rngs: nnx.Rngs, eps: float = 1e-05):
        self.num_features = num_features
        self.eps = eps
        self.scale = shard(
            nnx.Param(nnx.initializers.ones_init()(rngs.params(), (num_features,))),
            cfg.shd_cfg.rms_norm,
        )

    def __call__(self, x: Array) -> Array:
        assert x.shape[-1] == self.num_features
        dtype = x.dtype
        t = jnp.astype(x, jnp.float32)
        # Match PyTorch: t * rsqrt(mean(t**2) + eps)
        rms = jnp.sqrt(jnp.mean(t**2, axis=-1, keepdims=True) + self.eps)
        return jnp.astype((t / rms) * self.scale.value, dtype)


def _apply_rotary_emb(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary embedding to x.
    
    Args:
        x: Shape [n_tokens, n_heads, head_dim] or [n_tokens, n_heads, q_mult, head_dim]
        cos: Shape [n_tokens, head_dim//2]
        sin: Shape [n_tokens, head_dim//2]
    """
    # Expand dims for broadcasting: cos/sin to [n_tokens, 1, head_dim//2]
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)
    # Split x into two halves along head_dim
    x1, x2 = jnp.split(x, 2, axis=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.concatenate((o1, o2), axis=-1)


class RotaryEmbedding(nnx.Module):
    """Rotary Position Embedding with YaRN scaling."""
    
    def __init__(
        self,
        head_dim: int,
        base: float,
        cfg: ModelConfig,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.head_dim = head_dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta

    def _compute_concentration_and_inv_freq(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim)

        if self.scaling_factor > 1.0:
            concentration = 0.1 * jnp.log(self.scaling_factor) + 1.0  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * jnp.log(self.initial_context_length / (self.ntk_beta * 2 * jnp.pi))
                / jnp.log(self.base)
            )
            high = (
                d_half
                * jnp.log(self.initial_context_length / (self.ntk_alpha * 2 * jnp.pi))
                / jnp.log(self.base)
            )

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (jnp.arange(d_half, dtype=jnp.float32) - low) / (high - low)
            mask = 1 - jnp.clip(ramp, 0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = jnp.arange(num_tokens, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        cos = jnp.cos(freqs) * concentration
        sin = jnp.sin(freqs) * concentration
        return cos, sin

    def __call__(self, query: jnp.ndarray, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply rotary embeddings to query and key.
        
        Args:
            query: Shape [n_tokens, n_kv_heads, q_mult, head_dim]
            key: Shape [n_tokens, n_kv_heads, head_dim]
        """
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        # Apply RoPE to query: [n_tokens, n_kv_heads, q_mult, head_dim]
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        # Apply RoPE to key: [n_tokens, n_kv_heads, head_dim]
        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    S: jnp.ndarray | None,
    sm_scale: float,
    sliding_window: int = 0,
) -> jnp.ndarray:
    """Scaled dot-product attention with sliding window and sink tokens.
    
    Matches PyTorch implementation exactly.
    
    Args:
        Q: [n_tokens, n_heads, q_mult, head_dim]
        K: [n_tokens, n_heads, head_dim]
        V: [n_tokens, n_heads, head_dim]
        S: [num_attention_heads] - sink token weights (= n_heads * q_mult)
        sm_scale: attention scale factor (1/sqrt(head_dim))
        sliding_window: sliding window size (0 means no sliding window)
    
    Returns:
        attention output: [n_tokens, n_heads * q_mult * head_dim]
    """
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)

    # Compute attention in float32 for numerical parity
    Q32 = Q.astype(jnp.float32)
    K32 = K.astype(jnp.float32)
    V32 = V.astype(jnp.float32)

    # Expand K and V to match Q's q_mult dimension
    K32 = jnp.expand_dims(K32, axis=2)
    K32 = jnp.broadcast_to(K32, (n_tokens, n_heads, q_mult, d_head))
    V32 = jnp.expand_dims(V32, axis=2)
    V32 = jnp.broadcast_to(V32, (n_tokens, n_heads, q_mult, d_head))

    # Expand S to match attention shape when provided
    if S is not None:
        S32 = S.astype(jnp.float32).reshape(n_heads, q_mult, 1, 1)
        S32 = jnp.broadcast_to(S32, (n_heads, q_mult, n_tokens, 1))

    # Create causal mask with correct dtype to match Torch behavior
    neg_large = jnp.array(-1e30, dtype=jnp.float32)
    mask = jnp.triu(jnp.full((n_tokens, n_tokens), neg_large, dtype=jnp.float32), k=1)
    if sliding_window > 0:
        mask = mask + jnp.tril(jnp.full((n_tokens, n_tokens), neg_large, dtype=jnp.float32), k=-sliding_window)

    # Compute attention scores: einsum("qhmd,khmd->hmqk", Q, K)
    QK = jnp.einsum("qhmd,khmd->hmqk", Q32, K32)
    QK = QK * jnp.array(sm_scale, dtype=jnp.float32)
    QK = QK + mask[None, None, :, :]

    # Concatenate sink tokens when provided
    if S is not None:
        QK = jnp.concatenate([QK, S32], axis=-1)

    # Softmax over key dimension
    W = jax.nn.softmax(QK, axis=-1)
    if S is not None:
        W = W[..., :-1]  # Remove sink dimension

    # Apply attention weights: einsum("hmqk,khmd->qhmd", W, V)
    attn32 = jnp.einsum("hmqk,khmd->qhmd", W, V32)
    attn = attn32.astype(Q.dtype)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(nnx.Module):
    """Attention block matching PyTorch implementation."""
    
    def __init__(self, config: ModelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        # Sink tokens parameter
        self.sinks = shard(
            nnx.Param(jnp.zeros((config.num_attention_heads,), dtype=jnp.bfloat16)),
            config.shd_cfg.sinks,
        )

        self.norm = RMSNorm(config.hidden_size, config, rngs=rngs)

        # QKV projection: Linear(hidden_size, qkv_dim)
        # This model uses attention_bias=True
        qkv_dim = config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads)
        self.qkv = shard(
            nnx.Linear(config.hidden_size, qkv_dim, use_bias=True, dtype=jnp.bfloat16, rngs=rngs),
            config.shd_cfg.qkv_weight_dqkv,
        )

        # Output projection
        self.out = shard(
            nnx.Linear(
                config.head_dim * config.num_attention_heads,
                config.hidden_size,
                use_bias=True,
                dtype=jnp.bfloat16,
                rngs=rngs,
            ),
            config.shd_cfg.o_weight_dhd,
        )

        self.sm_scale = 1.0 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            config,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        """Forward pass.
        
        Args:
            x: [n_tokens, hidden_size]
        
        Returns:
            [n_tokens, hidden_size]
        """
        t = self.norm(x)
        qkv = self.qkv(t)  # [n_tokens, qkv_dim]

        # Split QKV - matching PyTorch slicing
        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        
        q = qkv[:, :q_dim]  # [n_tokens, num_attention_heads * head_dim]
        k = qkv[:, q_dim:q_dim + kv_dim]  # [n_tokens, num_key_value_heads * head_dim]
        v = qkv[:, q_dim + kv_dim:q_dim + 2 * kv_dim]  # [n_tokens, num_key_value_heads * head_dim]

        # Reshape for attention - matching PyTorch view operations
        q_mult = self.num_attention_heads // self.num_key_value_heads
        q = q.reshape(-1, self.num_key_value_heads, q_mult, self.head_dim)
        k = k.reshape(-1, self.num_key_value_heads, self.head_dim)
        v = v.reshape(-1, self.num_key_value_heads, self.head_dim)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Apply attention
        t = sdpa(q, k, v, self.sinks.value, self.sm_scale, self.sliding_window)
        t = self.out(t)
        t = x + t
        return t


def swiglu(x: jnp.ndarray, alpha: float = 1.702, limit: float = 7.0) -> jnp.ndarray:
    """Swish-Gated Linear Unit activation matching PyTorch.
    
    Args:
        x: input tensor
        alpha: sigmoid scaling factor
        limit: clamp limit
    """
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = jnp.clip(x_glu, a_min=None, a_max=limit)
    x_linear = jnp.clip(x_linear, a_min=-limit, a_max=limit)
    out_glu = x_glu * jax.nn.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class MLPBlock(nnx.Module):
    """MLP block with Mixture of Experts matching PyTorch implementation."""
    
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit

        self.norm = RMSNorm(config.hidden_size, config, rngs=rngs)

        # Gate projection - match Torch (bias=True)
        self.gate = shard(
            nnx.Linear(config.hidden_size, config.num_experts, use_bias=True, dtype=jnp.bfloat16, rngs=rngs),
            config.shd_cfg.gate_weight_de,
        )

        # MLP weights (per expert)
        # mlp1: [num_experts, intermediate_size * 2, hidden_size]
        self.mlp1_weight = shard(
            nnx.Param(
                nnx.initializers.normal()(
                    rngs.params(),
                    (config.num_experts, config.intermediate_size * 2, config.hidden_size),
                    dtype=jnp.bfloat16,
                )
            ),
            config.shd_cfg.mlp1_weight_edh,
        )

        # mlp1 bias: [num_experts, intermediate_size * 2]
        self.mlp1_bias = shard(
            nnx.Param(
                jnp.zeros((config.num_experts, config.intermediate_size * 2), dtype=jnp.bfloat16)
            ),
            config.shd_cfg.mlp1_weight_edh[:2],
        )

        # mlp2: [num_experts, hidden_size, intermediate_size]
        self.mlp2_weight = shard(
            nnx.Param(
                nnx.initializers.normal()(
                    rngs.params(),
                    (config.num_experts, config.hidden_size, config.intermediate_size),
                    dtype=jnp.bfloat16,
                )
            ),
            config.shd_cfg.mlp2_weight_edh,
        )

        # mlp2 bias: [num_experts, hidden_size]
        self.mlp2_bias = shard(
            nnx.Param(jnp.zeros((config.num_experts, config.hidden_size), dtype=jnp.bfloat16)),
            config.shd_cfg.mlp2_weight_edh[:2],
        )

    def __call__(self, x: Array) -> Array:
        """Forward pass.
        
        Args:
            x: [n_tokens, hidden_size] - note: b here means tokens, not batch
        
        Returns:
            [n_tokens, hidden_size]
        """
        t = self.norm(x)
        g = self.gate(t)  # [n_tokens, num_experts]

        # Top-k expert selection
        expert_weights, expert_indices = jax.lax.top_k(g, k=self.experts_per_token)
        # expert_weights: [n_tokens, experts_per_token]
        # expert_indices: [n_tokens, experts_per_token]
        
        # PyTorch: softmax(experts.values, dim=1) - dim=1 is the experts_per_token dimension
        expert_weights = jax.nn.softmax(expert_weights, axis=1)

        # MLP #1
        # Gather expert weights: [n_tokens, experts_per_token, intermediate_size * 2, hidden_size]
        mlp1_weight = self.mlp1_weight.value[expert_indices, ...]
        mlp1_bias = self.mlp1_bias.value[expert_indices, ...]
        
        # einsum("beck,bk->bec", mlp1_weight, t) where b=tokens, e=experts_per_token, c=intermediate*2, k=hidden
        t = jnp.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        # MLP #2
        # Gather expert weights: [n_tokens, experts_per_token, hidden_size, intermediate_size]
        mlp2_weight = self.mlp2_weight.value[expert_indices, ...]
        mlp2_bias = self.mlp2_bias.value[expert_indices, ...]
        
        # einsum("beck,bek->bec", mlp2_weight, t) where c=hidden_size, k=intermediate_size
        t = jnp.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias

        # Weighted sum of experts
        # einsum("bec,be->bc", t, expert_weights)
        t = jnp.einsum("bec,be->bc", t, expert_weights)

        return x + t


class TransformerBlock(nnx.Module):
    """Transformer block combining attention and MLP."""
    
    def __init__(self, config: ModelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, rngs=rngs)
        self.mlp = MLPBlock(config, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = self.attn(x)
        x = self.mlp(x)
        return x

        


class Transformer(nnx.Module):
    """GPT-OSS Transformer model.
    
    Matches PyTorch implementation: processes sequences as [n_tokens, hidden_size].
    For batched inference, input shape [batch, seq_len] is flattened to [batch*seq_len]
    internally, then output is reshaped back to [batch, seq_len, vocab_size].
    """
    
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.embedding = shard(
            nnx.Embed(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                dtype=jnp.bfloat16,
                rngs=rngs,
            ),
            config.shd_cfg.emb_vd,
        )

        self.block = nnx.List(
            [TransformerBlock(config, layer_idx, rngs=rngs) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, config, rngs=rngs)

        self.unembedding = shard(
            nnx.Linear(
                config.hidden_size,
                config.vocab_size,
                use_bias=False,
                dtype=jnp.bfloat16,
                rngs=rngs,
            ),
            config.shd_cfg.emb_dv,
        )

    def __call__(self, x: Array) -> Array:
        """Forward pass.
        
        Args:
            x: Token IDs with shape [batch, seq_len] or [n_tokens]
        
        Returns:
            Logits with shape [batch, seq_len, vocab_size] or [n_tokens, vocab_size]
        """
        # Handle batched input
        input_shape = x.shape
        if len(input_shape) == 2:
            batch_size, seq_len = input_shape
            x = x.reshape(-1)  # Flatten to [batch * seq_len]
        else:
            batch_size = None
        
        # Embedding lookup
        x = self.embedding.embedding.value[x]  # [n_tokens, hidden_size]
        
        # Transformer blocks
        for block in self.block:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # Output projection
        x = self.unembedding(x)  # [n_tokens, vocab_size]
        
        # Reshape back to batched format if needed
        if batch_size is not None:
            x = x.reshape(batch_size, seq_len, -1)  # [batch, seq_len, vocab_size]
        
        return x
