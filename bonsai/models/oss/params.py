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

"""Parameter loading for GPT-OSS model.

Handles conversion from PyTorch/HuggingFace checkpoint format to JAX/Flax NNX format.
Follows the same pattern as qwen3/params.py.
"""

import gc
import math
import re
from enum import Enum

import jax
import jax.numpy as jnp
import safetensors
from etils import epath
from flax import nnx

from bonsai.models.oss import modeling as model_lib


# FP4 lookup table values for MXFP4 dequantization
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _get_mxfp4_tensor(
    blocks: jnp.ndarray,
    scales: jnp.ndarray,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jnp.ndarray:
    """Convert MXFP4 blocks and scales to full precision tensor."""
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

    lut = jnp.array(FP4_VALUES, dtype=dtype)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1).astype(jnp.int32) - 127

    # nibble indices -> int64
    idx_lo = (blocks & 0x0F).astype(jnp.int64)
    idx_hi = (blocks >> 4).astype(jnp.int64)

    # Get FP4 values
    sub_lo = lut[idx_lo]  # [rows_total, B]
    sub_hi = lut[idx_hi]  # [rows_total, B]

    # Interleave low and high nibbles: [rows_total, B*2]
    sub = jnp.empty((rows_total, B * 2), dtype=dtype)
    sub = sub.at[:, 0::2].set(sub_lo)
    sub = sub.at[:, 1::2].set(sub_hi)

    # Apply scaling: ldexp(x, exp) = x * 2^exp
    exp = scales.astype(jnp.float32)
    sub = sub.astype(jnp.float32) * jnp.power(2.0, exp)
    sub = sub.astype(dtype)

    return sub.reshape(*prefix_shape, G, B * 2).reshape(*prefix_shape, G * B * 2)


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    """Get mapping from torch checkpoint keys to JAX model keys with transformations."""
    
    class Transform(Enum):
        """Transformations for model parameters.
        
        Value format: (permute_axes, reshape_dims, reshape_first)
        - permute_axes: tuple of axis order for transpose, or None
        - reshape_dims: tuple of new shape, or None
        - reshape_first: if True, reshape before transpose
        """
        EMBED = None  # No transform
        SCALE = None  # No transform (RMSNorm scale)
        SINKS = None  # No transform (attention sinks)
        LINEAR = ((1, 0), None, False)  # Standard linear: [out, in] -> [in, out]
        MLP_WEIGHT = None  # Expert MLP weights: [experts, dim1, dim2] - no transform
        MLP_BIAS = None  # Expert MLP biases: [experts, dim] - no transform
        BIAS = None  # Standard bias: no transform

    # Mapping of torch_keys -> (nnx_keys, transform)
    # Uses regex patterns with capture groups for layer indices
    return {
        # Embedding
        r"model\.embed_tokens\.weight": ("embedding.embedding", Transform.EMBED),
        
        # Attention - Q, K, V weight projections (will be merged by special handler)
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"block.\1.attn.qkv.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"block.\1.attn.qkv.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"block.\1.attn.qkv.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"block.\1.attn.out.kernel", Transform.LINEAR),
        
        # Attention - Q, K, V bias projections (will be merged by special handler)
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (r"block.\1.attn.qkv.bias", Transform.BIAS),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (r"block.\1.attn.qkv.bias", Transform.BIAS),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (r"block.\1.attn.qkv.bias", Transform.BIAS),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.bias": (r"block.\1.attn.out.bias", Transform.BIAS),
        
        # Attention sinks
        r"model\.layers\.([0-9]+)\.self_attn\.sinks": (r"block.\1.attn.sinks", Transform.SINKS),
        
        # Layer norms
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (r"block.\1.attn.norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (r"block.\1.mlp.norm.scale", Transform.SCALE),
        r"model\.norm\.weight": ("norm.scale", Transform.SCALE),
        
        # MLP router/gate
        r"model\.layers\.([0-9]+)\.mlp\.router\.weight": (r"block.\1.mlp.gate.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.router\.bias": (r"block.\1.mlp.gate.bias", Transform.BIAS),
        
        # MLP expert weights (MXFP4 format - handled specially)
        r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_up_proj_blocks": (r"block.\1.mlp.mlp1_weight", Transform.MLP_WEIGHT),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_up_proj_scales": (r"block.\1.mlp.mlp1_weight", Transform.MLP_WEIGHT),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_up_proj_bias": (r"block.\1.mlp.mlp1_bias", Transform.MLP_BIAS),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj_blocks": (r"block.\1.mlp.mlp2_weight", Transform.MLP_WEIGHT),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj_scales": (r"block.\1.mlp.mlp2_weight", Transform.MLP_WEIGHT),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj_bias": (r"block.\1.mlp.mlp2_bias", Transform.MLP_BIAS),
        
        # Output projection (lm_head / unembedding)
        r"lm_head\.weight": ("unembedding.kernel", Transform.LINEAR),
    }, Transform


def _torch_key_to_jax_key(mapping, source_key):
    """Convert torch checkpoint key to JAX model key."""
    subs = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) == 0:
        return None, None
    if len(subs) > 1:
        # Multiple matches is OK for Q/K/V which all map to same QKV
        pass
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            permute, reshape, reshape_first = transform
            if reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
            if permute:
                tensor = tensor.transpose(permute)
            if not reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}")
        # Only apply sharding if sharding_dict is provided
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)


def _stoi(s):
    """Convert string to int if possible, otherwise return string."""
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_checkpoint(
    checkpoint_path: str, cfg: model_lib.ModelConfig, mesh: jax.sharding.Mesh | None = None
) -> model_lib.Transformer:
    """Load tensors from checkpoint and create an OSS Transformer model.
    
    Handles MXFP4 dequantization for expert weights and QKV concatenation.
    """
    files = list(epath.Path(checkpoint_path).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {checkpoint_path}")

    # Load config.json if available
    import dataclasses
    import json
    config_file = epath.Path(checkpoint_path) / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            json_config = json.load(f)
            # Update cfg with values from config.json
            updates = {}
            if "num_hidden_layers" in json_config:
                updates["num_hidden_layers"] = json_config["num_hidden_layers"]
            if "num_local_experts" in json_config:
                updates["num_experts"] = json_config["num_local_experts"]
            elif "num_experts" in json_config:
                updates["num_experts"] = json_config["num_experts"]
            
            if updates:
                print(f"Updating config from config.json: {updates}")
                cfg = dataclasses.replace(cfg, **updates)

    # Infer num_experts from router weight if not in config
    if config_file.exists():
        with open(config_file) as f:
            json_config = json.load(f)
            if "num_local_experts" not in json_config and "num_experts" not in json_config:
                need_infer = True
            else:
                need_infer = False
    else:
        need_infer = True
    
    if need_infer:
        for f in files:
            with safetensors.safe_open(f, framework="numpy") as sf:
                router_keys = [k for k in sf.keys() if "router" in k and "weight" in k]
                if router_keys:
                    router_tensor = sf.get_tensor(router_keys[0])
                    if len(router_tensor.shape) == 2:
                        inferred_num_experts = int(router_tensor.shape[0])
                        print(f"Inferred num_experts from router: {inferred_num_experts}")
                        cfg = dataclasses.replace(cfg, num_experts=inferred_num_experts)
                        break

    # Create model structure
    print(f"Creating model with num_experts={cfg.num_experts}, num_hidden_layers={cfg.num_hidden_layers}")
    transformer = nnx.eval_shape(lambda: model_lib.Transformer(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(transformer)
    state_dict = abs_state.to_pure_dict()
    
    # Collect state_dict keys
    def get_all_keys(d, prefix=""):
        keys = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                keys.extend(get_all_keys(v, full_key))
            else:
                keys.append(full_key)
        return keys
    all_state_keys = get_all_keys(state_dict)
    
    # Only use sharding if mesh is provided
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping, Transform = _get_key_and_transform_mapping(cfg)
    conversion_errors = []
    
    # Debug: collect all checkpoint keys
    all_checkpoint_keys = set()
    used_checkpoint_keys = set()
    assigned_model_keys = set()
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            all_checkpoint_keys.update(sf.keys())
    # Minimal: no verbose printing
    
    # Collect Q, K, V weights for each layer to merge into QKV
    qkv_weights = {}  # layer_idx -> {'q': tensor, 'k': tensor, 'v': tensor}
    qkv_biases = {}  # layer_idx -> {'q': tensor, 'k': tensor, 'v': tensor}
    
    # Collect MXFP4 tensor pairs
    mxfp4_pairs = {}  # jax_key -> {'blocks': tensor, 'scales': tensor}
    
    # Track successful loads
    loaded_count = 0

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = jnp.array(sf.get_tensor(torch_key))
                
                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    continue
                
                # Handle Q, K, V weights separately - collect them for merging
                qkv_weight_match = re.match(r"model\.layers\.([0-9]+)\.self_attn\.(q|k|v)_proj\.weight", torch_key)
                if qkv_weight_match:
                    layer_idx = int(qkv_weight_match.group(1))
                    proj_type = qkv_weight_match.group(2)
                    if layer_idx not in qkv_weights:
                        qkv_weights[layer_idx] = {}
                    qkv_weights[layer_idx][proj_type] = tensor
                    # Mark that this torch key will be used later
                    used_checkpoint_keys.add(torch_key)
                    continue
                
                # Handle Q, K, V biases separately - collect them for merging
                qkv_bias_match = re.match(r"model\.layers\.([0-9]+)\.self_attn\.(q|k|v)_proj\.bias", torch_key)
                if qkv_bias_match:
                    layer_idx = int(qkv_bias_match.group(1))
                    proj_type = qkv_bias_match.group(2)
                    if layer_idx not in qkv_biases:
                        qkv_biases[layer_idx] = {}
                    qkv_biases[layer_idx][proj_type] = tensor
                    used_checkpoint_keys.add(torch_key)
                    continue
                
                # Handle MXFP4 tensors - collect blocks and scales pairs
                if "_blocks" in torch_key or "_scales" in torch_key:
                    if jax_key not in mxfp4_pairs:
                        mxfp4_pairs[jax_key] = {}
                    if "_blocks" in torch_key:
                        mxfp4_pairs[jax_key]["blocks"] = tensor
                    else:
                        mxfp4_pairs[jax_key]["scales"] = tensor
                    used_checkpoint_keys.add(torch_key)
                    continue
                
                # Standard tensor assignment
                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform.value, sharding)
                    loaded_count += 1
                    assigned_model_keys.add(".".join([str(k) for k in keys]))
                    used_checkpoint_keys.add(torch_key)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()
    
    # Minimal: no verbose printing

    # Merge Q, K, V weights into QKV for each layer
    for layer_idx, qkv_dict in sorted(qkv_weights.items()):
        if 'q' in qkv_dict and 'k' in qkv_dict and 'v' in qkv_dict:
            q, k, v = qkv_dict['q'], qkv_dict['k'], qkv_dict['v']
            # PyTorch: [out_dim, in_dim], concatenate along out_dim
            qkv = jnp.concatenate([q, k, v], axis=0)
            # Transpose for JAX: [in_dim, out_dim]
            qkv = qkv.T
            
            jax_key = f"block.{layer_idx}.attn.qkv.kernel"
            keys = [_stoi(k) for k in jax_key.split(".")]
            try:
                _assign_weights(keys, qkv, state_dict, f"qkv_layer_{layer_idx}", None, sharding)
                assigned_model_keys.add(".".join([str(k) for k in keys]))
            except Exception as e:
                conversion_errors.append(f"Failed to merge QKV weights for layer {layer_idx}: {e}")

    # Merge Q, K, V biases into QKV for each layer
    for layer_idx, bias_dict in sorted(qkv_biases.items()):
        if 'q' in bias_dict and 'k' in bias_dict and 'v' in bias_dict:
            q_bias, k_bias, v_bias = bias_dict['q'], bias_dict['k'], bias_dict['v']
            # Concatenate along bias dim
            qkv_bias = jnp.concatenate([q_bias, k_bias, v_bias], axis=0)
            
            jax_key = f"block.{layer_idx}.attn.qkv.bias"
            keys = [_stoi(k) for k in jax_key.split(".")]
            try:
                _assign_weights(keys, qkv_bias, state_dict, f"qkv_bias_layer_{layer_idx}", None, sharding)
                assigned_model_keys.add(".".join([str(k) for k in keys]))
            except Exception as e:
                conversion_errors.append(f"Failed to merge QKV biases for layer {layer_idx}: {e}")

    # Process MXFP4 tensor pairs
    for jax_key, pair in mxfp4_pairs.items():
        if "blocks" in pair and "scales" in pair:
            tensor = _get_mxfp4_tensor(pair["blocks"], pair["scales"])
            keys = [_stoi(k) for k in jax_key.split(".")]
            try:
                _assign_weights(keys, tensor, state_dict, jax_key, None, sharding)
                assigned_model_keys.add(".".join([str(k) for k in keys]))
            except Exception as e:
                conversion_errors.append(f"Failed to assign MXFP4 '{jax_key}': {e}")

    # Minimal: no summary prints

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    gc.collect()
    return nnx.merge(graph_def, state_dict)
