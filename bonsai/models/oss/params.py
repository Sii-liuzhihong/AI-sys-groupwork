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

import dataclasses
import gc
import json
import math
import re
from enum import Enum
from typing import Tuple

import jax
import jax.numpy as jnp
import safetensors
from etils import epath
from flax import nnx


from bonsai.models.oss import modeling as model_lib

# Bytes per MXFP4 block: 32 FP4 numbers packed in 16 bytes
BYTES_PER_BLOCK = 16

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

# Map the names assumed in this implementation to the checkpoint names.
PARAM_NAME_MAP = {
    f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.mlp1_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp1_weight": (f"block.{n}.mlp.mlp1_weight.blocks", f"block.{n}.mlp.mlp1_weight.scales")
    for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.mlp2_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_weight": (f"block.{n}.mlp.mlp2_weight.blocks", f"block.{n}.mlp.mlp2_weight.scales")
    for n in range(36)
}


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
    class Transform(Enum):
        """Transformations for model parameters"""

        BIAS = None
        LINEAR = ((1, 0), None, False)
        EMBED = None
        QKV = ((1, 0), None, False)  # QKV is concatenated, just transpose
        O = ((1, 0), None, False)
        GATE = ((1, 0), None, False)
        MLP1_WEIGHT = None  # [num_experts, intermediate_size * 2, hidden_size]
        MLP1_BIAS = None  # [num_experts, intermediate_size * 2]
        MLP2_WEIGHT = None  # [num_experts, hidden_size, intermediate_size]
        MLP2_BIAS = None  # [num_experts, hidden_size]
        SCALE = None
        SINKS = None

    # Mapping of torch_keys -> (nnx_keys, transform)
    # Note: The actual checkpoint uses model.layers.X format, not block.X
    return {
        # Embedding
        r"model\.embed_tokens\.weight": ("embedding.embedding", Transform.EMBED),
        r"embedding\.weight": ("embedding.embedding", Transform.EMBED),  # Fallback
        r"embed_tokens\.weight": ("embedding.embedding", Transform.EMBED),  # Fallback
        
        # Attention - QKV are separate in checkpoint, need to concatenate
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"block.\1.attn.qkv.kernel", Transform.QKV),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"block.\1.attn.qkv.kernel", Transform.QKV),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"block.\1.attn.qkv.kernel", Transform.QKV),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"block.\1.attn.out.kernel", Transform.O),
        r"model\.layers\.([0-9]+)\.self_attn\.sinks": (r"block.\1.attn.sinks", Transform.SINKS),
        
        # Norms - to_pure_dict() flattens Param.value, so scale is directly ShapeDtypeStruct
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (r"block.\1.attn.norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (r"block.\1.mlp.norm.scale", Transform.SCALE),
        r"model\.norm\.weight": ("norm.scale", Transform.SCALE),
        
        # MLP - router is the gate, experts contain the weights
        r"model\.layers\.([0-9]+)\.mlp\.router\.weight": (r"block.\1.mlp.gate.kernel", Transform.GATE),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_up_proj_blocks": (
            (r"block.\1.mlp.mlp1_weight.blocks", r"block.\1.mlp.mlp1_weight.scales"),
            Transform.MLP1_WEIGHT,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_up_proj_scales": (
            (r"block.\1.mlp.mlp1_weight.blocks", r"block.\1.mlp.mlp1_weight.scales"),
            Transform.MLP1_WEIGHT,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_up_proj_bias": (r"block.\1.mlp.mlp1_bias", Transform.MLP1_BIAS),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj_blocks": (
            (r"block.\1.mlp.mlp2_weight.blocks", r"block.\1.mlp.mlp2_weight.scales"),
            Transform.MLP2_WEIGHT,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj_scales": (
            (r"block.\1.mlp.mlp2_weight.blocks", r"block.\1.mlp.mlp2_weight.scales"),
            Transform.MLP2_WEIGHT,
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj_bias": (r"block.\1.mlp.mlp2_bias", Transform.MLP2_BIAS),
        
        # Fallback patterns for block.X format (if used)
        r"block\.([0-9]+)\.attn\.qkv\.weight": (r"block.\1.attn.qkv.kernel", Transform.QKV),
        r"block\.([0-9]+)\.attn\.out\.weight": (r"block.\1.attn.out.kernel", Transform.O),
        r"block\.([0-9]+)\.attn\.sinks": (r"block.\1.attn.sinks", Transform.SINKS),
        r"block\.([0-9]+)\.attn\.norm\.scale": (r"block.\1.attn.norm.scale", Transform.SCALE),
        r"block\.([0-9]+)\.mlp\.gate\.weight": (r"block.\1.mlp.gate.kernel", Transform.GATE),
        r"block\.([0-9]+)\.mlp\.mlp1_weight": (
            (r"block.\1.mlp.mlp1_weight.blocks", r"block.\1.mlp.mlp1_weight.scales"),
            Transform.MLP1_WEIGHT,
        ),
        r"block\.([0-9]+)\.mlp\.mlp1_bias": (r"block.\1.mlp.mlp1_bias", Transform.MLP1_BIAS),
        r"block\.([0-9]+)\.mlp\.mlp2_weight": (
            (r"block.\1.mlp.mlp2_weight.blocks", r"block.\1.mlp.mlp2_weight.scales"),
            Transform.MLP2_WEIGHT,
        ),
        r"block\.([0-9]+)\.mlp\.mlp2_bias": (r"block.\1.mlp.mlp2_bias", Transform.MLP2_BIAS),
        r"block\.([0-9]+)\.mlp\.norm\.scale": (r"block.\1.mlp.norm.scale", Transform.SCALE),
        r"norm\.scale": ("norm.scale", Transform.SCALE),
        r"unembedding\.weight": ("unembedding.kernel", Transform.LINEAR),
        r"lm_head\.weight": ("unembedding.kernel", Transform.LINEAR),
    }


def _torch_key_to_jax_key(mapping, source_key):
    """Convert torch checkpoint key to JAX model key."""
    subs = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) == 0:
        # Return None to skip this key (will be handled by caller)
        return None, None
    if len(subs) > 1:
        raise ValueError(f"Multiple matches found for '{source_key}': {subs}")
    return subs[0]


def _stoi(s):
    """Convert string to int if possible, otherwise return string."""
    try:
        return int(s)
    except ValueError:
        return s


def _assign_weights(keys, tensor, state_dict, st_key, transform):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            # transform can be None (for EMBED, BIAS, etc.) or a tuple (permute, reshape, reshape_first)
            if isinstance(transform, tuple):
                permute, reshape, reshape_first = transform
                if reshape_first and reshape is not None:
                    tensor = tensor.reshape(reshape)
                if permute:
                    tensor = tensor.transpose(permute)
                if not reshape_first and reshape is not None:
                    tensor = tensor.reshape(reshape)

        # Handle ShapeDtypeStruct - get shape attribute
        target_shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else None
        if target_shape is None:
            raise ValueError(f"Cannot determine target shape for {st_key} at key {key}, got {type(state_dict[key])}")
        
        if tensor.shape != target_shape:
            raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {target_shape}")
       
        # Convert tensor to the correct dtype if needed
        if hasattr(state_dict[key], 'dtype') and tensor.dtype != state_dict[key].dtype:
            tensor = jnp.astype(tensor, state_dict[key].dtype)
        # Assign the tensor - this should replace the ShapeDtypeStruct with the actual array
        assigned_value = jax.device_put(tensor)
        state_dict[key] = assigned_value
        # Verify assignment worked
        final_type = type(state_dict[key]).__name__
        if final_type == 'ShapeDtypeStruct':
            # This shouldn't happen, but if it does, the assignment didn't work
            # This might be because state_dict[key] is a reference that got overwritten
            pass  # Don't raise here, let it fail later with a clearer error
    else:
        # Handle nested access - state_dict[key] should be a dict/list
        if key not in state_dict:
            raise ValueError(f"Key {key} not found in state_dict for {st_key}")
        
        next_state = state_dict[key]
        # next_state should be a dict or list (from to_pure_dict())
        if not isinstance(next_state, (dict, list)):
            raise ValueError(f"Expected dict or list for nested key {key} in state_dict for {st_key}, got {type(next_state)}")
        
        # Recursively assign - this will modify next_state in place, which modifies state_dict[key]
        _assign_weights(rest, tensor, next_state, st_key, transform)



def create_model_from_checkpoint(
    checkpoint_path: str, cfg: model_lib.ModelConfig,
) -> model_lib.Transformer:
    """Load tensors from checkpoint and create an OSS Transformer model."""
    # Load config if available and update cfg
    config_path = epath.Path(checkpoint_path) / "config.json"
    json_config = {}
    if config_path.exists():
        with config_path.open() as f:
            json_config = json.load(f)
            # Update cfg with loaded config
            if "num_hidden_layers" in json_config:
                cfg = dataclasses.replace(cfg, num_hidden_layers=json_config["num_hidden_layers"])
            if "num_experts" in json_config:
                cfg = dataclasses.replace(cfg, num_experts=json_config["num_experts"])
    
    # Infer num_experts from router weight if not in config
    need_infer_experts = "num_experts" not in json_config

    files = list(epath.Path(checkpoint_path).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {checkpoint_path}")

    # Infer num_experts from router weight if needed (BEFORE creating model structure)
    if need_infer_experts:
        for f in files:
            with safetensors.safe_open(f, framework="numpy") as sf:
                router_keys = [k for k in sf.keys() if "router" in k and "weight" in k]
                if router_keys:
                    # Get shape from tensor directly
                    # Router shape is (hidden_size, num_experts), so num_experts is the second dimension
                    try:
                        # Try to get shape from metadata first (avoids loading bfloat16)
                        try:
                            metadata = sf.metadata()
                            if router_keys[0] in metadata and 'shape' in metadata[router_keys[0]]:
                                router_shape = tuple(metadata[router_keys[0]]['shape'])
                            else:
                                # Fallback: load tensor and get shape
                                import numpy as np
                                router_tensor = np.array(sf.get_tensor(router_keys[0]), dtype=np.float32)
                                router_shape = router_tensor.shape
                        except:
                            # Final fallback: try to get shape directly
                            router_tensor = sf.get_tensor(router_keys[0])
                            router_shape = router_tensor.shape
                        
                        if len(router_shape) == 2:
                            # Router in checkpoint is PyTorch format: (num_experts, hidden_size)
                            # So num_experts is shape[0], not shape[1]
                            inferred_num_experts = int(router_shape[0])
                            print(f"Inferred num_experts from router: {inferred_num_experts} (router shape: {router_shape})")
                            cfg = dataclasses.replace(cfg, num_experts=inferred_num_experts)
                            print(f"Updated cfg.num_experts to: {cfg.num_experts}")
                            break
                    except Exception as e:
                        print(f"Warning: Could not infer num_experts from router: {e}")
                        import traceback
                        traceback.print_exc()
                        pass

    # Create model structure (AFTER inferring num_experts)
    print(f"Creating model with num_experts={cfg.num_experts}, num_hidden_layers={cfg.num_hidden_layers}")
    transformer = nnx.eval_shape(lambda: model_lib.Transformer(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(transformer)
    state_dict = abs_state.to_pure_dict()

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []

    # Track MXFP4 tensors that need special handling
    mxfp4_pairs = {}
    
    # Track Q, K, V weights separately to merge them into QKV
    qkv_weights = {}  # layer_idx -> {'q': tensor, 'k': tensor, 'v': tensor}

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                # Check if this is an MXFP4 tensor
                is_mxfp4 = False
                for pattern, (repl, transform) in key_mapping.items():
                    match = re.match(pattern, torch_key)
                    if match:
                        if isinstance(repl, tuple):  # MXFP4 tensor pair
                            is_mxfp4 = True
                            # Extract the base name by replacing the pattern with the replacement
                            # repl[0] is like "block.\1.mlp.mlp1_weight.blocks", need to substitute \1
                            # But we need to handle both _blocks and _scales cases (note: underscore, not dot)
                            base_name = match.expand(repl[0])  # This gives "block.X.mlp.mlp1_weight.blocks"
                            if torch_key.endswith("_blocks"):
                                # Find the corresponding scales key in checkpoint
                                scales_key = torch_key.replace("_blocks", "_scales")
                                mxfp4_pairs[base_name] = (torch_key, scales_key)
                            elif torch_key.endswith("_scales"):
                                # For _scales, we still use the same base_name (which has .blocks suffix)
                                # This is correct because both _blocks and _scales map to the same JAX key
                                blocks_key = torch_key.replace("_scales", "_blocks")
                                if base_name not in mxfp4_pairs:
                                    mxfp4_pairs[base_name] = (blocks_key, torch_key)
                            break

                if is_mxfp4:
                    continue  # Skip individual MXFP4 tensors, process them in pairs

                tensor = jnp.array(sf.get_tensor(torch_key))

                # Handle Q, K, V separately - collect them first, merge later
                qkv_match = re.match(r"model\.layers\.([0-9]+)\.self_attn\.(q|k|v)_proj\.weight", torch_key)
                if qkv_match:
                    layer_idx = int(qkv_match.group(1))
                    proj_type = qkv_match.group(2)
                    if layer_idx not in qkv_weights:
                        qkv_weights[layer_idx] = {}
                    qkv_weights[layer_idx][proj_type] = tensor
                    continue  # Skip individual assignment, will merge later

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    # Skip keys that don't match any pattern (might be metadata or unused)
                    continue

                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    # Pass transform.value if it's an Enum, otherwise pass transform directly
                    transform_val = transform.value if hasattr(transform, 'value') else transform
                    _assign_weights(keys, tensor, state_dict, torch_key, transform_val)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    # Merge Q, K, V into QKV and assign
    for layer_idx, qkv_dict in qkv_weights.items():
        if 'q' in qkv_dict and 'k' in qkv_dict and 'v' in qkv_dict:
            # Concatenate Q, K, V along the first dimension
            # PyTorch checkpoint format: [out_features, in_features] = [qkv_dim, hidden_size]
            # JAX nnx.Linear format: [out_features, in_features] = [qkv_dim, hidden_size]
            # We need to ensure all are in [qkv_dim, hidden_size] format
            q = qkv_dict['q']  # Could be [q_dim, hidden_size] or [hidden_size, q_dim]
            k = qkv_dict['k']  # Could be [k_dim, hidden_size] or [hidden_size, k_dim]
            v = qkv_dict['v']  # Could be [v_dim, hidden_size] or [hidden_size, v_dim]
            
            # Normalize to [qkv_dim, hidden_size] format
            # If second dim is hidden_size, it's already correct
            # If first dim is hidden_size, need to transpose
            if q.shape[1] == cfg.hidden_size:
                # Already in [qkv_dim, hidden_size] format
                pass
            elif q.shape[0] == cfg.hidden_size:
                q = q.T
            else:
                raise ValueError(f"Unexpected Q shape: {q.shape}")
            
            if k.shape[1] == cfg.hidden_size:
                # Already in [qkv_dim, hidden_size] format
                pass
            elif k.shape[0] == cfg.hidden_size:
                k = k.T
            else:
                raise ValueError(f"Unexpected K shape: {k.shape}")
            
            if v.shape[1] == cfg.hidden_size:
                # Already in [qkv_dim, hidden_size] format
                pass
            elif v.shape[0] == cfg.hidden_size:
                v = v.T
            else:
                raise ValueError(f"Unexpected V shape: {v.shape}")
            
            # Now Q, K, V are all [qkv_dim, hidden_size] format
            # Concatenate along first dimension: [qkv_dim_total, hidden_size]
            qkv = jnp.concatenate([q, k, v], axis=0)
            # QKV is now [qkv_dim, hidden_size]
            # nnx.Linear(in_features, out_features) has kernel shape [in_features, out_features]
            # So for nnx.Linear(hidden_size, qkv_dim), kernel is [hidden_size, qkv_dim]
            # We need to transpose: [qkv_dim, hidden_size] -> [hidden_size, qkv_dim]
            qkv = qkv.T  # Now [hidden_size, qkv_dim]
            
            jax_key = f"block.{layer_idx}.attn.qkv.kernel"
            keys = [_stoi(k) for k in jax_key.split(".")]
            try:
                # QKV is now [hidden_size, qkv_dim] - correct format for nnx.Linear
                # No transform needed
                transform_val = None
                _assign_weights(keys, qkv, state_dict, f"qkv_layer_{layer_idx}", transform_val)
            except Exception as e:
                conversion_errors.append(
                    f"Failed to assign merged QKV for layer {layer_idx}: {type(e).__name__}: {e}"
                )

    # Process MXFP4 tensors
    # base_name is the JAX key with .blocks/.scales suffix (e.g., "block.6.mlp.mlp1_weight.blocks")
    # blocks_key and scales_key are the torch checkpoint keys
    # Debug: print mxfp4_pairs before processing
    if len(mxfp4_pairs) == 0:
        # Try to rebuild mxfp4_pairs if it's empty (this shouldn't happen, but let's debug)
        import warnings
        warnings.warn(f"mxfp4_pairs is empty! This indicates a bug. Rebuilding...")
        # The issue is that mxfp4_pairs should have been populated in the loop above
        # Let's check if the issue is with the loop logic
        raise RuntimeError(f"No MXFP4 pairs found! This indicates a bug in mxfp4_pairs building.")
    
    for base_name, (blocks_key, scales_key) in mxfp4_pairs.items():
        if blocks_key is None or scales_key is None:
            continue

        # Find the corresponding safetensors file and load both tensors
        blocks_tensor = None
        scales_tensor = None
        for f in files:
            with safetensors.safe_open(f, framework="numpy") as sf:
                if blocks_key in sf.keys():
                    blocks_tensor = jnp.array(sf.get_tensor(blocks_key))
                if scales_key in sf.keys():
                    scales_tensor = jnp.array(sf.get_tensor(scales_key))
                if blocks_tensor is not None and scales_tensor is not None:
                    break

        if blocks_tensor is None or scales_tensor is None:
            conversion_errors.append(f"Failed to find MXFP4 pair: {blocks_key}, {scales_key}")
            continue

        # Convert MXFP4 to full precision
        tensor = _get_mxfp4_tensor(blocks_tensor, scales_tensor)

        # base_name is already the JAX key with suffix (e.g., "block.6.mlp.mlp1_weight.blocks")
        # Remove .blocks/.scales suffix to get the actual JAX key
        jax_key = base_name.replace(".blocks", "").replace(".scales", "")
        
        # Find the transform from key_mapping by matching blocks_key
        transform = None
        blocks_key_base = blocks_key.replace(".blocks", "").replace(".scales", "")
        # Match blocks_key to find the transform - need to match the exact pattern that created base_name
        for pattern, (repl, t) in key_mapping.items():
            if isinstance(repl, tuple):
                match = re.match(pattern, blocks_key_base)
                if match:
                    # Verify this is the correct pattern by checking if repl[0] matches base_name pattern
                    expected_base = match.expand(repl[0])
                    if expected_base == base_name:
                        transform = t
                        break

        if transform is None:
            conversion_errors.append(f"Failed to find transform for MXFP4 tensor: {blocks_key}")
            continue

        keys = [_stoi(k) for k in jax_key.split(".")]
        try:
            # Pass transform.value if it's an Enum, otherwise pass transform directly
            transform_val = transform.value if hasattr(transform, 'value') else transform
            _assign_weights(keys, tensor, state_dict, base_name, transform_val)
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(
                f"Failed to assign MXFP4 '{base_name}' to '{full_jax_key}': {type(e).__name__}: {e}"
            )

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    # Note: After nnx.merge, Param values might still appear as ShapeDtypeStruct
    # until they are actually accessed during computation. This is expected behavior.

    gc.collect()
    return nnx.merge(graph_def, state_dict)
