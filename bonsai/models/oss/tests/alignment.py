import argparse
import os
import sys
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers import AutoTokenizer, AutoModelForCausalLM

from bonsai.models.oss import modeling
from bonsai.models.oss.params import create_model_from_checkpoint


def compare_models(
    torch_model_path: str,
    checkpoint_path: str,
    tokenizer_path: Optional[str] = None,
    torch_device: str = "cuda:0",
    jax_device: str = "cuda:1",
    dtype: str = "bfloat16",
    max_new_tokens: int = 50,
    diff_layers: bool = False,
):
    """
    Compare JAX OSS model with PyTorch reference model.
    
    Args:
        torch_model_path: Path to the PyTorch model checkpoint directory
        checkpoint_path: Path to the checkpoint directory for JAX model loading
        tokenizer_path: Path or HF ID for tokenizer (defaults to torch_model_path)
        torch_device: Device to run PyTorch model on (e.g., 'cuda:0')
        jax_device: Device to run JAX model on (e.g., 'cuda:1')
        dtype: Data type for comparison ('float32', 'float16', 'bfloat16')
    """
    print(f"Loading PyTorch model from {torch_model_path} on {torch_device}...")
    torch_model = AutoModelForCausalLM.from_pretrained(
        torch_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": torch_device},
    )
    # Enable hidden states for layer-wise comparison
    if hasattr(torch_model, "config"):
        torch_model.config.output_hidden_states = True

    
    tokenizer_path = tokenizer_path or torch_model_path
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    print(f"Loading JAX model from {checkpoint_path} on {jax_device}...")
    
    # Configure JAX device
    try:
        # Parse jax_device string (e.g., "cuda:1" -> platform="gpu", id=1)
        if "cuda" in jax_device or "gpu" in jax_device:
            device_id = int(jax_device.split(":")[-1]) if ":" in jax_device else 0
            jax_dev = jax.devices("gpu")[device_id]
        else:
            jax_dev = jax.devices("cpu")[0]
        print(f"Target JAX device: {jax_dev}")
    except Exception as e:
        print(f"Warning: Failed to parse/set JAX device '{jax_device}': {e}")
        print(f"Available devices: {jax.devices()}")
        jax_dev = jax.devices()[0]
        print(f"Using default device: {jax_dev}")

    # Configure JAX model
    config = modeling.ModelConfig.default(use_sharding=False)
    
    # Load JAX model
    with jax.default_device(jax_dev):
        jax_model = create_model_from_checkpoint(checkpoint_path, config)

    # Report parameter counts and estimated memory for Torch vs JAX
    def torch_param_stats(model: torch.nn.Module):
        total_params = 0
        total_bytes = 0
        for p in model.parameters():
            n = p.numel()
            total_params += n
            # dtype size
            elem_size = torch.finfo(p.dtype).bits // 8 if p.dtype.is_floating_point else p.element_size()
            total_bytes += n * elem_size
        return total_params, total_bytes

    def jax_param_stats(module: nnx.Module):
        # Traverse nnx state dict and sum param sizes
        graph_def, abs_state = nnx.split(module)
        sd = abs_state.to_pure_dict()
        def walk(d):
            params = 0
            bytes_ = 0
            for v in d.values():
                if isinstance(v, dict):
                    p2, b2 = walk(v)
                    params += p2
                    bytes_ += b2
                else:
                    arr = np.array(v)
                    params += arr.size
                    bytes_ += arr.size * arr.dtype.itemsize
            return params, bytes_
        return walk(sd)

    t_params, t_bytes = torch_param_stats(torch_model)
    j_params, j_bytes = jax_param_stats(jax_model)
    print("\n=== PARAMETER STATS ===")
    print(f"Torch params: {t_params:,} (~{t_bytes/1e6:.1f} MB)")
    print(f"JAX  params: {j_params:,} (~{j_bytes/1e6:.1f} MB)")
    
    # Prepare input
    text = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(text, return_tensors="pt")
    # Ensure attention_mask is set to avoid HF warnings and unexpected behavior
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.eos_token_id).long()
    input_ids = inputs.input_ids.to(torch_device)
    attention_mask = inputs.attention_mask.to(torch_device)
    
    print(f"Running inference on input: '{text}'")
    
    # PyTorch Forward
    # Use standard HF forward returning logits [batch, seq_len, vocab]
    with torch.no_grad():
        torch_out = torch_model(input_ids, attention_mask=attention_mask)
        torch_logits = torch_out.logits
    
    # JAX Forward
    # Move input to JAX device
    jax_input_ids = jax.device_put(jnp.array(input_ids.cpu().numpy()), jax_dev)
    
    with jax.default_device(jax_dev):
        jax_logits = jax_model(jax_input_ids)
    
    # Compare
    print("\nComparing Logits...")
    torch_logits_np = torch_logits.float().cpu().numpy()
    jax_logits_np = np.array(jax_logits, dtype=np.float32)
    
    # Check shapes
    if torch_logits_np.shape != jax_logits_np.shape:
        print(f"SHAPE MISMATCH: Torch {torch_logits_np.shape} vs JAX {jax_logits_np.shape}")
        return False
        
    # Compute differences
    diff = np.abs(torch_logits_np - jax_logits_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max Difference: {max_diff:.6f}")
    print(f"Mean Difference: {mean_diff:.6f}")
    
    # Tolerances
    atol = 1e-3 if dtype == "float32" else 1e-2
    rtol = 1e-3 if dtype == "float32" else 1e-2
    
    try:
        np.testing.assert_allclose(torch_logits_np, jax_logits_np, rtol=rtol, atol=atol)
        print("\nSUCCESS: Logits match within tolerance!")
        # On success, also print generation outputs for both models
        print("\nGenerating outputs (greedy) for both models...")
        # Torch generation (greedy)
        with torch.no_grad():
            gen_ids_torch = torch_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            torch_text = tokenizer.decode(gen_ids_torch[0].tolist(), skip_special_tokens=True)
        print("Torch output:")
        print(torch_text)

        # JAX generation (greedy)
        def jax_generate(model: nnx.Module, prompt_ids: np.ndarray, max_new_tokens: int = 50, eos_id: int | None = None):
            ids = prompt_ids.tolist()
            for _ in range(max_new_tokens):
                arr = jnp.array([ids], dtype=jnp.int32)
                logits = model(arr)
                next_id = int(jnp.argmax(logits[0, -1, :]))
                ids.append(next_id)
                if eos_id is not None and next_id == eos_id:
                    break
            return np.array(ids, dtype=np.int32)

        with jax.default_device(jax_dev):
            gen_ids_jax = jax_generate(
                jax_model,
                np.array(input_ids.cpu().numpy()[0], dtype=np.int32),
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_token_id,
            )
        jax_text = tokenizer.decode(gen_ids_jax.tolist(), skip_special_tokens=True)
        print("JAX output:")
        print(jax_text)
        return True
    except AssertionError as e:
        print(f"\nFAILURE: Logits do not match.\n{e}")
        # Optional: attention vs MLP breakdown for first divergent layer
        if diff_layers:
            # Ensure Torch returns hidden states for per-layer comparison
            if hasattr(torch_model, "config"):
                torch_model.config.output_hidden_states = True
            with torch.no_grad():
                torch_out = torch_model(input_ids, attention_mask=attention_mask)
            if hasattr(torch_out, "hidden_states") and torch_out.hidden_states is not None:
                print("\nPer-layer breakdown (JAX attention vs MLP) for first divergent layer:")
                # JAX: use forward_debug to collect attn/mlp/final per block
                def jax_block_debug(model: nnx.Module, ids: np.ndarray):
                    # embeddings
                    x = model.embedding.embedding[ids]
                    outputs = []
                    for blk in model.block:
                        attn_out, mlp_out, final_out = blk.forward_debug(x)
                        outputs.append((np.array(attn_out, dtype=np.float32),
                                        np.array(mlp_out, dtype=np.float32),
                                        np.array(final_out, dtype=np.float32)))
                        x = final_out
                    return outputs
                j_outputs = jax_block_debug(jax_model, np.array(input_ids.cpu().numpy()[0], dtype=np.int32))
                # Torch hidden states include embeddings first; align per-layer outputs
                t_layer_outs = [h.float().cpu().numpy()[0] for h in torch_out.hidden_states[1:1+len(j_outputs)]]
                threshold = 0.5
                divergent_idx = None
                for i, ((attn_out, mlp_out, final_out), t_out) in enumerate(zip(j_outputs, t_layer_outs)):
                    d = np.abs(final_out - t_out)
                    if d.max() > threshold:
                        divergent_idx = i
                        print(f"  First divergent layer: {i} (max={d.max():.6f}, mean={d.mean():.6f})")
                        # Print component magnitudes to guide debugging
                        print(f"    JAX attn_out:   max={np.max(np.abs(attn_out)):.6f}, mean={np.mean(np.abs(attn_out)):.6f}")
                        print(f"    JAX mlp_out:    max={np.max(np.abs(mlp_out)):.6f}, mean={np.mean(np.abs(mlp_out)):.6f}")
                        # Heuristic: compare final_out vs torch, and also x_mid vs torch to infer which component misaligns
                        # x_mid = final_out - mlp_out
                        x_mid = final_out - mlp_out
                        d_mid = np.abs(x_mid - t_out)
                        print(f"    Residual after attn (x_mid) diff: max={d_mid.max():.6f}, mean={d_mid.mean():.6f}")
                        # residual after mlp (approx): not available directly; report final diff
                        break
                if divergent_idx is None:
                    print("  No layer exceeded divergence threshold; consider lowering threshold or inspecting later layers.")
        # Optional: layer-wise diff using Torch hidden_states and JAX per-block outputs
        if diff_layers and hasattr(torch_out, "hidden_states") and torch_out.hidden_states is not None:
            print("\nLayer-wise hidden state diffs (Torch vs JAX, after each block):")
            # Compute JAX hidden states per block
            def jax_hidden_states(model: nnx.Module, ids: np.ndarray):
                # Use Variable indexing instead of deprecated .value
                x = model.embedding.embedding[ids]  # [n_tokens, hidden]
                h_list = []
                for blk in model.block:
                    x = blk(x)
                    h_list.append(np.array(x, dtype=np.float32))
                return h_list
            j_hs = jax_hidden_states(jax_model, np.array(input_ids.cpu().numpy()[0], dtype=np.int32))
            # Torch hidden_states include embeddings; per-layer
            t_hs = [h.float().cpu().numpy()[0] for h in torch_out.hidden_states[1:1+len(j_hs)]]
            for i, (th, jh) in enumerate(zip(t_hs, j_hs)):
                d = np.abs(th - jh)
                print(f"  Layer {i}: max={d.max():.6f}, mean={d.mean():.6f}")
        # Also print generation outputs to compare responses when logits differ
        print("\nGenerating outputs (greedy) for both models...")
        # Torch generation (greedy)
        with torch.no_grad():
            gen_ids_torch = torch_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            torch_text = tokenizer.decode(gen_ids_torch[0].tolist(), skip_special_tokens=True)
        print("Torch output:")
        print(torch_text)

        # JAX generation (greedy)
        def jax_generate(model: nnx.Module, prompt_ids: np.ndarray, max_new_tokens: int = 50, eos_id: int | None = None):
            ids = prompt_ids.tolist()
            for _ in range(max_new_tokens):
                arr = jnp.array([ids], dtype=jnp.int32)
                logits = model(arr)
                next_id = int(jnp.argmax(logits[0, -1, :]))
                ids.append(next_id)
                if eos_id is not None and next_id == eos_id:
                    break
            return np.array(ids, dtype=np.int32)

        with jax.default_device(jax_dev):
            gen_ids_jax = jax_generate(
                jax_model,
                np.array(input_ids.cpu().numpy()[0], dtype=np.int32),
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_token_id,
            )
        jax_text = tokenizer.decode(gen_ids_jax.tolist(), skip_special_tokens=True)
        print("JAX output:")
        print(jax_text)
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify GPT-OSS JAX implementation against PyTorch")
    parser.add_argument("--torch-model", required=True, help="Path to the PyTorch model checkpoint directory")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory (containing safetensors)")
    # No external repo required; use HF models
    parser.add_argument("--tokenizer", help="Path or HF ID of tokenizer (optional)")
    parser.add_argument("--torch-device", default="cuda:0", help="Device for PyTorch model (e.g., cuda:0)")
    parser.add_argument("--jax-device", default="cuda:1", help="Device for JAX model (e.g., cuda:1)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Data type")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Tokens to generate for output comparison")
    parser.add_argument("--diff-layers", action="store_true", help="Print layer-wise hidden state diffs")
    parser.add_argument("--layer-breakdown", action="store_true", help="Print attention vs MLP diffs for first divergent layer")
    
    args = parser.parse_args()

    # No external path needed

    success = compare_models(
        torch_model_path=args.torch_model,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        torch_device=args.torch_device,
        jax_device=args.jax_device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        diff_layers=args.diff_layers,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
