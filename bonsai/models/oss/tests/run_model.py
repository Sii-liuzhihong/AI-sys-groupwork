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

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import P
from jax.sharding import AxisType

from bonsai.models.oss import modeling
from bonsai.models.oss.params import create_model_from_checkpoint


def run_model(checkpoint_path: str, use_sharding: bool = False):
    """
    Run OSS model inference.
    
    Args:
        checkpoint_path: Path to the model checkpoint directory
        use_sharding: Whether to use sharding for distributed inference
    """
    # Create model configuration
    config = modeling.ModelConfig.default(use_sharding=use_sharding)
    
    # Setup sharding if needed
    mesh = None
    if use_sharding:
        # Example: 2x2 mesh for FSDP and TP
        mesh = jax.make_mesh(
            (2, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit)
        )
        jax.set_mesh(mesh)
    
    # Load model from checkpoint
    print(f"Loading model from {checkpoint_path}...")
    model = create_model_from_checkpoint(checkpoint_path, config, mesh)
    print("Model loaded successfully!")
    
    # Example input tokens (you should replace this with actual tokenization)
    # This is a simple example with dummy tokens
    batch_size = 1
    seq_len = 10
    tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=jnp.int32)
    
    print(f"Running forward pass with input shape: {tokens.shape}")
    
    # Forward pass
    logits = model(tokens)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Vocabulary size: {logits.shape[-1]}")
    
    # Get predictions (greedy decoding)
    predicted_tokens = jnp.argmax(logits, axis=-1)
    print(f"Predicted tokens: {predicted_tokens}")
    
    # Example: generate next token
    last_logits = logits[0, -1, :]  # Get logits for last position
    next_token = jnp.argmax(last_logits)
    print(f"Next predicted token: {next_token}")
    
    return model, logits


def generate_tokens(
    model: nnx.Module,
    prompt_tokens: list[int],
    max_tokens: int = 50,
    temperature: float = 1.0,
    stop_tokens: list[int] | None = None,
):
    """
    Generate tokens using the OSS model.
    
    Args:
        model: The OSS Transformer model
        prompt_tokens: Initial prompt tokens
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        stop_tokens: List of token IDs to stop generation at
    
    Yields:
        Generated token IDs
    """
    if stop_tokens is None:
        stop_tokens = []
    
    tokens = list(prompt_tokens)
    num_generated = 0
    
    while num_generated < max_tokens:
        # Convert to JAX array
        input_tokens = jnp.array([tokens], dtype=jnp.int32)
        
        # Forward pass
        logits = model(input_tokens)
        
        # Get logits for the last position
        last_logits = logits[0, -1, :]
        
        # Sample next token
        if temperature == 0.0:
            next_token = int(jnp.argmax(last_logits))
        else:
            # Apply temperature
            scaled_logits = last_logits / temperature
            probs = jax.nn.softmax(scaled_logits)
            next_token = int(jax.random.categorical(jax.random.key(0), probs))
        
        tokens.append(next_token)
        num_generated += 1
        
        yield next_token
        
        # Check stop condition
        if next_token in stop_tokens:
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m bonsai.models.oss.tests.run_model <checkpoint_path> [--sharding]")
        print("\nExample:")
        print("  python -m bonsai.models.oss.tests.run_model /path/to/checkpoint")
        print("  python -m bonsai.models.oss.tests.run_model /path/to/checkpoint --sharding")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    use_sharding = "--sharding" in sys.argv
    
    # Run the model
    model, logits = run_model(checkpoint_path, use_sharding=use_sharding)
    
    # Example generation
    print("\n" + "=" * 50)
    print("Example token generation:")
    print("=" * 50)
    
    prompt = [1, 2, 3, 4, 5]  # Replace with actual tokenized prompt
    print(f"Prompt tokens: {prompt}")
    print("Generated tokens:", end=" ")
    
    for token in generate_tokens(model, prompt, max_tokens=10, temperature=1.0):
        print(token, end=" ")
    print()


__all__ = ["run_model", "generate_tokens"]

