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
from transformers import AutoTokenizer


from bonsai.models.oss import modeling
from bonsai.models.oss.params import create_model_from_checkpoint


def run_model(checkpoint_path: str):
    """
    Run OSS model inference.
    
    Args:
        checkpoint_path: Path to the model checkpoint directory
    """
    # Create model configuration
    config = modeling.ModelConfig.default()
    
    
    
    # Load model from checkpoint
    print(f"Loading model from {checkpoint_path}...")
    model = create_model_from_checkpoint(checkpoint_path, config)
    print("Model loaded successfully!")
    
    # Tokenize a single paragraph query instead of dummy numeric tokens
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    paragraph = (
        "In a serene afternoon, the sky stretches wide and clear above the city, "
    )
    encoded = tokenizer([paragraph], return_tensors="np", padding=True)
    tokens = jnp.array(encoded["input_ids"], dtype=jnp.int32)
    
    print(f"Running forward pass with input shape: {tokens.shape}")
    
    logits = None
    # # Forward pass
    # logits = model(tokens)
    
    # print(f"Output logits shape: {logits.shape}")
    # print(f"Vocabulary size: {logits.shape[-1]}")
    
    # # Get predictions (greedy decoding)
    # predicted_tokens = jnp.argmax(logits, axis=-1)
    # print(f"Predicted tokens: {predicted_tokens}")
    
    # # Example: generate next token
    # last_logits = logits[0, -1, :]  # Get logits for last position
    # next_token = jnp.argmax(last_logits)
    # print(f"Next predicted token: {next_token}")
    
    return model, logits, tokens


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
            next_token = int(jax.random.categorical(jax.random.PRNGKey(0),probs))
        
        tokens.append(next_token)
        num_generated += 1
        
        yield next_token
        
        # Check stop condition
        if next_token in stop_tokens:
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:

        print("\nExample:")
        print("  python -m bonsai.models.oss.tests.run_model /path/to/checkpoint")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]

    
    # Run the model
    model, logits, tokens = run_model(checkpoint_path)
    
    # Example generation
    print("\n" + "=" * 50)
    print("Example token generation:")
    print("=" * 50)
    
    # Use the same paragraph for generation starting point
    prompt_tokens = list(np.array(tokens[0]).tolist())
    print(f"Prompt tokens (first 32): {prompt_tokens[:32]}")
    print("Generated tokens:", end=" ")
    for token in generate_tokens(model, prompt_tokens, max_tokens=10, temperature=1.0):
        print(token, end=" ")
    print()


__all__ = ["run_model", "generate_tokens"]

