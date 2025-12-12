# OSS (Open Source System) in JAX

This directory contains a pure JAX implementation of the OSS language model, using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.

## Model Architecture

The OSS model is a transformer-based language model with the following key features:

- **Mixture of Experts (MoE)**: Uses 128 experts with 4 experts per token
- **Sliding Window Attention**: Alternating layers use sliding window attention (128 tokens)
- **Sink Attention**: Implements sink tokens for improved attention
- **RoPE with YaRN**: Rotary Position Embedding with YaRN scaling
- **Swish-Gated Linear Unit (SwiGLU)**: Activation function with clamping
- **MXFP4 Quantization**: Supports loading MXFP4 quantized weights

## Model Configuration

The default configuration includes:
- 36 transformer layers
- 128 experts (MoE)
- 4 experts per token
- 64 attention heads
- 8 key-value heads
- Hidden size: 2880
- Intermediate size: 2880
- Vocabulary size: 201088

## Usage

### Loading a Model

```python
from bonsai.models.oss import ModelConfig, Transformer
from bonsai.models.oss.params import create_model_from_checkpoint
from flax import nnx

# Create configuration
cfg = ModelConfig.default(use_sharding=False)

# Load from checkpoint
model = create_model_from_checkpoint(
    checkpoint_path="/path/to/checkpoint",
    cfg=cfg,
    mesh=None  # Optional: provide mesh for sharding
)
```

### Forward Pass

```python
import jax.numpy as jnp

# Input tokens
tokens = jnp.array([[1, 2, 3, 4, 5]])

# Forward pass
logits = model(tokens)
```

## Implementation Details

### Key Components

1. **RMSNorm**: Root Mean Square Layer Normalization
2. **RotaryEmbedding**: RoPE with YaRN scaling for extended context
3. **AttentionBlock**: Multi-head attention with sliding window and sink tokens
4. **MLPBlock**: Mixture of Experts feed-forward network
5. **TransformerBlock**: Combines attention and MLP with residual connections

### Weight Loading

The implementation supports loading weights from safetensors format, including:
- Standard float16/bfloat16 weights
- MXFP4 quantized weights (for MoE layers)

The `params.py` module handles the conversion from checkpoint format to JAX arrays, including:
- MXFP4 dequantization
- Weight reshaping and transposition
- Sharding support for distributed inference

## Differences from PyTorch Implementation

- Uses Flax NNX instead of PyTorch modules
- JAX arrays instead of PyTorch tensors
- Functional transformations instead of imperative updates
- Sharding support via JAX mesh

## Contributing

Contributions are welcome! Areas for improvement:
- Additional model configurations
- Performance optimizations
- Better sharding strategies
- Extended context length support
