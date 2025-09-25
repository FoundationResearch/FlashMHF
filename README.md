# FlashMHF: Flash Multi-Head Feed-Forward Network

## Overview

FlashMHF introduces a novel **Flash Multi-Head Feed-Forward Network** architecture that addresses the fundamental limitations of naive Multi-Head FFNs through two key innovations:

1. **Scale-Balanced Parallel Sub-Networks**: Maintains optimal dimension ratios across model scales through dynamically weighted parallel sub-networks
2. **Memory-Efficient Flash Kernel**: I/O-aware fused kernel that computes outputs online in SRAM, eliminating the memory explosion problem

### Key Features

- 🚀 **Superior Performance**: Consistently outperforms SwiGLU FFN baselines on perplexity and downstream tasks
- ⚡ **Memory Efficient**: 3-5x reduction in peak GPU memory usage compared to SwiGLU FFN  
- 🔥 **Fast Inference**: Up to 1.08x speedup on Hopper architecture
- 📈 **Scalable**: Validated on models from 128M to 1.3B parameters
- 🛠️ **Production Ready**: Optimized kernels for both Triton and ThunderKittens (CUDA)

## Architecture

FlashMHF draws inspiration from the structural symmetry between self-attention and FFNs:

```
Attention: softmax(QK^T/√d_k)V
FFN:       σ(XW₁)W₂
```

Our multi-head design partitions computation across parallel sub-networks with learned gating:

```
O = Σₑ Rₑ * ((SiLU(Q @ Kₑ^T) ⊙ (Q @ Uₑ^T)) @ Vₑ)
```

### Key Innovations

1. **Parallel FFN Sub-Networks**: Each head uses E sub-networks with balanced dimension dₑ ≈ (8/3)dₕ
2. **Flash Algorithm**: Blockwise computation avoids materializing large intermediate tensors
3. **Gating Mechanism**: Sigmoid-normalized routing weights for dynamic expert selection

## Model Configurations

We provide pre-configured models at different scales:

| Model Size | Hidden Size | Layers | FFN Heads | Experts/Head | Config File |
|------------|-------------|--------|-----------|--------------|-------------|
| 128M | 768 | 10 | 8 | 8 | [configure-1-tinymhffnmoe.json](configs/mhffnmoe/configure-1-tinymhffnmoe.json) |
| 370M | 1024 | 20 | 8 | 11 | [configure-2-mhffnmoe-TriDao-350M.json](configs/mhffnmoe/configure-2-mhffnmoe-TriDao-350M.json) |
| 1.3B | 2048 | 20 | 16 | 22 | [configure-3-mhffnmoe-TriDao-1.3B.json](configs/mhffnmoe/configure-3-mhffnmoe-TriDao-1.3B.json) |

## Performance Results

### Language Modeling (370M Scale)

| Variant | Eval Loss ↓ | Eval PPL ↓ |
|---------|-------------|------------|
| Baseline (SwiGLU) | 3.030 | 20.69 |
| PKV (d_h=128) | 3.334 | 28.07 |
| MH-FFN (d_h=128) | 3.031 | 20.71 |
| **FlashMHF (d_h=128)** | **3.014** | **20.36** |

### Memory & Speed Improvements

- **Memory**: 3-5x reduction in peak GPU memory vs SwiGLU FFN
- **Speed**: Up to 1.08x inference speedup on H100 GPUs
- **Scaling**: Consistent improvements from 128M to 1.3B parameters

## Kernels

FlashMHF includes optimized kernel implementations:

### Triton Kernels
- **Location**: `kernels/triton/`
- **Features**: Auto-tuned block sizes, memory-coalesced access patterns
- **Usage**: Default backend, works on most CUDA GPUs

### ThunderKittens CUDA Kernels  
- **Location**: `kernels/thunderkittens/`
- **Features**: Highly optimized for H100, warp-level programming
- **Requirements**: CUDA 12.0+, H100 GPU recommended

## Repository Structure

```
FlashMHF/
├── models/flashmhf/              # Model implementations
│   ├── modeling_mhffnmoe.py      # Main model class
│   ├── configuration_mhffnmoe.py # Model configuration
│   └── convert_mhffnmoe_weights_to_hf.py
├── kernels/                      # Optimized kernel implementations  
│   ├── triton/                   # Triton-based kernels
│   │   ├── flash_mlp.py          # Single-head flash MLP
│   │   └── flash_mlp_moe.py      # Multi-expert flash MLP
│   └── thunderkittens/           # CUDA-based kernels
│       ├── flash_ffn_moe.cu      # CUDA kernel implementation
│       ├── flash_ffn_moe_torch.py # PyTorch wrapper
│       └── run_from_python.py    # Test script
├── configs/                      # Model configurations
│   ├── baseline/                 # SwiGLU baseline configs
│   ├── mhffn/                    # Naive multi-head FFN configs  
└── └── mhffnmoe/                 # FlashMHF configs
```