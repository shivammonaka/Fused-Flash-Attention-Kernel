# Flash Attention CUDA — From Scratch

A fused self-attention CUDA kernel inspired by [FlashAttention](https://arxiv.org/abs/2205.14135), implemented from scratch in CUDA C++ for learning and demonstration.

The kernel tiles Q/K/V matrices into SRAM-sized blocks to avoid materialising the full N×N attention matrix in HBM, using **online softmax** for numerically stable streaming computation.

## Why This Exists

Standard attention computes `softmax(QK^T/√d) × V` by writing the full N×N score matrix to GPU global memory (HBM). For sequence length N=4096 with FP32, that's **64MB per attention head** — and the data is written only to be immediately read back. This makes attention **memory-bandwidth bound**, not compute bound.

FlashAttention's key insight: tile Q, K, V into GPU shared memory (SRAM), compute attention scores locally, and use **online softmax** to accumulate results without ever writing the N×N matrix to HBM.

Memory traffic (per head):
- **Naive:** O(N²) — reads/writes the full N×N attention score matrix to HBM
- **Fused:** O(N²·d / Br) — Q read once, K/V re-read once per Q tile, no N×N intermediate. FlashAttention-2 further improves this by swapping the loop order.

This repository implements that algorithm from first principles.

## Architecture

```
┌──────────────────────────────────────────────────┐
│ For each Q tile (Br rows loaded to SRAM once):   │
│                                                  │
│   ┌──────────────────────────────────────────┐   │
│   │ For each K/V block (Bc rows at a time):  │   │
│   │                                          │   │
│   │  1. Load K_tile, V_tile → SRAM           │   │
│   │  2. S_tile = Q_tile @ K_tile^T  (SRAM!)  │   │
│   │  3. Online softmax update:               │   │
│   │     • m_new = max(m_old, max(S_tile))    │   │
│   │     • rescale = exp(m_old − m_new)       │   │
│   │     • l = l × rescale + Σexp(S − m_new)  │   │
│   │     • O = O × rescale + exp(S−m) @ V     │   │
│   │  4. Update running max                   │   │
│   └──────────────────────────────────────────┘   │
│                                                  │
│   Final: O = O / l                               │
│   Write output tile → HBM                        │
└──────────────────────────────────────────────────┘
```

## Results

Benchmarked on NVIDIA Tesla T4 (sm_75, 40 SMs, 320 GB/s HBM bandwidth, ~8.1 TFLOPS FP32 peak).

### Correctness

Fused kernel matches naive 3-kernel baseline within FP32 precision across all configurations:

| Config | Max Absolute Diff |
|--------|-------------------|
| B=1, H=1, N=32, d=64 | 5.96e-08 |
| B=1, H=1, N=128, d=64 | 5.59e-08 |
| B=1, H=1, N=1024, d=64 | 7.26e-08 |
| B=2, H=8, N=512, d=64 | 8.94e-08 |

All diffs are within ~1e-07, confirming the online softmax is mathematically exact — not an approximation.

### Performance

| Config | Naive (ms) | Fused (ms) | Speedup | Fused GFLOPS |
|--------|-----------|-----------|---------|--------------|
| N=128, d=64 | 0.10 | 0.13 | 0.77x | 32 |
| N=256, d=64 | 0.30 | 0.26 | 1.17x | 65 |
| N=512, d=64 | 0.96 | 0.50 | 1.93x | 134 |
| N=1024, d=64 | 3.67 | 1.45 | 2.54x | 186 |
| N=2048, d=64 | 10.91 | 4.10 | 2.66x | 262 |
| GPT2-like (H=12, N=512) | 8.27 | 2.63 | **3.14x** | 306 |
| Batch=2, H=8, N=512 | 10.97 | 3.64 | **3.01x** | 295 |

Key observations:
- **Speedup grows with N**, matching the theoretical prediction: the naive kernel's O(N²) intermediate becomes increasingly expensive while the fused kernel avoids it entirely.
- **At small N (128), fused is slower** — the online softmax overhead (extra exp, rescaling per block) exceeds the memory savings. The crossover happens around N≈192.
- **Multi-head configs show the best speedup (3.14x)** because more thread blocks keep all 40 SMs busy, maximizing parallelism.
- **Peak throughput: 306 GFLOPS** at GPT2-like config (~3.8% of T4's FP32 peak). See gap analysis below.

### Performance Gap vs FlashAttention-2

This implementation achieves ~3.8% of T4's theoretical FP32 peak. FlashAttention-2 achieves 50-75% of hardware peak. The gap comes from specific, well-understood causes (in order of impact):

| Optimization | This Kernel | FlashAttention-2 | Estimated Impact |
|-------------|-------------|-------------------|-----------------|
| Precision & tensor cores | FP32 scalar ops | FP16 + `mma.sync` tensor core instructions | ~10-16x throughput gain |
| Warp-level tiling | One thread computes one dot product | Tiles decomposed across warps with register-level blocking | ~2-3x |
| Memory pipelining | Load tile, wait, compute, wait | `cp.async` overlaps next tile load with current compute | ~1.5-2x |
| Shared memory layout | Naive row-major | Swizzled layout to eliminate bank conflicts | ~1.2-1.5x |
| Causal masking | Processes all K/V blocks | Skips entirely-masked blocks, saving ~50% work for causal attention | ~2x for causal |
| Loop order | Outer Q, inner K/V (K/V reloaded per Q tile) | Outer K/V, inner Q (better K/V reuse) | ~1.2-1.5x |

Each of these is a focused optimization pass that can be added incrementally. The algorithm (tiling + online softmax) is identical — the difference is entirely in how the hardware is utilized.

## Project Structure

```
flash-attention-cuda/
├── include/
│   └── flash_attn.cuh          # Headers, constants, CUDA error handling
├── src/
│   ├── naive_attention.cu      # Baseline: 3-kernel unfused attention
│   ├── flash_attention.cu      # Fused kernel with online softmax
│   └── utils.cu                # Random fill, validation utilities
├── tests/
│   └── test_correctness.cu     # Validates fused vs naive across configs
├── benchmarks/
│   └── benchmark.cu            # Timing, GFLOPS, bandwidth analysis
├── scripts/
│   └── validate_against_pytorch.py  # Cross-check against torch SDPA
├── Makefile
└── README.md
```

## Building

**Prerequisites:** NVIDIA CUDA Toolkit ≥ 11.0, a compatible GPU.

```bash
# Set your GPU architecture (T4=sm_75, A100=sm_80, H100=sm_90, RTX 4090=sm_89)
# Build everything
make all GPU_ARCH=sm_75

# Or individually
make test GPU_ARCH=sm_75
make benchmark GPU_ARCH=sm_75
```

### Google Colab

```python
!git clone https://github.com/YOUR_USERNAME/flash-attention-cuda.git
%cd flash-attention-cuda
!make test GPU_ARCH=sm_75       # T4
!make benchmark GPU_ARCH=sm_75
```

## Running

### Correctness Tests

```bash
make test
```

Runs the fused kernel against the naive 3-kernel baseline across 12 configurations (varying B, H, N, d), checking max absolute difference is below tolerance.

### Performance Benchmarks

```bash
make benchmark
```

Reports per-config: time (ms), throughput (GFLOPS), estimated HBM bandwidth (GB/s), and fused/naive speedup ratio.

### Nsight Compute Profiling

```bash
make profile
```

Generates a detailed `.ncu-rep` profile targeting the fused kernel. Open with `ncu-ui` to see roofline position, shared memory vs HBM throughput, warp occupancy, and instruction mix.

### PyTorch Cross-Validation

```bash
pip install torch numpy
python scripts/validate_against_pytorch.py --B 1 --H 4 --N 512 --d 64
```

## Key Design Decisions

**FP32 throughout.** Real FlashAttention uses FP16/BF16 with tensor cores. This implementation uses FP32 for clarity — the algorithm is identical, the hardware utilization is lower. Converting to FP16 is a natural next step.

**Static tile sizes (Br=Bc=32).** Production FlashAttention auto-tunes tile sizes per GPU. We fix them for readability. The constants are in `flash_attn.cuh`. With MAX_D=64, total shared memory is 28KB, fitting within T4's 48KB limit. For A100 (164KB), MAX_D can be increased to 128.

**One thread-row per query.** Each thread in the ty dimension handles one query row's online softmax state. This is simpler than FlashAttention2's warp-cooperative approach but leaves performance on the table.

**Serial reductions within tiles.** The per-row max and sum in online softmax use serial loops over Bc=32 elements. A warp-shuffle reduction would be faster but harder to follow.

## What To Read

The source files are heavily commented and designed to be read in this order:

1. **`include/flash_attn.cuh`** — Constants, layout, error handling
2. **`src/naive_attention.cu`** — The 3-step baseline (understand what we're fusing)
3. **`src/flash_attention.cu`** — The fused kernel (the core algorithm)
4. **`benchmarks/benchmark.cu`** — How to measure GPU kernel performance correctly
5. **`tests/test_correctness.cu`** — Validation strategy

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, 2018)
- [CUDA C++ Programming Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)

## License

MIT