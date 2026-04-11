# Flash Attention CUDA — From Scratch

A fused self-attention CUDA kernel inspired by [FlashAttention](https://arxiv.org/abs/2205.14135), implemented from scratch in CUDA C++ for learning and demonstration.

The kernel tiles Q/K/V matrices into SRAM-sized blocks to avoid materialising the full N×N attention matrix in HBM, using **online softmax** for numerically stable streaming computation.

## Why This Exists

Standard attention computes `softmax(QK^T/√d) × V` by writing the full N×N score matrix to GPU global memory (HBM). For sequence length N=4096 with FP32, that's **64MB per attention head** — and the data is written only to be immediately read back. This makes attention **memory-bandwidth bound**, not compute bound.

FlashAttention's key insight: tile Q, K, V into GPU shared memory (SRAM), compute attention scores locally, and use **online softmax** to accumulate results without ever writing the N×N matrix to HBM. This turns O(N²) memory traffic into O(N·d).

This repository implements that algorithm from first principles as an educational project.

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
# Set your GPU architecture (A100=sm_80, H100=sm_90, RTX 4090=sm_89)
export GPU_ARCH=sm_80

# Build everything
make all

# Or individually
make test
make benchmark
```

## Running

### Correctness Tests

```bash
make test
```

Runs the fused kernel against the naive 3-kernel baseline across 13 configurations (varying B, H, N, d), checking max absolute difference is below tolerance.

### Performance Benchmarks

```bash
make benchmark
```

Reports per-config: time (ms), throughput (GFLOPS), estimated HBM bandwidth (GB/s), and fused/naive speedup ratio.

### Nsight Compute Profiling

```bash
make profile
```

Generates a detailed `.ncu-rep` profile targeting the fused kernel. Open with `ncu-ui` to see:
- Roofline position (memory-bound vs compute-bound)
- Shared memory throughput vs HBM throughput  
- Warp occupancy and stall reasons
- Instruction mix

### PyTorch Cross-Validation

```bash
pip install torch numpy
python scripts/validate_against_pytorch.py --B 1 --H 4 --N 512 --d 64
```

## Key Design Decisions

**FP32 throughout.** Real FlashAttention uses FP16/BF16 with tensor cores. This implementation uses FP32 for clarity — the algorithm is identical, the hardware utilization is lower. Converting to FP16 is a natural next step.

**Static tile sizes (Br=Bc=32).** Production FlashAttention auto-tunes tile sizes per GPU. We fix them for readability. The constants are in `flash_attn.cuh`.

**One thread-row per query.** Each thread in the ty dimension handles one query row's online softmax state. This is simpler than FlashAttention2's warp-cooperative approach but leaves performance on the table.

**Serial reductions within tiles.** The per-row max and sum in online softmax use serial loops over Bc=32 elements. A warp-shuffle reduction would be faster but harder to follow.

## Performance Gap vs FlashAttention2

This implementation will be slower than FA2. The main reasons (in order of impact):

1. **No tensor cores** — FA2 uses `mma` instructions for the tile GEMMs; we use FP32 scalar ops
2. **No warp-level tiling** — FA2 decomposes tiles further across warps with careful register allocation
3. **Bank conflicts** — our shared memory layout doesn't use swizzling to avoid conflicts
4. **No causal masking** — FA2 skips entirely-masked K/V blocks; we process all blocks
5. **No pipelining** — FA2 overlaps shared memory loads with computation using async copies

Each of these is a focused optimization pass you can add incrementally.

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
