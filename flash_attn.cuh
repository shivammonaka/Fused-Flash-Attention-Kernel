#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

// ============================================================================
// Error Handling
// ============================================================================
// Wraps every CUDA API call. If it fails, prints the file/line and exits.
// Use it everywhere: CUDA_CHECK(cudaMalloc(...));
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// Constants
// ============================================================================
// Tile sizes for the fused kernel. These control how much data fits in SRAM.
//
// On A100: 192KB shared memory per SM, configurable up to 164KB.
// With d=64, FP32:
//   Q_tile: Br×d = 32×64 = 8KB
//   K_tile: Bc×d = 32×64 = 8KB
//   V_tile: Bc×d = 32×64 = 8KB
//   S_tile: Br×Bc = 32×32 = 4KB
//   Total: ~28KB — well within limits.
//
// With d=128, you'd halve Br or Bc to stay within budget.
constexpr int TILE_BR = 32;   // Number of query rows per tile
constexpr int TILE_BC = 32;   // Number of key/value rows per tile

// ============================================================================
// Kernel Declarations
// ============================================================================

// --- Kernel 1: Naive (unfused) attention ---
// Three separate operations, materialises the full N×N attention matrix in HBM.
// This is the SLOW baseline. We write it to:
//   (a) have a correct reference to validate against
//   (b) measure how much the fused kernel improves things
void launch_naive_attention(
    const float* Q,    // [B, H, N, d] queries
    const float* K,    // [B, H, N, d] keys
    const float* V,    // [B, H, N, d] values
    float* O,          // [B, H, N, d] output
    int B,             // batch size
    int H,             // number of heads
    int N,             // sequence length
    int d              // head dimension
);

// --- Kernel 2: Fused attention with online softmax ---
// Single kernel, no N×N matrix in HBM.
// Tiles Q/K/V into shared memory, uses online softmax to stream through K/V blocks.
void launch_flash_attention(
    const float* Q,    // [B, H, N, d] queries
    const float* K,    // [B, H, N, d] keys
    const float* V,    // [B, H, N, d] values
    float* O,          // [B, H, N, d] output
    int B,             // batch size
    int H,             // number of heads
    int N,             // sequence length
    int d              // head dimension
);

// ============================================================================
// Utility Functions
// ============================================================================

// Fill a device buffer with random values in [-0.5, 0.5].
// We use small values to avoid numerical issues during development.
void fill_random(float* d_ptr, int count);

// Compare two device buffers element-wise. Returns max absolute difference.
// Used to validate fused kernel output against naive reference.
float max_abs_diff(const float* d_a, const float* d_b, int count);
