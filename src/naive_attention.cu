// ============================================================================
// NAIVE ATTENTION — The Baseline
// ============================================================================
//
// This implements standard scaled dot-product attention in three separate steps:
//   Step 1: S = Q @ K^T / sqrt(d)    → writes full N×N matrix to HBM
//   Step 2: P = softmax(S, dim=-1)    → reads N×N, writes N×N to HBM
//   Step 3: O = P @ V                 → reads N×N + N×d, writes N×d to HBM
//
// Total HBM traffic: O(N²) reads + writes. This is exactly the bottleneck
// that FlashAttention eliminates.
//
// We keep this around for two reasons:
//   1. Correctness reference — our fused kernel must match this output
//   2. Benchmarking baseline — shows the speedup from fusion
//
// ============================================================================

#include "flash_attn.cuh"

// ============================================================================
// Step 1: S = Q @ K^T / sqrt(d)
// ============================================================================
// Standard matrix multiply. Each thread computes one element of the N×N
// attention score matrix.
//
// Memory layout: Q, K are [N, d] for a single (batch, head).
// S is [N, N] — this is the expensive intermediate we want to eliminate.
//
// Grid:  (ceil(N/32), ceil(N/32))  — one thread block per 32×32 tile of S
// Block: (32, 32)                  — one thread per element within the tile

__global__ void matmul_qk_kernel(
    const float* Q,     // [N, d] — query matrix
    const float* K,     // [N, d] — key matrix
    float* S,           // [N, N] — output score matrix (the expensive one!)
    int N, int d,
    float scale         // 1/sqrt(d) — scaling factor
) {
    // Which element of S are we computing?
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // key index

    if (row >= N || col >= N) return;

    // Dot product: Q[row, :] · K[col, :]
    // Note: K is NOT transposed in memory — we just read K[col, :] directly.
    // The "transpose" happens logically by which index we iterate over.
    float sum = 0.0f;
    for (int k = 0; k < d; k++) {
        sum += Q[row * d + k] * K[col * d + k];
    }

    // Scale by 1/sqrt(d) to keep softmax inputs in a reasonable range.
    // Without this, for d=128, dot products grow like O(sqrt(d)), pushing
    // softmax into saturation where gradients vanish.
    S[row * N + col] = sum * scale;
}

// ============================================================================
// Step 2: Row-wise Softmax over S
// ============================================================================
// Computes softmax independently for each row of the N×N score matrix.
// Two-pass stable softmax:
//   Pass 1: find max of each row (for numerical stability)
//   Pass 2: compute exp(x - max) and normalize
//
// One thread block per row. Uses shared memory reduction for max and sum.
// This is the operation that online softmax replaces in the fused kernel.

__global__ void softmax_kernel(
    float* S,           // [N, N] — in-place softmax over each row
    int N
) {
    int row = blockIdx.x;  // One block per row
    if (row >= N) return;

    // Pointer to this row's data
    float* row_data = S + row * N;

    // --- Pass 1: Find row max ---
    // Shared memory for parallel reduction
    extern __shared__ float sdata[];

    float local_max = -FLT_MAX;
    // Each thread handles multiple elements (grid-stride loop within the row)
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        local_max = fmaxf(local_max, row_data[j]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Tree reduction to find the global max for this row
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];  // Broadcast to all threads via shared memory

    // --- Pass 2: Compute exp(x - max) and sum ---
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float val = expf(row_data[j] - row_max);
        row_data[j] = val;           // Store exp values back (reuse the buffer)
        local_sum += val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Tree reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];

    // --- Normalize ---
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        row_data[j] /= row_sum;
    }
}

// ============================================================================
// Step 3: O = P @ V
// ============================================================================
// Standard matrix multiply: [N, N] × [N, d] → [N, d]
// After softmax, P contains attention weights. Multiply by V to get output.

__global__ void matmul_pv_kernel(
    const float* P,     // [N, N] — attention weights (softmax output)
    const float* V,     // [N, d] — value matrix
    float* O,           // [N, d] — output
    int N, int d
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // dimension index

    if (row >= N || col >= d) return;

    // Weighted sum of V rows, using attention weights from P
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += P[row * N + k] * V[k * d + col];
    }
    O[row * d + col] = sum;
}

// ============================================================================
// Launch Function — Orchestrates the three steps
// ============================================================================
// Loops over batch and head dimensions, launches kernels for each (b, h) pair.
// In production, you'd batch these launches. Here clarity beats performance.

void launch_naive_attention(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d
) {
    float scale = 1.0f / sqrtf((float)d);

    // Allocate the N×N intermediate. THIS is the memory we're trying to eliminate.
    // For N=4096, FP32: 4096² × 4 bytes = 64MB per (batch, head). Ouch.
    float* S;
    CUDA_CHECK(cudaMalloc(&S, (size_t)N * N * sizeof(float)));

    // How many elements per (batch, head) slice
    int slice_size = N * d;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // Pointer arithmetic to get this (batch, head) slice
            // Layout: [B, H, N, d] → offset = (b * H + h) * N * d
            int offset = (b * H + h) * slice_size;
            const float* Qi = Q + offset;
            const float* Ki = K + offset;
            const float* Vi = V + offset;
            float* Oi = O + offset;

            // Step 1: S = Q @ K^T / sqrt(d)
            dim3 block1(32, 32);
            dim3 grid1((N + 31) / 32, (N + 31) / 32);
            matmul_qk_kernel<<<grid1, block1>>>(Qi, Ki, S, N, d, scale);

            // Step 2: softmax(S) — one block per row
            int softmax_threads = 256;
            int smem_size = softmax_threads * sizeof(float);
            softmax_kernel<<<N, softmax_threads, smem_size>>>(S, N);

            // Step 3: O = softmax(S) @ V
            dim3 block3(32, 32);
            dim3 grid3((d + 31) / 32, (N + 31) / 32);
            matmul_pv_kernel<<<grid3, block3>>>(S, Vi, Oi, N, d);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(S));
}
