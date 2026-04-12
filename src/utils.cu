#include "flash_attn.cuh"
#include <curand_kernel.h>

// ============================================================================
// Random Fill Kernel
// ============================================================================
// Each thread generates one random float using cuRAND.
// We seed with thread index for reproducibility across runs.
__global__ void fill_random_kernel(float* data, int count, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // curand_init is expensive — in production you'd store states.
    // For one-time init this is fine.
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Uniform in [-0.5, 0.5] — small values prevent exp() overflow during debug
    data[idx] = curand_uniform(&state) - 0.5f;
}

void fill_random(float* d_ptr, int count) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    fill_random_kernel<<<blocks, threads>>>(d_ptr, count, 42ULL);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Validation: Max Absolute Difference
// ============================================================================
// Reduction kernel to find max|a[i] - b[i]| across all elements.
// Uses shared memory reduction within each block, then atomicMax across blocks.
//
// We use an integer atomicMax trick because CUDA doesn't have atomicMax for float.
// Since our values are positive (absolute differences), float bit-pattern order
// matches integer order, so reinterpret_cast works.

__global__ void max_abs_diff_kernel(const float* a, const float* b, int count,
                                     float* result) {
    // Shared memory for block-level reduction
    __shared__ float sdata[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread computes its local absolute difference
    float val = 0.0f;
    if (idx < count) {
        val = fabsf(a[idx] - b[idx]);
    }
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction within the block
    // At each step, half the threads drop out, and the remaining threads
    // take the max of their value and their partner's value.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Thread 0 of each block writes to global result using atomicMax trick
    if (tid == 0) {
        // Atomic max for floats via integer reinterpretation
        // Works because positive floats have the same ordering as their bit patterns
        int* result_as_int = reinterpret_cast<int*>(result);
        int val_as_int = __float_as_int(sdata[0]);
        atomicMax(result_as_int, val_as_int);
    }
}

float max_abs_diff(const float* d_a, const float* d_b, int count) {
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));  // 0.0f in IEEE 754

    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    max_abs_diff_kernel<<<blocks, threads>>>(d_a, d_b, count, d_result);

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}
