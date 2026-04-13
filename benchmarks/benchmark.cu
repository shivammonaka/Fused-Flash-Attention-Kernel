// ============================================================================
// BENCHMARK — Performance comparison: naive vs fused attention
// ============================================================================
// Measures wall-clock time and computes derived metrics:
//   - Throughput in GFLOPS
//   - HBM bandwidth utilization (estimated)
//   - Speedup of fused over naive
//
// Uses CUDA events for GPU-side timing (more accurate than host timers).
//
// Run with: ./benchmark
// Profile with: ncu --set full ./benchmark  (Nsight Compute)
// ============================================================================

#include "flash_attn.cuh"
#include <cstdio>
#include <vector>

struct BenchConfig {
    int B, H, N, d;
    const char* name;
};

// ============================================================================
// FLOP Counting
// ============================================================================
// For attention: O = softmax(Q @ K^T / sqrt(d)) @ V
//
// Q @ K^T:     2 * N * N * d  FLOPs  (matmul: N×d times d×N, 2 ops per element)
// Softmax:     ~5 * N * N     FLOPs  (exp, subtract, divide per element)
// P @ V:       2 * N * N * d  FLOPs  (matmul: N×N times N×d)
//
// Total ≈ 4 * N² * d + 5 * N²   (dominated by matmuls)
//
// We simplify to: 4 * N² * d (for GFLOPS reporting)

double compute_flops(int B, int H, int N, int d) {
    double n = (double)N;
    double flops_per_head = 4.0 * n * n * d;  // Two matmuls
    return (double)B * H * flops_per_head;
}

// ============================================================================
// Memory Traffic Estimation
// ============================================================================
// Naive: reads/writes the N×N intermediate
//   Read:  Q(N*d) + K(N*d) + S(N*N) + P(N*N) + V(N*d) = 3Nd + 2N²
//   Write: S(N*N) + P(N*N) + O(N*d) = 2N² + Nd
//   Total: 4Nd + 4N²
//
// Fused: no N×N intermediate
//   Read:  Q(N*d) + K(N*d) + V(N*d) = 3Nd
//   Write: O(N*d) = Nd
//   Total: 4Nd
//
// Ratio: fused saves 4N² bytes of HBM traffic!

double naive_bytes(int B, int H, int N, int d) {
    double per_head = (4.0 * N * d + 4.0 * N * N) * sizeof(float);
    return (double)B * H * per_head;
}

double fused_bytes(int B, int H, int N, int d) {
    double per_head = 4.0 * N * d * sizeof(float);
    return (double)B * H * per_head;
}

// ============================================================================
// Benchmark Runner
// ============================================================================

void run_benchmark(const BenchConfig& cfg) {
    printf("\n--- %s (B=%d, H=%d, N=%d, d=%d) ---\n",
           cfg.name, cfg.B, cfg.H, cfg.N, cfg.d);

    int total = cfg.B * cfg.H * cfg.N * cfg.d;

    // Allocate
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, total * sizeof(float)));
    fill_random(d_Q, total);
    fill_random(d_K, total);
    fill_random(d_V, total);

    // CUDA events for precise GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int warmup = 3;    // Warm up GPU caches and frequency
    int repeats = 10;  // Average over multiple runs
    float elapsed_ms;

    // --- Benchmark Naive ---
    for (int i = 0; i < warmup; i++) {
        launch_naive_attention(d_Q, d_K, d_V, d_O, cfg.B, cfg.H, cfg.N, cfg.d);
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) {
        launch_naive_attention(d_Q, d_K, d_V, d_O, cfg.B, cfg.H, cfg.N, cfg.d);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float naive_ms = elapsed_ms / repeats;

    // --- Benchmark Fused ---
    for (int i = 0; i < warmup; i++) {
        launch_flash_attention(d_Q, d_K, d_V, d_O, cfg.B, cfg.H, cfg.N, cfg.d);
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) {
        launch_flash_attention(d_Q, d_K, d_V, d_O, cfg.B, cfg.H, cfg.N, cfg.d);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float fused_ms = elapsed_ms / repeats;

    // --- Compute metrics ---
    double flops = compute_flops(cfg.B, cfg.H, cfg.N, cfg.d);
    double naive_gflops = flops / (naive_ms * 1e6);     // GFLOPS = FLOP / (ms * 1e6)
    double fused_gflops = flops / (fused_ms * 1e6);

    double naive_bw = naive_bytes(cfg.B, cfg.H, cfg.N, cfg.d) / (naive_ms * 1e6);
    double fused_bw = fused_bytes(cfg.B, cfg.H, cfg.N, cfg.d) / (fused_ms * 1e6);

    printf("  Naive:  %8.3f ms  |  %7.1f GFLOPS  |  %7.1f GB/s HBM\n",
           naive_ms, naive_gflops, naive_bw);
    printf("  Fused:  %8.3f ms  |  %7.1f GFLOPS  |  %7.1f GB/s HBM\n",
           fused_ms, fused_gflops, fused_bw);
    printf("  Speedup: %.2fx\n", naive_ms / fused_ms);

    // Context: A100 has ~312 TFLOPS FP32 peak, ~2 TB/s HBM bandwidth
    // Your numbers should be well below these — that's expected for
    // non-tensor-core FP32 code. FA2 hits higher by using FP16 + tensor cores.

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
}

int main() {
    printf("============================================================\n");
    printf("Flash Attention — Performance Benchmark\n");
    printf("============================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Peak FP32: ~%.0f GFLOPS\n", prop.clockRate * 1e-6 * 2 * prop.multiProcessorCount * 64);
    printf("HBM Bandwidth: ~%.0f GB/s\n", prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2 / 1e6);

    BenchConfig benchmarks[] = {
        // Small — shows overhead comparison
        {1, 1,  128, 64,  "Small"},

        // Medium — transition region
        {1, 1,  256, 64,  "Medium-256"},
        {1, 1,  512, 64,  "Medium-512"},

        // Large — where fused really wins (N² savings kick in)
        {1, 1, 1024, 64,  "Large-1K"},
        {1, 1, 2048, 64,  "Large-2K"},
        {1, 1, 4096, 64,  "Large-4K"},

        // Realistic LLM shapes
        {1, 12, 512, 64,  "GPT2-like (H=12)"},
        {2,  8, 512, 64,  "Batch=2, H=8"},
        {1, 32, 512, 64,  "LLaMA-like (H=32)"},
        {2, 12, 1024, 64, "GPT2 long (B=2, H=12, N=1K)"},

        // Different head dims
        {1, 1,  512, 32,  "d=32"},
        {1, 1,  512, 64,  "d=64"},
    };

    int num = sizeof(benchmarks) / sizeof(benchmarks[0]);
    for (int i = 0; i < num; i++) {
        run_benchmark(benchmarks[i]);
    }

    printf("\n============================================================\n");
    printf("Notes:\n");
    printf("  - Times are GPU-side (CUDA events), averaged over %d runs\n", 10);
    printf("  - GFLOPS counts both Q@K^T and P@V matmuls\n");
    printf("  - HBM bandwidth is estimated from theoretical access patterns\n");
    printf("  - For roofline analysis, profile with: ncu --set full ./benchmark\n");
    printf("============================================================\n");

    return 0;
}
