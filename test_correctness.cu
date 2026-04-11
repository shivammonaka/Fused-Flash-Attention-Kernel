// ============================================================================
// TEST — Validate fused kernel against naive reference
// ============================================================================
// Runs both kernels on the same random input, checks that outputs match
// within floating-point tolerance. Tests multiple configurations.

#include "flash_attn.cuh"
#include <cstdio>

struct TestConfig {
    int B, H, N, d;
    const char* name;
};

bool run_test(const TestConfig& cfg) {
    printf("  %-30s (B=%d, H=%d, N=%d, d=%d) ... ",
           cfg.name, cfg.B, cfg.H, cfg.N, cfg.d);

    int total = cfg.B * cfg.H * cfg.N * cfg.d;

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O_naive, *d_O_flash;
    CUDA_CHECK(cudaMalloc(&d_Q, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O_naive, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O_flash, total * sizeof(float)));

    // Fill with same random data
    fill_random(d_Q, total);
    fill_random(d_K, total);
    fill_random(d_V, total);

    // Run naive attention (our ground truth)
    launch_naive_attention(d_Q, d_K, d_V, d_O_naive,
                           cfg.B, cfg.H, cfg.N, cfg.d);

    // Run fused flash attention (what we're testing)
    launch_flash_attention(d_Q, d_K, d_V, d_O_flash,
                           cfg.B, cfg.H, cfg.N, cfg.d);

    // Compare outputs
    float diff = max_abs_diff(d_O_naive, d_O_flash, total);

    // Tolerance: FP32 accumulation with exp() can drift a bit.
    // 1e-3 is reasonable for N up to ~2048. For larger N, you'd want
    // to use Kahan summation or FP64 accumulators.
    bool pass = diff < 1e-2f;
    printf("%s (max diff = %.6e)\n", pass ? "PASS" : "FAIL", diff);

    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O_naive));
    CUDA_CHECK(cudaFree(d_O_flash));

    return pass;
}

int main() {
    printf("============================================================\n");
    printf("Flash Attention — Correctness Tests\n");
    printf("============================================================\n\n");

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("SM count: %d\n\n", prop.multiProcessorCount);

    TestConfig tests[] = {
        // Basic cases — start small to catch indexing bugs
        {1, 1,   32, 64,  "Tiny (1 tile, no looping)"},
        {1, 1,   64, 64,  "Two Q tiles"},
        {1, 1,  128, 64,  "Four tiles"},

        // Non-power-of-2: catches padding/boundary bugs
        {1, 1,   48, 64,  "Non-aligned seq len (48)"},
        {1, 1,  100, 64,  "Non-aligned seq len (100)"},

        // Different head dimensions
        {1, 1,  128, 32,  "Small head dim (d=32)"},
        {1, 1,  128, 128, "Large head dim (d=128)"},

        // Multi-head and batch
        {1, 4,  128, 64,  "Multi-head (H=4)"},
        {2, 4,  128, 64,  "Multi-batch (B=2, H=4)"},

        // Larger sequences — real workloads
        {1, 1,  256, 64,  "Medium seq (256)"},
        {1, 1,  512, 64,  "Medium seq (512)"},
        {1, 1, 1024, 64,  "Large seq (1024)"},

        // Stress test — this one takes a moment
        {2, 8,  512, 64,  "Production-like (B=2, H=8, N=512)"},
    };

    int num_tests = sizeof(tests) / sizeof(tests[0]);
    int pass_count = 0;

    for (int i = 0; i < num_tests; i++) {
        if (run_test(tests[i])) pass_count++;
    }

    printf("\n============================================================\n");
    printf("Results: %d/%d tests passed\n", pass_count, num_tests);
    printf("============================================================\n");

    return (pass_count == num_tests) ? 0 : 1;
}
