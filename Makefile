# ============================================================================
# Flash Attention CUDA — Makefile
# ============================================================================
#
# Targets:
#   make test       — Build and run correctness tests
#   make benchmark  — Build and run performance benchmarks
#   make all        — Build everything
#   make clean      — Remove build artifacts
#   make profile    — Run benchmark under Nsight Compute (requires ncu)
#
# Configuration:
#   Set GPU_ARCH to match your GPU. Common values:
#     A100:  sm_80
#     H100:  sm_90
#     RTX 3090: sm_86
#     RTX 4090: sm_89
#
# ============================================================================

# --- Compiler ---
NVCC = nvcc

# --- GPU Architecture ---
# Change this to match your GPU!
GPU_ARCH ?= sm_80

# --- Compiler Flags ---
# -O3:          Maximum optimization
# -arch:        Target GPU architecture
# --use_fast_math: Use faster (slightly less precise) math intrinsics
# -Iinclude:    Header search path
# -lineinfo:    Keep line info for profiling (no perf cost)
NVCC_FLAGS = -O3 -arch=$(GPU_ARCH) --use_fast_math -Iinclude -lineinfo

# For debugging (uncomment these, comment the above):
# NVCC_FLAGS = -G -g -arch=$(GPU_ARCH) -Iinclude

# --- Source files ---
SRCS = src/utils.cu src/naive_attention.cu src/flash_attention.cu

# --- Build directory ---
BUILD_DIR = build

# ============================================================================
# Targets
# ============================================================================

.PHONY: all test benchmark clean profile

all: $(BUILD_DIR)/test_correctness $(BUILD_DIR)/benchmark

# --- Correctness Tests ---
$(BUILD_DIR)/test_correctness: tests/test_correctness.cu $(SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

test: $(BUILD_DIR)/test_correctness
	./$<

# --- Performance Benchmarks ---
$(BUILD_DIR)/benchmark: benchmarks/benchmark.cu $(SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

benchmark: $(BUILD_DIR)/benchmark
	./$<

# --- Nsight Compute Profiling ---
# Generates a detailed report for the fused kernel only.
# Requires NVIDIA Nsight Compute (ncu) to be installed.
profile: $(BUILD_DIR)/benchmark
	ncu --set full \
	    --kernel-name flash_attention_kernel \
	    --launch-skip 3 --launch-count 1 \
	    -o flash_attn_profile \
	    ./$<
	@echo "Profile saved to flash_attn_profile.ncu-rep"
	@echo "Open with: ncu-ui flash_attn_profile.ncu-rep"

# --- Build directory ---
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# --- Clean ---
clean:
	rm -rf $(BUILD_DIR) *.ncu-rep
