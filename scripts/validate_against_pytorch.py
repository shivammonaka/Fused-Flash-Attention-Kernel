"""
Cross-validate CUDA kernels against PyTorch's scaled_dot_product_attention.

Writes random Q, K, V to binary files, runs PyTorch attention, writes output.
The C++ test can then load and compare.

Usage:
    python validate_against_pytorch.py --B 1 --H 1 --N 128 --d 64
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import subprocess
import struct
import os

def write_tensor(path: str, tensor: torch.Tensor):
    """Write tensor to raw binary file (float32, C-contiguous)."""
    np_arr = tensor.detach().cpu().numpy().astype(np.float32)
    np_arr.tofile(path)

def read_tensor(path: str, shape: tuple) -> torch.Tensor:
    """Read tensor from raw binary file."""
    np_arr = np.fromfile(path, dtype=np.float32).reshape(shape)
    return torch.from_numpy(np_arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--H", type=int, default=1, help="Number of heads")
    parser.add_argument("--N", type=int, default=128, help="Sequence length")
    parser.add_argument("--d", type=int, default=64, help="Head dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    B, H, N, d = args.B, args.H, args.N, args.d
    print(f"Config: B={B}, H={H}, N={N}, d={d}")

    # Generate random inputs (same distribution as CUDA fill_random)
    Q = torch.randn(B, H, N, d) * 0.5
    K = torch.randn(B, H, N, d) * 0.5
    V = torch.randn(B, H, N, d) * 0.5

    # PyTorch reference — uses the most numerically stable path
    # This is what FlashAttention2 also validates against.
    with torch.no_grad():
        O_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=False)

    print(f"Output shape: {O_ref.shape}")
    print(f"Output range: [{O_ref.min().item():.4f}, {O_ref.max().item():.4f}]")
    print(f"Output mean:  {O_ref.mean().item():.6f}")
    print(f"Output std:   {O_ref.std().item():.6f}")

    # Save for manual comparison if needed
    os.makedirs("test_data", exist_ok=True)
    write_tensor("test_data/Q.bin", Q)
    write_tensor("test_data/K.bin", K)
    write_tensor("test_data/V.bin", V)
    write_tensor("test_data/O_ref.bin", O_ref)

    print(f"\nSaved test data to test_data/")
    print("To compare with CUDA output, load these binary files in your test harness.")

    # Also run a manual softmax to verify understanding
    print("\n--- Manual verification ---")
    scale = d ** -0.5
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.softmax(S, dim=-1)
    O_manual = torch.matmul(P, V)
    diff = (O_ref - O_manual).abs().max().item()
    print(f"PyTorch SDPA vs manual softmax max diff: {diff:.2e}")
    # Should be ~1e-7 (FP32 precision)

if __name__ == "__main__":
    main()
