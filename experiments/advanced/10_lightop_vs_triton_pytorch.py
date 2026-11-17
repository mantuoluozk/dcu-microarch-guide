"""
Exp 10: lightop vs Triton vs PyTorch (RMSNorm + Kernel Dispatch Trace)
Usage: python 10_lightop_vs_triton_pytorch.py
"""

import time
import torch
import lightop
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N: tl.constexpr, eps: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    y = x * tl.rsqrt(var + eps) * w
    tl.store(Y_ptr + row * stride_y_row + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


def triton_rmsnorm(x, w, out, eps=1e-6):
    M, N = x.shape
    BLOCK_N = triton.next_power_of_2(N)
    rmsnorm_fwd_kernel[(M,)](
        x, w, out, x.stride(0), out.stride(0),
        N=N, eps=eps, BLOCK_N=BLOCK_N, num_warps=8,
    )
    return out


def torch_unfused_rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    M, N = 8192, 4096
    dtype = torch.float16
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    eps = 1e-6

    y_ref = torch_unfused_rmsnorm(x, w, eps)
    y_tri = triton_rmsnorm(x, w, out.clone(), eps)
    y_lop = lightop.fused_rms_norm_contiguous(out.clone(), x, w, eps)

    print(f"=== Exp 10: PyTorch vs Triton vs lightop RMSNorm ({M}x{N}, {dtype}) ===")
    print(f"  Max diff (Triton  vs PyTorch): {(y_tri - y_ref).abs().max().item():.6f}")
    print(f"  Max diff (lightop vs PyTorch): {(y_lop - y_ref).abs().max().item():.6f}")

    min_bytes = 2 * M * N * x.element_size()
    for name, fn in [
        ("PyTorch Unfused", lambda: torch_unfused_rmsnorm(x, w, eps)),
        ("Triton Fused", lambda: triton_rmsnorm(x, w, out, eps)),
        ("lightop fused_rms_norm", lambda: lightop.fused_rms_norm_contiguous(out, x, w, eps)),
    ]:
        ms = bench(fn)
        bw = (min_bytes / 1e9) / (ms / 1e3)
        print(f"  {name:<24s}: {ms:8.4f} ms | Effective BW = {bw:7.2f} GB/s")

    print("\n=== lightop Kernel Dispatch via torch.profiler ===")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        lightop.fused_rms_norm_contiguous(out, x, w, eps)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))


if __name__ == "__main__":
    main()
