"""
Exp 7: Triton Fused RMSNorm vs Unfused PyTorch (Memory-bound Fusion & ISA Inspection)
Usage: python 07_triton_fused_rmsnorm.py
"""

import time
import torch

try:
    import triton
    import triton.language as tl
except ImportError as e:
    raise SystemExit(f"Triton is not installed in this environment: {e}")


@triton.jit
def rmsnorm_fwd_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    stride_x_row,
    stride_y_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptrs = X_ptr + row * stride_x_row + cols
    w_ptrs = W_ptr + cols
    y_ptrs = Y_ptr + row * stride_y_row + cols

    # Load X once from HBM into VGPRs
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Reduction inside registers / Wavefront (no extra HBM round-trips)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    y = x * rstd * w

    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask)


def triton_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 8 if BLOCK_N >= 4096 else 4
    compiled = rmsnorm_fwd_kernel[(M,)](
        x, w, y,
        x.stride(0), y.stride(0),
        N=N, eps=eps, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return y, compiled


def torch_unfused_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    x_fp32 = x.float()
    var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
    y = x_fp32 * torch.rsqrt(var + eps)
    return (y * w.float()).to(x.dtype)


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
    assert torch.cuda.is_available(), "No DCU/GPU device available"
    M, N = 8192, 4096
    dtype = torch.float16
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)

    y_ref = torch_unfused_rmsnorm(x, w)
    y_tri, compiled = triton_rmsnorm(x, w)
    max_diff = (y_ref - y_tri).abs().max().item()

    t_torch = bench(lambda: torch_unfused_rmsnorm(x, w))
    t_triton = bench(lambda: triton_rmsnorm(x, w))

    # Minimum HBM traffic for Fused RMSNorm: read X + write Y (W is cached in L2)
    min_bytes = 2 * M * N * x.element_size()
    bw_torch = (min_bytes / 1e9) / (t_torch / 1e3)
    bw_triton = (min_bytes / 1e9) / (t_triton / 1e3)

    print(f"=== Exp 7: Triton Fused RMSNorm ({M}x{N}, {dtype}) ===")
    print(f"  Max diff vs PyTorch   : {max_diff:.6f}")
    print(f"  PyTorch Unfused       : {t_torch:8.4f} ms | Effective BW = {bw_torch:7.2f} GB/s")
    print(f"  Triton Fused          : {t_triton:8.4f} ms | Effective BW = {bw_triton:7.2f} GB/s")
    print(f"  Speedup               : {t_torch / t_triton:.2f}x")

    # Inspect Triton compiled metadata & AMDGCN ISA
    meta = getattr(compiled, "metadata", None)
    if meta is not None:
        print(f"\n[Triton Metadata] {meta}")

    asm = getattr(compiled, "asm", {})
    isa_key = next((k for k in ("amdgcn", "gcn", "ptx") if k in asm), None)
    if isa_key:
        isa_text = asm[isa_key]
        lines = isa_text.splitlines()
        n_load = sum("load_dword" in l for l in lines)
        n_wait = sum("s_waitcnt" in l for l in lines)
        n_barrier = sum("s_barrier" in l for l in lines)
        n_perm = sum(("bpermute" in l or "dpp" in l) for l in lines)
        print(f"\n[ISA Summary ({isa_key})] total_lines={len(lines)}, "
              f"load_dword={n_load}, s_waitcnt={n_wait}, s_barrier={n_barrier}, permute/dpp={n_perm}")
        with open("07_rmsnorm_isa.s", "w", encoding="utf-8") as f:
            f.write(isa_text)
        print("  Full ISA saved to 07_rmsnorm_isa.s")


if __name__ == "__main__":
    main()
