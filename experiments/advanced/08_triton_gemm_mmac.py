"""
Exp 8: Triton GEMM (tl.dot -> MMAC/MFMA) & Software Pipelining Sweep
Usage: python 08_triton_gemm_mmac.py
"""

import re
import time
import torch

try:
    import triton
    import triton.language as tl
except ImportError as e:
    raise SystemExit(f"Triton is not installed in this environment: {e}")


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def inspect_isa(compiled):
    asm = getattr(compiled, "asm", {})
    isa_key = next((k for k in ("amdgcn", "gcn", "ptx") if k in asm), None)
    if not isa_key:
        return "N/A"
    text = asm[isa_key]
    matrix_ops = len(re.findall(r"\bv_(mfma|mmac|wmma)_", text))
    wait_ops = len(re.findall(r"\bs_waitcnt\b", text))
    vgpr_m = re.search(r"\.vgpr_count:\s*(\d+)", text)
    sgpr_m = re.search(r"\.sgpr_count:\s*(\d+)", text)
    vgpr = vgpr_m.group(1) if vgpr_m else "?"
    sgpr = sgpr_m.group(1) if sgpr_m else "?"
    return f"vgpr={vgpr}, sgpr={sgpr}, matrix_ops={matrix_ops}, s_waitcnt={wait_ops}"


def main():
    assert torch.cuda.is_available(), "No DCU/GPU device available"
    M = N = K = 4096
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)
    flops = 2.0 * M * N * K

    C_ref = torch.matmul(A, B)
    t_rocblas = bench(lambda: torch.matmul(A, B))
    print(f"=== Exp 8: Triton GEMM vs torch.matmul ({M}x{N}x{K}, FP16) ===")
    print(f"  torch.matmul (rocBLAS/hipBLAS) : {t_rocblas:7.3f} ms | {(flops / 1e12) / (t_rocblas / 1e3):6.2f} TFLOPS\n")

    configs = [
        (128, 128, 32, 4, 1),
        (128, 128, 32, 4, 2),
        (128, 128, 64, 8, 2),
        (128, 256, 32, 8, 2),
    ]

    for bm, bn, bk, nw, ns in configs:
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        try:
            compiled = gemm_kernel[grid](
                A, B, C, M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                num_warps=nw, num_stages=ns,
            )
            torch.cuda.synchronize()
            diff = (C - C_ref).abs().max().item()
            ms = bench(lambda: gemm_kernel[grid](
                A, B, C, M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                num_warps=nw, num_stages=ns,
            ))
            tflops = (flops / 1e12) / (ms / 1e3)
            lds_bytes = getattr(getattr(compiled, "metadata", None), "shared", "?")
            isa_info = inspect_isa(compiled)
            print(f"  BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, warps={nw}, stages={ns} | "
                  f"{ms:7.3f} ms | {tflops:6.2f} TFLOPS | diff={diff:.3f} | "
                  f"LDS={lds_bytes}B | {isa_info}")
        except Exception as e:
            print(f"  BM={bm:3d}, BN={bn:3d}, BK={bk:2d}, warps={nw}, stages={ns} | FAILED: {e}")


if __name__ == "__main__":
    main()
