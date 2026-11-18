"""
E09: one GEMM configuration per process, with separate timing and PMC runs.
"""

import argparse
import json
from pathlib import Path
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
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


def measure(fn, *, samples=5, iters=20):
    for _ in range(10):
        fn()
    times = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) / iters)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=(4096, 4096, 4096),
                        metavar=("M", "N", "K"))
    parser.add_argument("--tile", nargs=3, type=int, default=(128, 128, 32),
                        metavar=("BM", "BN", "BK"))
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--group-m", type=int, default=8)
    parser.add_argument("--backend", choices=("triton", "torch"), default="triton")
    parser.add_argument("--profile-only", action="store_true",
                        help="Run only the selected GEMM, without the reference or timed samples.")
    parser.add_argument("--isa", type=Path, help="Write the selected Triton kernel ISA.")
    args = parser.parse_args()

    M, N, K = args.shape
    bm, bn, bk = args.tile
    if min(M, N, K, args.warps, args.stages, args.group_m) < 1:
        parser.error("shape, warps, stages and group-m must be positive")
    if any(size < 1 or size & (size - 1) for size in args.tile):
        parser.error("tile dimensions must be positive powers of two")
    if args.isa and args.backend != "triton":
        parser.error("--isa requires --backend triton")

    torch.manual_seed(42)
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

    def launch_triton():
        return gemm_kernel[grid](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=args.group_m,
            num_warps=args.warps, num_stages=args.stages,
        )

    def launch_torch():
        torch.mm(A, B, out=C)

    launch = launch_triton if args.backend == "triton" else launch_torch
    compiled = launch()
    torch.cuda.synchronize()
    if args.isa:
        isa = compiled.asm.get("amdgcn") or compiled.asm.get("gcn")
        if not isa:
            raise RuntimeError("The selected backend did not expose DCU ISA")
        args.isa.write_text(isa, encoding="utf-8")

    result = {
        "backend": args.backend, "shape": args.shape, "seed": 42,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "triton": triton.__version__, "profile_only": args.profile_only,
    }
    if args.backend == "triton":
        result.update(tile=args.tile, warps=args.warps, stages=args.stages,
                      group_m=args.group_m, lds_bytes=compiled.metadata.shared)
    if args.profile_only:
        for _ in range(5):
            launch()
        torch.cuda.synchronize()
        result["gemm_launches"] = 6
    else:
        # Match the FP16 output contract; check against FP32 accumulation.
        reference = (A.float() @ B.float()).half()
        torch.testing.assert_close(C, reference, rtol=1e-2, atol=2e-2)
        result["max_abs_error"] = (C - reference).abs().max().item()
        result["validation"] = "pass"
        result["ms_samples"] = measure(launch)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
