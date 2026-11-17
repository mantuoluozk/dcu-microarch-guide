"""
Exp 9: Software Stack Tracing (decode-ai-accelerator) & HYGON Operator Library Probe
Usage: python 09_hygon_ops_and_stack.py
"""

import importlib
import os
import torch


def trace_loaded_libs():
    print("=== 1. Software Stack Probe (PyTorch -> HIP/DTK -> HSA) ===")
    print(f"  torch.__version__ : {torch.__version__}")
    print(f"  torch.version.hip : {getattr(torch.version, 'hip', None)}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  Device 0          : {props.name}")
        print(f"  gcnArchName       : {getattr(props, 'gcnArchName', 'N/A')}")
        print(f"  CU count          : {props.multi_processor_count}")

    maps_path = "/proc/self/maps"
    if os.path.exists(maps_path):
        keywords = ("libamdhip64", "libhsa-runtime64", "librocblas", "libhipblas", "librccl", "libMIOpen")
        found = set()
        with open(maps_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for kw in keywords:
                    if kw in line and ".so" in line:
                        path = line.strip().split()[-1]
                        found.add((kw, path))
        print("\n  Loaded Runtime Shared Libraries:")
        for kw, path in sorted(found):
            print(f"    {kw:<18s} -> {path}")


def probe_op_libraries():
    print("\n=== 2. HYGON / ROCm Operator Libraries Probe ===")
    candidates = [
        "triton",
        "lightop",
        "aiter",
        "deep_gemm",
        "deepgemm",
        "vllm",
        "sglang",
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "unknown")
            loc = getattr(mod, "__file__", "namespace")
            attrs = [a for a in dir(mod) if not a.startswith("_")][:12]
            print(f"  [INSTALLED] {name:<10s} | ver={ver} | path={loc}")
            print(f"              sample exports: {attrs}")
        except ImportError:
            print(f"  [MISSING]   {name}")


def profile_pytorch_kernels():
    print("\n=== 3. torch.profiler Kernel Dispatch Trace ===")
    if not torch.cuda.is_available():
        print("  Skipped: no GPU/DCU available.")
        return

    x = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    # Warmup
    _ = torch.matmul(x, w)
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        y = torch.matmul(x, w)
        # Unfused elementwise + reduction to see multiple kernel dispatches
        _ = (y * y).mean(dim=-1)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    trace_loaded_libs()
    probe_op_libraries()
    profile_pytorch_kernels()
