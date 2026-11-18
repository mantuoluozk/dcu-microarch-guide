#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: bash experiments/run_bw100_baseline.sh <offload-arch> <new-output-directory>" >&2
    exit 2
fi
arch=$1
source_dir=$(cd "$(dirname "$0")" && pwd)
mkdir "$2"
cd "$2"
export LC_ALL=C

{
    date -u +%FT%TZ
    hipcc --version | sed -E 's@ \(https?://[^)]*\)@@'
    python3 -c 'import torch,triton; p=torch.cuda.get_device_properties(0); print(p.name, p.gcnArchName, "CU", p.multi_processor_count, "L2 bytes", p.L2_cache_size, "PyTorch", torch.__version__, "Triton", triton.__version__)'
    rocm-smi --showuse --showclocks
} > environment.txt
(
    cd "$source_dir"
    sha256sum common.h 02_coalescing_and_vec.hip 07_memory_latency_ladder.hip \
        08_mmac_builtin_bench.hip advanced/08_triton_gemm_mmac.py \
        parse_hipprof_pmc.py run_bw100_baseline.sh
) > source-sha256.txt

for program in 02_coalescing_and_vec 07_memory_latency_ladder 08_mmac_builtin_bench; do
    hipcc -O3 -std=c++17 --offload-arch="$arch" -save-temps "$source_dir/$program.hip" \
        -o "$program" > "$program-build.log" 2>&1
done
./02_coalescing_and_vec > e01.csv
for mode in scalar lane; do
    for seed in 42 43 44; do
        ./07_memory_latency_ladder 0 "$seed" 64 "$mode" > "e06-$mode-seed$seed.csv"
    done
    for spacing in 32 128; do
        ./07_memory_latency_ladder 0 42 "$spacing" "$mode" > "e06-$mode-spacing$spacing.csv"
    done
    for size in 16 32 512 1024 4096 8192 16384; do
        prefix="e06-$mode-pmc-$size"
        hipprof --pmc --pmc-type 3 --kernel-name global_chase_kernel -o "$prefix" \
            ./07_memory_latency_ladder "$size" 42 64 "$mode" > "$prefix.log" 2>&1
        python3 "$source_dir/parse_hipprof_pmc.py" "$prefix.csv" \
            --kernel global_chase_kernel --expect-dispatches 5 > "$prefix-summary.txt"
    done
done
./08_mmac_builtin_bench > e07.txt

run_gemm() {
    local id=$1 kernel=$2
    shift 2
    local isa=()
    if [[ "$id" != torch ]]; then
        isa=(--isa "e09-$id.s")
    fi
    python3 "$source_dir/advanced/08_triton_gemm_mmac.py" "$@" "${isa[@]}" > "e09-$id.json"
    hipprof --pmc --pmc-type 3 -o "e09-$id-pmc" \
        python3 "$source_dir/advanced/08_triton_gemm_mmac.py" "$@" --profile-only > "e09-$id-pmc.log" 2>&1
    python3 "$source_dir/parse_hipprof_pmc.py" "e09-$id-pmc.csv" \
        --kernel "$kernel" --expect-dispatches 6 > "e09-$id-pmc-summary.txt"
}
run_gemm torch Cijk --backend torch
run_gemm base gemm_kernel --tile 128 128 32
run_gemm bk16 gemm_kernel --tile 128 128 16
run_gemm stages2 gemm_kernel --tile 128 128 32 --stages 2
run_gemm group1 gemm_kernel --tile 128 128 32 --group-m 1
run_gemm historical gemm_kernel --tile 128 256 32 --warps 8 --stages 2
rocm-smi --showclocks >> environment.txt
