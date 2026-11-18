#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: bash experiments/run_bw100_lds_occupancy.sh <offload-arch> <new-output-directory>" >&2
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
    rocm-smi --showdriverversion --showuse --showclocks
} > environment.txt
for file in common.h 03_lds_bank_conflict.hip 05_occupancy_and_ilp.hip parse_hipprof_pmc.py run_bw100_lds_occupancy.sh; do
    hash=$(sed 's/\r$//' "$source_dir/$file" | sha256sum)
    printf '%s  %s\n' "${hash%% *}" "$file"
done > source-sha256-lf.txt

for program in 03_lds_bank_conflict 05_occupancy_and_ilp; do
    hipcc -O3 -std=c++17 --offload-arch="$arch" -save-temps "$source_dir/$program.hip" \
        -o "$program" > "$program-build.log" 2>&1
done
./03_lds_bank_conflict > e02.csv
./05_occupancy_and_ilp > e04.csv

collect() {
    local prefix=$1 kernel=$2
    shift 2
    hipprof --pmc --pmc-type 3 --kernel-name "$kernel" -o "$prefix" "$@" profile > "$prefix.log" 2>&1
    python3 "$source_dir/parse_hipprof_pmc.py" "$prefix.csv" --kernel "$kernel" \
        --expect-dispatches 6 > "$prefix-summary.txt"
}
for layout in degree1 degree2 degree4 degree8 degree16 degree32 padding33 xor; do
    collect "e02-$layout-pmc" lds_read_kernel ./03_lds_bank_conflict "$layout"
done
for lds in 0 16 32 48; do
    for chains in 1 2 4 8; do
        collect "e04-$chains-$lds-pmc" chase_rings ./05_occupancy_and_ilp "$chains" "$lds"
    done
done
rocm-smi --showclocks >> environment.txt
