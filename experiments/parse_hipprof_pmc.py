"""
Parse hipprof --pmc --pmc-type 3 CSV output and summarize key hardware PMC counters per kernel.
Usage: python3 parse_hipprof_pmc.py <pmc.csv>
"""

import csv
import sys
from collections import defaultdict


def sum_prefix(row, prefix):
    total = 0
    for k, v in row.items():
        if k.startswith(prefix + "["):
            try:
                total += int(v)
            except ValueError:
                pass
    return total


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 parse_hipprof_pmc.py <pmc.csv>")

    path = sys.argv[1]
    grouped = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kname = row.get("KernelName", "unknown")
            grouped[kname].append(row)

    print(f"=== PMC Summary: {path} ({len(grouped)} unique kernels) ===")
    for kname, rows in grouped.items():
        # Skip warmup rows by taking the last 5 invocations (or all if fewer)
        sample = rows[-5:] if len(rows) > 5 else rows
        n = len(sample)

        def avg_int(col):
            return sum(int(r.get(col, 0) or 0) for r in sample) / n

        def avg_pref(pref):
            return sum(sum_prefix(r, pref) for r in sample) / n

        dur_us = sum(
            (int(r["EndNs"]) - int(r["BeginNs"])) / 1e3 for r in sample
        ) / n
        tcc_hit = avg_pref("TCC_HIT")
        tcc_miss = avg_pref("TCC_MISS")
        tcc_tot = tcc_hit + tcc_miss
        l2_hit_rate = (tcc_hit / tcc_tot * 100.0) if tcc_tot > 0 else 0.0

        r0 = sample[0]
        short_name = kname if len(kname) <= 68 else kname[:65] + "..."
        print(f"\nKernel: {short_name} (calls={len(rows)})")
        print(f"  Grid={r0.get('grd')} | Block={r0.get('wgr')} | LDS={r0.get('lds')}B | "
              f"Scratch={r0.get('scr')}B | VGPR={r0.get('arch_vgpr')} (Accum={r0.get('accum_vgpr')}) | "
              f"SGPR={r0.get('sgpr')}")
        print(f"  Avg Kernel Time      : {dur_us:10.2f} us | GRBM_GUI_ACTIVE = {avg_int('GRBM_GUI_ACTIVE'):,.0f}")
        print(f"  Insts (VALU/LDS/Rd/Wr): {avg_int('SQ_INSTS_VALU'):,.0f} / "
              f"{avg_int('SQ_INSTS_LDS'):,.0f} / "
              f"{avg_int('SQ_INSTS_VMEM_RD'):,.0f} / "
              f"{avg_int('SQ_INSTS_VMEM_WR'):,.0f}")
        print(f"  SQ_LDS_BANK_CONFLICT : {avg_int('SQ_LDS_BANK_CONFLICT'):,.0f} | "
              f"SQ_WAIT_INST_LDS = {avg_int('SQ_WAIT_INST_LDS'):,.0f}")
        print(f"  L2 (TCC) Hit / Miss  : {tcc_hit:,.0f} / {tcc_miss:,.0f} (Hit Rate = {l2_hit_rate:.2f}%)")
        print(f"  TCP_TA_DATA_STALL    : {avg_pref('TCP_TCP_TA_DATA_STALL_CYCLES'):,.0f}")


if __name__ == "__main__":
    main()
