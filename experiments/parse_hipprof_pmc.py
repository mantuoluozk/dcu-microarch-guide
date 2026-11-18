"""Summarize hipprof PMC CSV. Separate experimental variants before profiling."""

import argparse
import csv
from collections import defaultdict
from statistics import mean

RESOURCE_COLUMNS = ("grd", "wgr", "lds", "scr", "arch_vgpr", "accum_vgpr", "sgpr")


def integer(value):
    return int(value) if value and value.strip() else None


def channel_total(row, prefix):
    columns = [value for key, value in row.items() if key.startswith(prefix + "[")]
    if not columns:
        return integer(row.get(prefix))
    values = [integer(value) for value in columns]
    return sum(values) if all(value is not None for value in values) else None


def available_mean(values):
    values = list(values)
    return mean(values) if values and all(value is not None for value in values) else None


def fmt(value, decimals=0):
    return "N/A" if value is None else f"{value:,.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file")
    parser.add_argument("--kernel", help="Only include kernel names containing this text.")
    parser.add_argument("--expect-dispatches", type=int,
                        help="Require one matching kernel/resource group containing this many dispatches.")
    args = parser.parse_args()

    grouped = defaultdict(list)
    with open(args.csv_file, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"KernelName", "BeginNs", "EndNs"}
        if not required.issubset(reader.fieldnames or ()):
            parser.error(f"missing CSV columns: {', '.join(sorted(required - set(reader.fieldnames or ())))}")
        for row in reader:
            name = row["KernelName"]
            if args.kernel and args.kernel not in name:
                continue
            key = (name, *(row.get(column) for column in RESOURCE_COLUMNS))
            grouped[key].append(row)

    if not grouped:
        parser.error("no matching kernel dispatches")
    if args.expect_dispatches is not None and (
        len(grouped) != 1 or sum(map(len, grouped.values())) != args.expect_dispatches
    ):
        parser.error(f"expected one group and {args.expect_dispatches} dispatches, "
                     f"found {len(grouped)} groups and {sum(map(len, grouped.values()))} dispatches")

    print(f"=== PMC Summary: {len(grouped)} kernel/resource groups (last 5 rows per group) ===")
    for key, rows in grouped.items():
        name = key[0]
        sample = rows[-5:]

        def avg(col):
            return available_mean(integer(row.get(col)) for row in sample)

        def avg_channels(prefix):
            return available_mean(channel_total(row, prefix) for row in sample)

        duration = mean((int(row["EndNs"]) - int(row["BeginNs"])) / 1e3 for row in sample)
        hits = avg_channels("TCC_HIT")
        misses = avg_channels("TCC_MISS")
        hit_rate = 100 * hits / (hits + misses) if hits is not None and misses is not None and hits + misses else None
        print(f"\nKernel: {name} (calls={len(rows)}, sampled={len(sample)})")
        print("  " + " | ".join(f"{column}={value or 'N/A'}"
                               for column, value in zip(RESOURCE_COLUMNS, key[1:])))
        print(f"  Avg Kernel Time (us): {fmt(duration, 2)}")
        for col in ("GRBM_GUI_ACTIVE", "SQ_INSTS_VALU", "SQ_INSTS_LDS", "SQ_INSTS_VMEM_RD",
                    "SQ_INSTS_VMEM_WR", "SQ_LDS_BANK_CONFLICT", "SQ_WAIT_INST_LDS"):
            print(f"  {col}: {fmt(avg(col), 1)}")
        print(f"  TCC_HIT / TCC_MISS: {fmt(hits, 1)} / {fmt(misses, 1)} ({fmt(hit_rate, 2)} %)")
        print(f"  TCP_TA_DATA_STALL: {fmt(avg_channels('TCP_TCP_TA_DATA_STALL_CYCLES'), 1)}")


if __name__ == "__main__":
    main()
