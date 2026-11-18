"""Render the E01, E06 and E09 figures from a BW100 evidence ZIP (requires matplotlib)."""

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

from parse_hipprof_pmc import channel_total


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/public/data/bw100-20261118"
BLUE, TEAL, SLATE, AMBER = "#1d4ed8", "#0f766e", "#64748b", "#a04c09"


def read_csv(archive, name):
    with archive.open(name) as source:
        return list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig", newline="")))


def finish(figure, output, name):
    figure.savefig(output / name, dpi=180, bbox_inches="tight", pad_inches=0.17)
    plt.close(figure)


def horizontal_bars(axis, labels, values, colors, number_format, xlabel):
    bars = axis.barh(labels, values, height=0.6, color=colors)
    axis.bar_label(bars, labels=[format(value, number_format) for value in values], padding=5)
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) * 1.25)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color="#e2e8f0")
    axis.set_axisbelow(True)


def latency_axis(axis):
    axis.set_xscale("log", base=2)
    axis.set_xlim(3, 90000)
    axis.set_xticks([4, 32, 256, 1024, 8192, 65536],
                    ["4 KiB", "32 KiB", "256 KiB", "1 MiB", "8 MiB", "64 MiB"])
    for size in (1024, 8192):
        axis.axvline(size, color=SLATE, linestyle=":", linewidth=1.2, zorder=0)
    axis.set_xlabel("工作集大小（对数刻度）")


def plot_copy(archive, output):
    samples = defaultdict(list)
    for row in read_csv(archive, "e01.csv"):
        if row["validation"] != "pass":
            raise ValueError("E01 contains a failed copy validation")
        samples[row["variant"]].append(float(row["effective_GB_per_s"]))

    figure, (vector, stride) = plt.subplots(
        2, figsize=(5.9, 5.1), height_ratios=[2, 4], layout="constrained"
    )
    vector.set_title("向量化对照 · 相同线程数", loc="left")
    horizontal_bars(vector, ["四次标量", "float4"],
                    [median(samples[key]) for key in ("scalar_x4", "float4")],
                    [SLATE, BLUE], ",.1f", "有效带宽 (GB/s)")
    stride.set_title("合并度对照 · 相同标量 kernel", loc="left")
    strides = [1, 2, 4, 8, 16, 32]
    horizontal_bars(stride, [f"stride {value}" for value in strides],
                    [median(samples[f"stride_{value}"]) for value in strides],
                    BLUE, ".1f", "有效带宽 (GB/s)")
    finish(figure, output, "e01-copy-bandwidth.png")


def plot_latency(archive, output):
    figure, axis = plt.subplots(figsize=(5.9, 3.8), layout="constrained")
    for path, label, color, marker, style in (
        ("scalar", "标量加载", BLUE, "o", "-"),
        ("lane", "向量加载", TEAL, "s", "--"),
    ):
        samples = defaultdict(list)
        for seed in (42, 43, 44):
            for row in read_csv(archive, f"e06-{path}-seed{seed}.csv"):
                if row["path"] == f"global_{path}" and row["spacing_bytes"] == "64":
                    samples[int(row["size_KiB"])].append(float(row["clock64_ticks_per_hop"]))
        if not samples or any(len(values) != 15 for values in samples.values()):
            raise ValueError(f"E06 {path} expects 15 samples at each size")
        sizes = sorted(samples)
        axis.fill_between(sizes, [min(samples[n]) for n in sizes],
                          [max(samples[n]) for n in sizes], color=color, alpha=0.13)
        axis.plot(sizes, [median(samples[n]) for n in sizes], label=label,
                  color=color, marker=marker, linestyle=style, linewidth=2, markersize=3.5)

    latency_axis(axis)
    axis.set_ylim(bottom=0)
    axis.set_ylabel("clock64 tick/跳")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e2e8f0")
    axis.legend(loc="upper left", frameon=False)
    finish(figure, output, "e06-latency-ladder.png")

    sizes = [16, 32, 512, 1024, 4096, 8192, 16384]
    figure, axis = plt.subplots(figsize=(5.9, 3.4), layout="constrained")
    for path, label, color, marker, face in (
        ("scalar", "标量加载", BLUE, "o", BLUE),
        ("lane", "向量加载", TEAL, "s", "none"),
    ):
        rates = []
        for size in sizes:
            rows = [row for row in read_csv(archive, f"e06-{path}-pmc-{size}.csv")
                    if "global_chase_kernel" in row["KernelName"]]
            if len(rows) != 5:
                raise ValueError(f"E06 {path} {size} KiB expects five PMC dispatches")
            hits = sum(channel_total(row, "TCC_HIT") for row in rows)
            misses = sum(channel_total(row, "TCC_MISS") for row in rows)
            rates.append(100 * hits / (hits + misses))
        axis.scatter(sizes, rates, label=label, marker=marker, edgecolors=color,
                     facecolors=face, s=67 if path == "lane" else 28, linewidth=1.5)
    latency_axis(axis)
    axis.set_ylim(-6, 106)
    axis.set_ylabel("TCC 命中率 (%)")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e2e8f0")
    axis.legend(loc="upper right", frameon=False)
    finish(figure, output, "e06-tcc-hit-rate.png")


def plot_lds_conflicts(archive, output):
    layouts = [f"degree{n}" for n in (1, 2, 4, 8, 16, 32)] + ["padding33", "xor"]
    labels = ["1", "2", "4", "8", "16", "32", "Padding", "XOR"]
    samples = defaultdict(list)
    for row in read_csv(archive, "e02.csv"):
        if row["validation"] != "pass":
            raise ValueError("E02 failed validation")
        samples[row["variant"]].append(float(row["ms"]))
    rates = []
    for layout in layouts:
        rows = read_csv(archive, f"e02-{layout}-pmc.csv")[-5:]
        rates.append(sum(int(row["SQ_LDS_BANK_CONFLICT"]) for row in rows)
                     / sum(int(row["SQ_INSTS_LDS"]) for row in rows))
    figure, (time, conflict) = plt.subplots(2, figsize=(5.9, 5.7), layout="constrained")
    colors = [BLUE] * 5 + [AMBER, TEAL, TEAL]
    time.set_title("耗时 · 无 profiler", loc="left")
    horizontal_bars(time, labels, [median(samples[key]) for key in layouts],
                    colors, ".3f", "耗时中位数 (ms)")
    conflict.set_title("冲突计数 · 按 LDS 指令归一化", loc="left")
    horizontal_bars(conflict, labels, rates, colors, ".2f", "冲突计数 / LDS 指令")
    finish(figure, output, "e02-lds-conflicts.png")


def plot_occupancy(archive, output):
    samples = defaultdict(list)
    for row in read_csv(archive, "e04.csv"):
        if row["validation"] != "pass":
            raise ValueError("E04 failed validation")
        samples[int(row["chains"]), int(row["max_waves_per_CU"])].append(
            float(row["logical_read_GB_per_s"])
        )
    figure, axis = plt.subplots(figsize=(5.9, 3.65), layout="constrained")
    for chains, color, marker, style in (
        (1, SLATE, "o", "-"), (2, TEAL, "s", "--"),
        (4, BLUE, "^", "-"), (8, AMBER, "D", ":"),
    ):
        waves = sorted(waves for count, waves in samples if count == chains)
        axis.plot(waves, [median(samples[chains, waves_at_cu]) for waves_at_cu in waves],
                  color=color, marker=marker, linestyle=style, linewidth=2, label=f"{chains} 条链")
    axis.set_xticks(sorted({waves for _, waves in samples}))
    axis.set_ylim(bottom=0)
    axis.set_xlabel("HIP 报告的驻留上限 (wave/CU)")
    axis.set_ylabel("逻辑读取速率 (GB/s)")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e2e8f0")
    axis.legend(loc="lower right", ncol=2, frameon=False)
    finish(figure, output, "e04-occupancy-ilp.png")


def plot_gemm(archive, output):
    configs = {
        "torch": "rocBLAS", "base": "Triton 基准", "bk16": "仅 BK=16",
        "stages2": "仅 stages=2", "group1": "仅 GROUP_M=1", "historical": "旧 Triton 配置",
    }
    times = {}
    for key in configs:
        result = json.loads(archive.read(f"e09-{key}.json"))
        if result["validation"] != "pass" or result["shape"] != [4096, 4096, 4096]:
            raise ValueError(f"E09 {key} expects validated 4096³ GEMM timings")
        times[key] = median(result["ms_samples"])

    ranking = sorted(configs, key=times.__getitem__)
    figure, axis = plt.subplots(figsize=(5.9, 3.65), layout="constrained")
    colors = {"torch": TEAL, "base": BLUE, "bk16": AMBER}
    horizontal_bars(axis, [configs[key] for key in ranking], [times[key] for key in ranking],
                    [colors.get(key, SLATE) for key in ranking], ".4f", "耗时中位数 (ms)")
    finish(figure, output, "e09-gemm-runtime.png")

    counters = {}
    for key in ("base", "bk16"):
        rows = [row for row in read_csv(archive, f"e09-{key}-pmc.csv")
                if "gemm_kernel" in row["KernelName"]]
        if len(rows) != 6:
            raise ValueError(f"E09 {key} expects six target dispatches")
        counters[key] = [mean(int(row[field]) for row in rows[-5:])
                         for field in ("arch_vgpr", "SQ_WAIT_INST_LDS")]
    reference = [times["base"], *counters["base"]]
    changed = [times["bk16"], *counters["bk16"]]
    deltas = [(new / old - 1) * 100 for old, new in zip(reference, changed)]
    figure, axis = plt.subplots(figsize=(5.9, 2.55), layout="constrained")
    bars = axis.barh(["耗时", "VGPR", "LDS 等待计数"], deltas,
                     height=0.55, color=[AMBER, BLUE, BLUE])
    axis.bar_label(bars, labels=[f"{value:+.1f}%" for value in deltas], padding=5)
    axis.invert_yaxis()
    axis.set_xlim(min(deltas) - 12, max(deltas) + 12)
    axis.axvline(0, color=SLATE, linewidth=1)
    axis.set_xlabel("相对 Triton 基准的变化 (%)")
    axis.grid(axis="x", color="#e2e8f0")
    axis.set_axisbelow(True)
    finish(figure, output, "e09-bk16-deltas.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DATA.with_suffix(".zip"))
    parser.add_argument("--lds-occupancy-archive", type=Path,
                        default=DATA.parent / "bw100-lds-occupancy.zip")
    parser.add_argument("--output", type=Path, default=DATA / "figures")
    parser.add_argument("--font", type=Path, help="Path to a font with Chinese glyphs")
    args = parser.parse_args()
    if args.font:
        font_manager.fontManager.addfont(args.font)
        family = font_manager.FontProperties(fname=args.font).get_name()
    else:
        available = set(font_manager.get_font_names())
        family = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC")
                       if name in available), None)
        if family is None:
            parser.error("Provide a Chinese font with --font")
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"], "font.size": 11.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#94a3b8", "axes.labelcolor": "#1e293b",
        "xtick.color": "#334155", "ytick.color": "#334155", "text.color": "#0f172a",
        "ytick.major.size": 0, "axes.unicode_minus": False,
    })
    args.output.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive:
        plot_copy(archive, args.output)
        plot_latency(archive, args.output)
        plot_gemm(archive, args.output)
    with ZipFile(args.lds_occupancy_archive) as archive:
        plot_lds_conflicts(archive, args.output)
        plot_occupancy(archive, args.output)


if __name__ == "__main__":
    main()
