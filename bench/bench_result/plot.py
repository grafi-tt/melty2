#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import textwrap
import yaml


def load_result(f):
    y = yaml.safe_load(f)
    benches = []
    libs = []
    table = []
    first = True
    for bench, d in y.items():
        if not all(isinstance(v, dict) for v in d.values()):
            # a reference without libs, e.g. memset
            continue
        benches.append(bench)
        keys = []
        vals = []
        for k, v in d.items():
            keys.append(k)
            vals.append(v["ns_per_number"])
        if first:
            libs = keys
            first = False
        elif libs != keys:
            raise ValueError("Inconsistent libs")
        table.append(vals)
    return libs, benches, np.transpose(table)


def plot_data(filename, libs, benches, table, title, unit, xlim):
    # unit is "ns" or "ps", and xlim is given in it
    table = table * {"ns": 1, "ps": 1000}[unit]
    dpi = 144
    fig, ax = plt.subplots(figsize=(1280 / dpi, 800 / dpi))
    plt.rcParams["font.family"] = ["M PLUS 2"]
    plt.xticks(fontfamily="M PLUS 2")

    ax.set_ymargin(0.02)
    ax.set_title(title, fontfamily="M PLUS 2")
    ax.set_xlabel(f"Time per 32bit number ({unit}, log scale, shorter is better)", fontfamily="M PLUS 2")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%g"))

    yticks = np.arange(len(benches))
    labels = [textwrap.fill(b, 30) for b in benches]
    ax.set_yticks(yticks, labels, fontfamily="M PLUS Code Latin 60")
    ax.invert_yaxis()
    ax.grid(True, axis="x", which="both", linestyle="dashed")

    tab20c = plt.color_sequences["tab20c"]
    groupheight = 0.75
    for i, lib in enumerate(libs):
        barcenter = (i - 0.5 * (len(libs) - 1)) * groupheight / len(libs)
        barheight = groupheight / len(libs)
        c = {"melty2": 0}.get(lib, 4 * i)
        ax.barh(yticks + barcenter, table[i], height=barheight, label=lib, color=tab20c[c])

    plt.rcParams["font.family"] = ["M PLUS Code Latin 60"]
    ax.legend(fontsize=11)
    fig.tight_layout()
    plt.savefig(filename, dpi=dpi)


def main():
    import matplotlib as mpl
    from pathlib import Path
    mpl.use("Agg")

    here = Path(__file__).resolve().parent
    for name, title, unit, xlim in [
        ("clang", "Benchmark on Ryzen 9 9950X, single thread (Clang 23.1.4)", "ns", (0.1, 2)),
        ("cuda", "Benchmark on GeForce RTX 4070 Laptop GPU (CUDA 12.9)", "ps", (1, 30)),
        ("hip", "Benchmark on Radeon RX 6800 (HIP 7.16)", "ps", (1, 30)),
    ]:
        with open(here / f"{name}.yml") as f:
            libs, benches, table = load_result(f)
        plot_data(here / f"{name}.png", libs, benches, table, title, unit, xlim)


if __name__ == "__main__":
    main()
