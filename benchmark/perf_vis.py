"""
Some very quick code to plot benchmarks for the paper.
"""

import os
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import pandas as pd

DATA_DIR = "/home/ubuntu/latte/data/latte_misc"


def abblation_T_time():
    with open(os.path.join(DATA_DIR, "benchm_128.json"), "r+") as f:
        perf = json.load(f, object_pairs_hook=OrderedDict)

    ticks = list(perf.keys())
    ticks = [int(t) for t in ticks]
    timings = {
        "causal_att": [],
        "stable_scan": [],
        "mem_ineff": [],
        "goog_lin": [],
        "bid_latte": [],
    }
    for k, t in perf.items():
        timings["causal_att"].append(t["self_attention"])
        timings["stable_scan"].append(t["stable_scan"])
        timings["mem_ineff"].append(t["mem_ineff"])
        timings["bid_latte"].append(t["bid_latte"])
        timings["goog_lin"].append(t["goog_lin"])

    plt.plot(
        ticks,
        timings["causal_att"],
        marker="8",
        label="Standard Causal Attention",
        linestyle="--",
    )
    plt.plot(
        ticks,
        timings["goog_lin"],
        marker="s",
        label="Standard Causal Attention (Sequential)",
        linestyle="--",
    )
    plt.plot(
        ticks, timings["stable_scan"], marker="x", label="Causal Latte", linestyle="--"
    )

    plt.plot(
        ticks,
        timings["bid_latte"],
        marker="*",
        label="Bidirectional Latte",
        linestyle="--",
    )

    plt.legend()
    plt.xticks(ticks)
    plt.xlabel("Sequence length")
    plt.ylabel("Time (ms)")
    plt.savefig(os.path.join(DATA_DIR, "perf_time_128.png"))


def abblation_L_time():
    with open(os.path.join(DATA_DIR, "benchm_L_512.json"), "r+") as f:
        perf = json.load(f, object_pairs_hook=OrderedDict)

    causal_att = perf.pop("causal_att")
    google_mem = perf.pop("google_mem")
    print(perf)
    ticks = list(perf["causal_latte"].keys())

    plt.axhline(y=causal_att, color="r", linestyle="--", label="SCA")
    plt.axhline(y=google_mem, color="g", linestyle="--", label="LMCA")

    plt.plot(
        ticks,
        list(perf["causal_latte"].values()),
        marker="*",
        label="CL",
        linestyle="--",
    )

    plt.plot(
        ticks,
        list(perf["bid_latte"].values()),
        marker="x",
        label="BL",
        linestyle="--",
    )

    # plt.legend()
    plt.legend(loc="center left", bbox_to_anchor=(0, 0.7))
    plt.xticks(ticks)
    plt.xlabel("Nr. Latent Variables")
    plt.ylabel("Time (ms)")
    plt.savefig(os.path.join(DATA_DIR, "perf_time_512_L.png"))


def plot_perf_hidden():
    with open("../benchm_hidd_5000.json", "r+") as f:
        perf = json.load(f, object_pairs_hook=OrderedDict)

    ticks = list(perf.keys())
    ticks = [int(t) for t in ticks]
    timings = {"causal_att": [], "stable_scan": [], "chnk_stable_scan": []}
    for k, t in perf.items():
        timings["causal_att"].append(t["self_attention"])
        timings["stable_scan"].append(t["stable_scan"])
        timings["chnk_stable_scan"].append(t["chk_scan"])

    plt.plot(
        ticks, timings["causal_att"], marker="8", label="causal_att", linestyle="--"
    )
    plt.plot(
        ticks, timings["stable_scan"], marker="x", label="stable_scan", linestyle="--"
    )
    plt.plot(
        ticks,
        timings["chnk_stable_scan"],
        marker="X",
        label="chnk_stable_scan",
        linestyle="--",
    )
    plt.legend()
    plt.xticks(ticks)
    plt.xlabel("Hidden dimension")
    plt.ylabel("Time(ms)")
    plt.title("Performance for seq. len 5000")
    plt.savefig(os.path.join(DATA_DIR, "hidden_perf_5000.png"))


def plot_mem():
    def _clean(x):
        x = x.replace("MiB", "")
        return float(x)

    data_path = os.path.join(DATA_DIR, "mem_bench_128_jit.csv")
    df = pd.read_csv(data_path)
    models = df["Model"]
    df = df.drop(columns=["Model"])
    df = df.applymap(_clean)
    # drop T=
    ticks = [int(x[2:]) for x in df.columns]

    plt.plot(
        ticks, df.iloc[0], marker="8", label="Standard Causal Attention", linestyle="--"
    )
    plt.plot(
        ticks,
        df.iloc[1],
        marker="s",
        label="Standard Causal Attention (Sequential)",
        linestyle="--",
    )
    plt.plot(ticks, df.iloc[2], marker="x", label="Causal Latte", linestyle="--")
    plt.plot(ticks, df.iloc[3], marker="*", label="Bidirectional Latte", linestyle="--")

    plt.legend()
    plt.xticks(ticks)
    plt.xlabel("Sequence length")
    plt.ylabel("Memory (MiB)")
    plt.savefig(os.path.join(DATA_DIR, "mem_128_mebibyte.png"))


if __name__ == "__main__":
    plot_mem()
    # abblation_T_time()
    # abblation_L_time()
    # plot_perf_hidden()
