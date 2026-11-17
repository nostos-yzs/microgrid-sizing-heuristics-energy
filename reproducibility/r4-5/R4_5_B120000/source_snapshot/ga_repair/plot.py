# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 19:24:30 2025

@author: Lenovo
"""

# plot_training.py
import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _load_iter_log(run_dir):
    p = os.path.join(run_dir, "iter_log.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"not found: {p}. 请先跑一遍训练生成日志。")
    df = pd.read_csv(p)
    # 容错：老版本列名兼容
    for c in ["treat_prob","gamma_fb","gbest","reward","diversity",
              "cnt_success","cnt_fail_supply","cnt_fail_overdis","cnt_fail_overchg"]:
        if c not in df.columns: df[c] = np.nan
    return df


# 统一画图规范：TimesNR + 20pt
def _set_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20


def plot_actions(df, outdir):
    _set_plot_style()
    plt.figure(figsize=(6, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(df["iter"], df["treat_prob"], color="tab:blue", label="treat_prob")
    ax2.plot(df["iter"], df["gamma_fb"], color="tab:red", label="gamma_fb")

    ax1.set_xlabel("iteration")
    ax1.set_ylabel("treat_prob", color="tab:blue")
    ax2.set_ylabel("gamma_fb", color="tab:red")

    ax1.grid(False)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_actions.pdf"))
    plt.savefig(os.path.join(outdir, "fig_actions.png"))
    plt.close()


def plot_convergence(df, outdir):
    _set_plot_style()
    plt.figure(figsize=(6, 4))
    y = df["gbest"].replace([np.inf, -np.inf], np.nan).dropna()
    x = df.loc[y.index, "iter"]
    plt.plot(x, y, color="tab:blue")
    plt.xlabel("iteration")
    plt.ylabel("best fitness")
    if y.max() / max(1e-9, y.min()) > 100:
        plt.yscale("log")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_convergence.pdf"))
    plt.savefig(os.path.join(outdir, "fig_convergence.png"))
    plt.close()


def plot_reward(df, outdir):
    _set_plot_style()
    plt.figure(figsize=(6, 4))
    y = df["reward"].astype(float)
    plt.plot(df["iter"], y, alpha=0.4, label="reward", color="tab:orange")
    w = max(1, len(y) // 30)
    roll = y.rolling(window=w, min_periods=1).mean()
    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.legend(loc="upper right")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_reward.pdf"))
    plt.savefig(os.path.join(outdir, "fig_reward.png"))
    plt.close()


def plot_diversity(df, outdir):
    _set_plot_style()
    plt.figure(figsize=(6, 4))
    plt.plot(df["iter"], df["diversity"], color="tab:green")
    plt.xlabel("iteration")
    plt.ylabel("population diversity (mean σ)")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_diversity.pdf"))
    plt.savefig(os.path.join(outdir, "fig_diversity.png"))
    plt.close()


def plot_diag_share(df, outdir):
    _set_plot_style()

    for c in ["cnt_success", "cnt_fail_supply", "cnt_fail_overdis", "cnt_fail_overchg"]:
        if c not in df.columns:
            df[c] = 0
    total = (
        df["cnt_success"]
        + df["cnt_fail_supply"]
        + df["cnt_fail_overdis"]
        + df["cnt_fail_overchg"]
    ).replace(0, np.nan)
    share = pd.DataFrame(
        {
            "iter": df["iter"],
            "success": df["cnt_success"] / total,
            "fail_supply": df["cnt_fail_supply"] / total,
            "fail_overdis": df["cnt_fail_overdis"] / total,
            "fail_overchg": df["cnt_fail_overchg"] / total,
        }
    ).fillna(0.0)

    plt.figure(figsize=(6, 4))
    plt.stackplot(
        share["iter"],
        share["success"],
        share["fail_supply"],
        share["fail_overdis"],
        share["fail_overchg"],
        labels=["success", "fail_supply", "fail_overdis", "fail_overchg"],
    )
    plt.xlabel("iteration")
    plt.ylabel("diagnosis share")
    plt.legend(loc="upper right")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_diag_share.pdf"))
    plt.savefig(os.path.join(outdir, "fig_diag_share.png"))
    plt.close()


def plot_one_trace_csv(trace_csv, outdir):
    _set_plot_style()
    df = pd.read_csv(trace_csv)
    itag = os.path.splitext(os.path.basename(trace_csv))[0].replace("trace_","")

    t = df["t"]

    plt.figure(figsize=(6,4))
    plt.plot(t, df["p_ec"], label="p_ec")
    plt.plot(t, df["p_fc"], label="p_fc")
    plt.plot(t, df["p_ch"], label="p_ch")
    plt.plot(t, df["p_dis"], label="p_dis")
    plt.plot(t, df["P_RES"]-df["Load"], label="net(P_RES-Load)", linestyle=":")
    plt.xlabel("hour"); plt.ylabel("power")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(os.path.join(outdir, f"fig_trace_power_{itag}.pdf"))
    plt.savefig(os.path.join(outdir, f"fig_trace_power_{itag}.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(t, df["SOC"], label="SOC")
    plt.xlabel("hour"); plt.ylabel("SOC")
    plt.legend(loc="upper right")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig_trace_soc_{itag}.pdf"))
    plt.savefig(os.path.join(outdir, f"fig_trace_soc_{itag}.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(t, df["H2"], label="H2_storage")
    plt.xlabel("hour"); plt.ylabel("H2 (kg)")
    plt.legend(loc="upper right")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig_trace_h2_{itag}.pdf"))
    plt.savefig(os.path.join(outdir, f"fig_trace_h2_{itag}.png"))
    plt.close()


### main() 完全保留
def main(run_dir):
    outdir = _ensure_dir(run_dir)
    df = _load_iter_log(run_dir)

    plot_actions(df, outdir)
    plot_convergence(df, outdir)
    plot_reward(df, outdir)
    plot_diversity(df, outdir)
    plot_diag_share(df, outdir)

    trace_dir = os.path.join(run_dir, "traces")
    if os.path.isdir(trace_dir):
        traces = sorted(glob.glob(os.path.join(trace_dir, "trace_*.csv")))
        if traces:
            plot_one_trace_csv(traces[-1], outdir)


if __name__ == "__main__":
    import glob, os, time
    if len(sys.argv) >= 2:
        run_dir = sys.argv[1]
        main(run_dir)
    else:
        candidates = glob.glob(os.path.join("runs", "ga_sac_seed*"))
        if not candidates:
            print("未找到 runs/ga_sac_seed* 目录。请先运行训练，或手动执行：python plot_training.py runs/ga_sac_seed0")
            sys.exit(1)
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        auto_dir = candidates[0]
        print(f"[info] 未提供参数，自动使用最近目录：{auto_dir}")
        main(auto_dir)
