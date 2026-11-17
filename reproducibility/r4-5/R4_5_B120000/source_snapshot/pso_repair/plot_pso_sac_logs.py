# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 12:40:34 2025

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
PSO-SAC 训练过程可视化：
1) Reward (左轴) + treat_prob / gamma_fb (右轴)
2) LCOE (左轴) + 四个状态 (右轴，0-1 归一化)
"""
 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 根据你实际的运行目录修改这里：
RUN_DIR = r"./runs_pso_sac/sac_pso_seed0"
OUT_DIR = RUN_DIR


def _set_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"


def _normalize(series):
    """Normalize to 0-1, avoid divide-by-zero"""
    s = series.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def plot_reward_and_actions(csv_path=None, out_dir=None):
    """
    图1：Reward (左轴) + treat_prob / gamma_fb 的滚动平均(右轴)
    """
    if csv_path is None:
        csv_path = os.path.join(RUN_DIR, "iter_log.csv")
    if out_dir is None:
        out_dir = OUT_DIR

    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    _set_style()
    fig, ax_left = plt.subplots(figsize=(6, 4))

    it = df["iter"].astype(int)

    # ---------------- 左轴：reward ----------------
    reward = df["reward"].astype(float)
    ax_left.plot(
        it, reward,
        color="tab:orange", linewidth=1.2, label="Reward"
    )
    ax_left.set_xlabel("Iteration")
    ax_left.set_ylabel("Reward", color="black")
    ax_left.tick_params(axis="y", labelcolor="black")
    ax_left.grid(False)

    r_min, r_max = reward.min(), reward.max()
    if r_max > r_min:
        ax_left.set_ylim(r_min - (r_max - r_min) * 0.2,
                         r_max + (r_max - r_min) * 0.1)

    # ---------------- 右轴：treat_prob + gamma_fb 的 rolling mean ----------------
    ax_right = ax_left.twinx()

    tp_raw = df["treat_prob"].astype(float)*10
    gf_raw = df["gamma_fb"].astype(float)

    window = 3  # 你可以调大，比如 10，看得更平滑
    tp = tp_raw.rolling(window, center=True).mean().fillna(tp_raw)
    gf = gf_raw.rolling(window, center=True).mean().fillna(gf_raw)

    ax_right.plot(it, tp, linewidth=2.0, label="Possibility",color="tab:blue")
    ax_right.plot(it, gf, linewidth=2.0, label="Strength",color="tab:red")

    ax_right.set_ylabel("Possibility*10 / Strength", color="black")
    ax_right.tick_params(axis="y", labelcolor="black")
    ax_right.grid(False)

    # 分层显示，避免重叠
    tp_min, tp_max = tp.min(), tp.max()
    gf_min, gf_max = gf.min(), gf.max()
    right_span = max(gf_max - gf_min, 1e-6)
    right_min = min(tp_min, gf_min) - right_span * 0.2
    right_max = max(tp_max, gf_max) + right_span * 0.9
    ax_right.set_ylim(right_min, right_max)

    # legend 合并
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        frameon=False,
        ncol=3
    )

    fig.tight_layout()

    pdf_path = os.path.join(out_dir, "fig_pso_sac_reward_actions.pdf")
    png_path = os.path.join(out_dir, "fig_pso_sac_reward_actions.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.show(fig)

    print("[saved]", pdf_path)
    print("[saved]", png_path)


def plot_lcoe_and_states(csv_path=None, out_dir=None):
    """
    图2：LCOE (左轴) + state0..3 (右轴，归一化到 0-1)
    """
    if csv_path is None:
        csv_path = os.path.join(RUN_DIR, "iter_log.csv")
    if out_dir is None:
        out_dir = OUT_DIR

    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    _set_style()
    fig, ax_left = plt.subplots(figsize=(6, 4))

    it = df["iter"].astype(int)

    # ---------------- 左轴：LCOE ----------------
    lcoe = df["lcoe"].astype(float)
    ax_left.plot(it, lcoe, linewidth=1.5, label="LCOE")
    ax_left.set_xlabel("Iteration")
    ax_left.set_ylabel("LCOE", color="black")
    ax_left.tick_params(axis="y", labelcolor="black")
    ax_left.grid(False)

    l_min, l_max = lcoe.min(), lcoe.max()
    if l_max > l_min:
        ax_left.set_ylim(l_min - (l_max - l_min) * 0.05,
                         l_max + (l_max - l_min) * 0.2)

    # ---------------- 右轴：四个 state，归一化 ----------------
    ax_right = ax_left.twinx()

    state_cols = ["state0", "state1", "state2", "state3"]
    for idx, col in enumerate(state_cols):
        s = df[col].astype(float)
        s_norm = _normalize(s)
        ax_right.plot(it, s_norm, linewidth=1.2, label=f"{col} (norm)")

    ax_right.set_ylabel("Normalized states", color="black")
    ax_right.tick_params(axis="y", labelcolor="black")
    ax_right.grid(False)
    ax_right.set_ylim(-0.05, 1.05)

    # legend 合并
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        frameon=False,
        ncol=2
    )

    fig.tight_layout()

    pdf_path = os.path.join(out_dir, "fig_pso_sac_lcoe_states.pdf")
    png_path = os.path.join(out_dir, "fig_pso_sac_lcoe_states.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.show(fig)

    print("[saved]", pdf_path)
    print("[saved]", png_path)


if __name__ == "__main__":
    plot_reward_and_actions()
    # plot_lcoe_and_states()
