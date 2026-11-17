# -*- coding: utf-8 -*-
"""
Reward (bottom) + treat_prob (middle) + gamma_fb (top)
All axes black, legend non-overlapping, curves visually separated.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


RUN_DIR = r"E:/重要桌面搬来/for heuristic solving/energy-based heuristic/energy-based heuristic/pyprj_update_visualize/ga_deg_ini/runs/ga_sac_seed0"
OUT_DIR = r"E:/重要桌面搬来/for heuristic solving/energy-based heuristic/energy-based heuristic/pyprj_update_visualize/ga_deg_ini"


def _set_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"


def _normalize(series):
    """Normalize to 0-1, avoid divide-by-zero"""
    s = series.astype(float)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s


def plot_reward_joint():
    csv_path = os.path.join(RUN_DIR, "iter_log.csv")
    df = pd.read_csv(csv_path)
    os.makedirs(OUT_DIR, exist_ok=True)

    _set_style()
    fig, ax_left = plt.subplots(figsize=(6, 4))

    it = df["iter"]

    # ---------------- 左轴 reward ----------------
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
    ax_left.set_ylim(r_min - (r_max - r_min) * 0.05,
                     r_max + (r_max - r_min) * 0.7)

    # ---------------- 右轴 treat_prob + gamma_fb（Rolling Mean） ----------------
    ax_right = ax_left.twinx()

    # 原始值
    tp_raw = df["treat_prob"].astype(float)
    gf_raw = df["gamma_fb"].astype(float)

    # Rolling mean（窗口=20，可改）
    window = 3
    tp = tp_raw.rolling(window, center=True).mean()
    gf = gf_raw.rolling(window, center=True).mean()

    # 注意：rolling 前面会有 NaN → 用原始值补齐
    tp = tp.fillna(tp_raw)
    gf = gf.fillna(gf_raw)

    # 绘图（Rolling Mean）
    ax_right.plot(it, tp, color="tab:blue", linewidth=2.0, label="Possibility")
    ax_right.plot(it, gf, color="tab:red", linewidth=2.0, label="Strength")

    ax_right.set_ylabel("Possibility / Strength", color="black")
    ax_right.tick_params(axis="y", labelcolor="black")
    ax_right.grid(False)

    # --- 分层显示，避免重叠 ---
    tp_min, tp_max = tp.min(), tp.max()
    gf_min, gf_max = gf.min(), gf.max()

    right_min = min(tp_min, gf_min) - (gf_max - gf_min) * 0.8
    right_max = gf_max + (gf_max - gf_min) * 0.4
    ax_right.set_ylim(right_min, right_max)

    # ---------------- legend ----------------
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

    # ---------------- 保存 ----------------
    pdf_path = os.path.join(OUT_DIR, "fig_reward_joint.pdf")
    png_path = os.path.join(OUT_DIR, "fig_reward_joint.png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.show(fig)

    print("[saved]", pdf_path)
    print("[saved]", png_path)


if __name__ == "__main__":
    plot_reward_joint()
