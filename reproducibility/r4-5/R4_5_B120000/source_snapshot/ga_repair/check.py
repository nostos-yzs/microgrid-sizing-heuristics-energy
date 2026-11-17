# -*- coding: utf-8 -*-
"""
Action effectiveness analysis for GA + SAC hybrid optimizer

分析内容：
1) action_t → reward_{t+1} 的滞后相关性
2) action_t → state_{t+1} 的因果关系（diversity, success_rate, gbest_improve）
3) 自动绘图（相关性、散点图）
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# 配置路径：请按你的真实路径修改
# ================================
CSV_PATH = r"E:/重要桌面搬来/for heuristic solving/energy-based heuristic/energy-based heuristic/pyprj_update_visualize/ga_deg_ini/runs/ga_sac_seed0/iter_log.csv"


# =============================================================================
# 工具：滚动相关性
# =============================================================================
def rolling_corr(a, b, window):
    """Compute rolling correlation manually to avoid NaN issues"""
    corrs = []
    for i in range(len(a) - window):
        seg_a = a[i:i+window]
        seg_b = b[i:i+window]
        if np.std(seg_a) < 1e-12 or np.std(seg_b) < 1e-12:
            corrs.append(0)
            continue
        corrs.append(np.corrcoef(seg_a, seg_b)[0, 1])
    return np.array(corrs)


# =============================================================================
# B：Action → Reward 的滞后相关性分析
# =============================================================================
def analyze_action_reward_lag(df):
    print("\n" + "="*80)
    print("B) ACTION → REWARD (Lag-1 Cross-Correlation)")
    print("="*80)

    # 对齐 t vs t+1
    tp_t = df["treat_prob"][:-1].values
    gf_t = df["gamma_fb"][:-1].values
    reward_next = df["reward"][1:].values

    # Pearson 相关性
    corr_tp = np.corrcoef(tp_t, reward_next)[0, 1]
    corr_gf = np.corrcoef(gf_t, reward_next)[0, 1]

    print(f"treat_prob_t  → reward_(t+1) : {corr_tp:.4f}")
    print(f"gamma_fb_t    → reward_(t+1) : {corr_gf:.4f}")

    # 滚动相关性
    window = max(20, len(df) // 20)
    roll_corr_tp = rolling_corr(tp_t, reward_next, window)
    roll_corr_gf = rolling_corr(gf_t, reward_next, window)

    # ---- 画图 ----
    plt.figure(figsize=(10, 4))
    plt.plot(roll_corr_tp, label="corr(treat_prob_t , reward_{t+1})", color="blue")
    plt.plot(roll_corr_gf, label="corr(gamma_fb_t , reward_{t+1})", color="red")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Rolling Corr")
    plt.title("Lag-1 Rolling Correlation")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# C：Action → State 变化分析
# =============================================================================
def analyze_action_state_causality(df):
    print("\n" + "="*80)
    print("C) ACTION → NEXT-STATE (STATE TRANSITION CAUSALITY)")
    print("="*80)

    # gbest improvement indicator
    gbest = df["gbest"].values
    gbest_improve = (pd.Series(gbest).diff() < 0).astype(int)
    gbest_improve.iloc[0] = 0

    # success rate
    total_fail = (df["cnt_fail_supply"] +
                  df["cnt_fail_overdis"] +
                  df["cnt_fail_overchg"])
    success_rate = df["cnt_success"] / (df["cnt_success"] + total_fail + 1e-9)

    diversity = df["diversity"].values

    # 对齐 t vs t+1
    tp_t = df["treat_prob"][:-1].values
    gf_t = df["gamma_fb"][:-1].values

    div_next = diversity[1:]
    suc_next = success_rate[1:]
    gbest_imp_next = gbest_improve[1:].values

    # 打印相关性
    def safe_corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    print("\n-- treat_prob_t correlations --")
    print("diversity_(t+1)      :", safe_corr(tp_t, div_next))
    print("success_rate_(t+1)   :", safe_corr(tp_t, suc_next))
    print("gbest_improve_(t+1)  :", safe_corr(tp_t, gbest_imp_next))

    print("\n-- gamma_fb_t correlations --")
    print("diversity_(t+1)      :", safe_corr(gf_t, div_next))
    print("success_rate_(t+1)   :", safe_corr(gf_t, suc_next))
    print("gbest_improve_(t+1)  :", safe_corr(gf_t, gbest_imp_next))

    # ---- 绘制 Scatter 图 ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    axes[0].scatter(tp_t, div_next, alpha=0.5)
    axes[0].set_title("treat_prob_t → diversity_{t+1}")

    axes[1].scatter(tp_t, suc_next, alpha=0.5)
    axes[1].set_title("treat_prob_t → success_rate_{t+1}")

    axes[2].scatter(tp_t, gbest_imp_next, alpha=0.5)
    axes[2].set_title("treat_prob_t → gbest_improve_{t+1}")

    axes[3].scatter(gf_t, div_next, alpha=0.5, color="red")
    axes[3].set_title("gamma_fb_t → diversity_{t+1}")

    axes[4].scatter(gf_t, suc_next, alpha=0.5, color="red")
    axes[4].set_title("gamma_fb_t → success_rate_{t+1}")

    axes[5].scatter(gf_t, gbest_imp_next, alpha=0.5, color="red")
    axes[5].set_title("gamma_fb_t → gbest_improve_{t+1}")

    plt.tight_layout()
    plt.show()


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"iter_log.csv not found at:\n{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    analyze_action_reward_lag(df)
    analyze_action_state_causality(df)

    print("\n分析完成。")
