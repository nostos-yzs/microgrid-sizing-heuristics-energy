# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:35:11 2025

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
ts_grid_search_gamma_diag.py
对 Classic Guided TS 的 (diag_prob, gamma_feedback) 做网格搜索，并画 LCOE 面/平面图

使用：
    python ts_grid_search_gamma_diag.py

依赖：
    - config_tabu.get_config
    - microgrid_env.MicrogridEnv
    - ts_core.initialize_tabu_state, ts_core.run_tabu_iteration_guided
    - data/typicalday0~4.csv
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config_tabu import get_config
from microgrid_env import MicrogridEnv
import ts_core


# ========= 工具：读典型日 =========

def load_typical_days_data(data_folder="data"):
    """
    与 main_tabu_comparison 里保持一致：
    读取 typicalday0~4.csv，要求包含 ['GHI','Tair','Vwind','Load'] 列
    """
    import pandas as pd

    typical_days_data = []
    required = ['GHI', 'Tair', 'Vwind', 'Load']
    for i in range(5):
        fp = os.path.join(data_folder, f"typicalday{i}.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"[ERROR] not found: {fp}")
        df = pd.read_csv(fp)
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"[ERROR] missing columns in {fp}: {miss}")
        typical_days_data.append(df)
    print(f"[INFO] loaded {len(typical_days_data)} typical days from '{data_folder}'")
    return typical_days_data


# ========= 单个 (diag_prob, gamma_fb) 下跑一遍 Classic Guided TS =========

def run_single_guided_ts(env, config, diag_prob, gamma_fb, total_iterations, seed=42):
    """
    仿照 classic_tabu_guided.run_classic_tabu_guided，只是把 diag_prob / gamma_fb 从参数传入
    返回：best_fitness (LCOE), fitness_history
    """
    import random

    np.random.seed(seed)
    random.seed(seed)

    # 固定动作向量：action_vec = [diag_prob, gamma_feedback]
    action_vec = [float(diag_prob), float(gamma_fb)]

    state = ts_core.initialize_tabu_state(config, env, seed=seed)
    fit_hist = [state["best_f"]]

    for _ in range(total_iterations):
        state = ts_core.run_tabu_iteration_guided(state, action_vec, env)
        fit_hist.append(state["best_f"])

    return float(state["best_f"]), fit_hist


# ========= 网格搜索主函数 =========

def main():
    # ---- 1. 载入配置和环境 ----
    config = get_config()

    typical_days_data = load_typical_days_data(data_folder="data")
    env = MicrogridEnv(config, typical_days_data)

    ts_cfg = config.get("ts_params", {})
    min_b = np.array(ts_cfg.get("min_bounds", np.zeros(6)), float)
    max_b = np.array(ts_cfg.get("max_bounds", np.ones(6)), float)
    print(f"[INFO] decision bounds: min={min_b}, max={max_b}")

    sim_cfg = config.get("simulation_params", {})
    total_iterations = int(sim_cfg.get("total_iterations", 200))
    print(f"[INFO] total_iterations per run = {total_iterations}")

    # ---- 2. 定义网格范围 ----
    # 建议范围：
    #   diag_prob: 0.2 ~ 0.9，步长 0.1
    #   gamma_fb : 10  ~ 80，步长 10
    #
    # 你可以根据需要调整这些值
    PROB_MIN, PROB_MAX, PROB_STEP = 0.60, 0.99, 0.01
    GAMMA_MIN, GAMMA_MAX, GAMMA_STEP = 40, 80, 0.5

    prob_values = np.arange(PROB_MIN, PROB_MAX + 1e-8, PROB_STEP)
    gamma_values = np.arange(GAMMA_MIN, GAMMA_MAX + 1e-8, GAMMA_STEP)

    n_prob = len(prob_values)
    n_gamma = len(gamma_values)
    print(f"[INFO] prob grid size = {n_prob}, gamma grid size = {n_gamma}, total = {n_prob * n_gamma}")

    # 存 LCOE（best_fitness），形状 [n_prob, n_gamma]
    # 约定：行索引 = prob，列索引 = gamma
    lcoe_grid = np.full((n_prob, n_gamma), np.nan, dtype=float)

    base_seed = int(config.get("random_seed", 42))

    # ---- 3. 双层 loop 网格搜索 ----
    for i_p, prob in enumerate(prob_values):
        print(f"[ROW {i_p+1}/{n_prob}] diag_prob = {prob:.3f}")
        for j_g, gamma in enumerate(gamma_values):
            seed = base_seed  # 如果想每个点 seed 不同，可以加上 i_p, j_g

            best_f, _ = run_single_guided_ts(
                env=env,
                config=config,
                diag_prob=prob,
                gamma_fb=gamma,
                total_iterations=total_iterations,
                seed=seed,
            )
            lcoe_grid[i_p, j_g] = best_f

    # ====== 初步全局最优（不区分可行/不可行，只看最小值）=====
    min_idx = np.unravel_index(np.nanargmin(lcoe_grid), lcoe_grid.shape)
    best_prob_raw = prob_values[min_idx[0]]
    best_gamma_raw = gamma_values[min_idx[1]]
    best_lcoe_raw = lcoe_grid[min_idx]

    print("\n========= RAW GRID SEARCH RESULT (may include penalized) =========")
    print(f"Raw best LCOE = {best_lcoe_raw:.9f}")
    print(f"Raw best gamma_fb  = {best_gamma_raw:.3f}")
    print(f"Raw best diag_prob = {best_prob_raw:.3f}")
    print("=================================================================\n")

    # 可选：保存所有点（含不可行）成 CSV
    out_csv_all = "ts_gamma_diag_lcoe_grid.csv"
    with open(out_csv_all, "w", encoding="utf-8") as f:
        f.write("gamma_fb,diag_prob,lcoe\n")
        for i_p, prob in enumerate(prob_values):
            for j_g, gamma in enumerate(gamma_values):
                f.write(f"{gamma:.6f},{prob:.6f},{lcoe_grid[i_p, j_g]:.9f}\n")
    print(f"[INFO] full grid results saved to {out_csv_all}")

    # ====== 4. 只看“可行点”的结果 ======
    # 判定可行阈值：LCOE < penalty_cutoff
    penalty_cutoff = 1e5

    G, P = np.meshgrid(gamma_values, prob_values)
    G_flat = G.ravel()
    P_flat = P.ravel()
    L_flat = lcoe_grid.ravel()

    feasible_mask = L_flat < penalty_cutoff
    G_f = G_flat[feasible_mask]
    P_f = P_flat[feasible_mask]
    L_f = L_flat[feasible_mask]

    print(f"[INFO] feasible points kept = {len(L_f)} / {len(L_flat)}")

    if len(L_f) == 0:
        raise RuntimeError("No feasible TS solutions found in grid search!")

    best_idx = np.argmin(L_f)
    best_gamma = G_f[best_idx]
    best_prob = P_f[best_idx]
    best_lcoe = L_f[best_idx]

    print("\n================ FEASIBLE-ONLY GRID SEARCH BEST RESULT ================")
    print(f"Best LCOE = {best_lcoe:.9f}")
    print(f"Best gamma_fb  = {best_gamma:.3f}")
    print(f"Best diag_prob = {best_prob:.3f}")
    print("=======================================================================\n")

    # ====== 保存可行点 CSV ======
    out_csv_feasible = "ts_gamma_diag_lcoe_feasible_only.csv"
    with open(out_csv_feasible, "w", encoding="utf-8") as f:
        f.write("gamma_fb,diag_prob,lcoe\n")
        for g, p, l in zip(G_f, P_f, L_f):
            f.write(f"{g:.6f},{p:.6f},{l:.12f}\n")
    print(f"[INFO] feasible-only grid saved to {out_csv_feasible}")

    # ====== 2D 平面图（仅可行点 + 最优点）=====
    fig, ax = plt.subplots(figsize=(7, 5))

    sc = ax.scatter(
        G_f, P_f,
        c=L_f,
        s=35,
        cmap="viridis",
        edgecolors="none",
    )

    # 最优点：红色星号
    ax.scatter(
        best_gamma,
        best_prob,
        marker="*",
        s=160,
        color="red",
        label="Best",
    )

    ax.set_xlabel(r"$\gamma_{\mathrm{fb}}$")
    ax.set_ylabel(r"$p_{\mathrm{diag}}$")

    # 按你之前 SA 的习惯，先不画颜色条
    # plt.colorbar(sc, ax=ax, label="Best LCOE")

    ax.legend(loc="upper right")

    plt.tight_layout()
    fig_name = "ts_gamma_diag_feasible_2d.png"
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    print(f"[INFO] figure saved to {fig_name}")

    plt.show()


if __name__ == "__main__":
    main()
