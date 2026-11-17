# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:39:39 2025

@author: Lenovo
"""

# sa_grid_search_gamma_diag.py
# 对 SA guided-fixed 的 (gamma_fb, diag_prob) 做网格搜索，并画 LCOE 曲面

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config_sa import get_config
from microgrid_env import MicrogridEnv
from sa_core import initialize_sa_state, run_sa_iteration


# ========= 一些小工具：读典型日 & 读两层退火参数 =========

def load_typical_days_data(data_folder="data"):
    """
    与 main_sa_comparison 里保持一致：
    读取 typicalday0~4.csv，要求包含 ['GHI','Tair','Vwind','Load'] 列
    """
    import pandas as pd

    typical_days_data = []
    required_columns = ["GHI", "Tair", "Vwind", "Load"]
    for i in range(5):
        fp = os.path.join(data_folder, f"typicalday{i}.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"[ERROR] not found: {fp}")
        df = pd.read_csv(fp)
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"[ERROR] missing columns in {fp}: {missing}")
        typical_days_data.append(df)
    print(f"[INFO] loaded {len(typical_days_data)} typical days from '{data_folder}'")
    return typical_days_data


def read_layers(cfg):
    sim = cfg.get("simulation_params", {})
    n_layers = int(sim.get("n_layers", 30))
    inner_loops = int(sim.get("inner_loops", 1))
    return n_layers, inner_loops


# ========= 单个 (gamma_fb, diag_prob) 下跑一遍 guided-fixed SA =========

def run_single_guided_sa(env, config, gamma_fb, diag_prob, seed=42):
    """
    完全仿照 main_sa_comparison.run_sa_guided，只是把 gamma_fb / diag_prob 从参数传入
    返回：best_fitness (LCOE), fitness_history
    """
    import numpy as np

    np.random.seed(seed)

    min_b = config["min_bounds"].astype(float)
    max_b = config["max_bounds"].astype(float)
    sched = config["sa_params"]["schedule"]

    # 初始化 SA 状态
    sa = initialize_sa_state(
        env=env,
        min_b=min_b,
        max_b=max_b,
        schedule=sched,
        seed=seed,
        rw_cfg={"rewards": {}, "penalties": {}},
    )

    # 两层退火参数
    n_layers, inner_loops = read_layers(config)
    T0, alpha, Tmin = float(sched["T0"]), float(sched["alpha"]), float(sched["Tmin"])

    fixed_action = np.array([gamma_fb, diag_prob], dtype=float)
    fit_hist = []

    for layer in range(n_layers):
        sa["T"] = max(T0 * (alpha ** layer), Tmin)  # 外层降温
        for _ in range(inner_loops):
            sa = run_sa_iteration(sa, fixed_action, env)
            fit_hist.append(float(sa["f_best"]))

    return float(sa["f_best"]), fit_hist


# ========= 网格搜索主函数 =========

def main():
    # ---- 1. 载入配置和环境 ----
    config = get_config()

    # 典型日数据（路径和 main_sa_comparison 一致）
    typical_days_data = load_typical_days_data(data_folder="data")
    env = MicrogridEnv(config, typical_days_data)

    min_b = config["min_bounds"].astype(float)
    max_b = config["max_bounds"].astype(float)
    print(f"[INFO] decision bounds: min={min_b}, max={max_b}")

    # ---- 2. 定义网格范围 ----
    # gamma_fb: 0~10，步长 0.1（如果要 0.05，把 GAMMA_STEP 改为 0.05 就行）
    GAMMA_MIN, GAMMA_MAX, GAMMA_STEP = 1.8, 3.8, 0.1
    # diag_prob: 0~1，步长 0.1
    PROB_MIN, PROB_MAX, PROB_STEP = 0.26, 0.46, 0.01

    gamma_values = np.arange(GAMMA_MIN, GAMMA_MAX + 1e-8, GAMMA_STEP)
    prob_values = np.arange(PROB_MIN, PROB_MAX + 1e-8, PROB_STEP)

    n_gamma = len(gamma_values)
    n_prob = len(prob_values)
    print(f"[INFO] gamma grid size = {n_gamma}, prob grid size = {n_prob}, total = {n_gamma * n_prob}")

    # 存 LCOE（best_fitness），形状 [n_prob, n_gamma]
    # 约定：行索引 = prob，列索引 = gamma
    lcoe_grid = np.full((n_prob, n_gamma), np.nan, dtype=float)

    # ---- 3. 双层 loop 网格搜索 ----
    base_seed = int(config.get("random_seed", 42))

    for i_p, prob in enumerate(prob_values):
        print(f"[ROW {i_p+1}/{n_prob}] diag_prob = {prob:.3f}")
        for j_g, gamma in enumerate(gamma_values):
            # 为了可复现，可以让 seed 跟网格点相关，这样每个点的随机性固定
            seed = base_seed 

            best_f, _ = run_single_guided_sa(
                env=env,
                config=config,
                gamma_fb=gamma,
                diag_prob=prob,
                seed=seed,
            )
            lcoe_grid[i_p, j_g] = best_f
            
    # ====== 找到 LCOE 最小的参数组合 ======
    min_idx = np.unravel_index(np.nanargmin(lcoe_grid), lcoe_grid.shape)
    best_prob = prob_values[min_idx[0]]
    best_gamma = gamma_values[min_idx[1]]
    best_lcoe = lcoe_grid[min_idx]

    print("\n================ GRID SEARCH BEST RESULT ================")
    print(f"Best LCOE = {best_lcoe:.9f}")
    print(f"Best gamma_fb  = {best_gamma:.3f}")
    print(f"Best diag_prob = {best_prob:.3f}")
    print("==========================================================\n")
    
    
    # 可选：保存成 CSV 方便后续分析
    out_csv = "sa_gamma_diag_lcoe_grid.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("gamma_fb,diag_prob,lcoe\n")
        for i_p, prob in enumerate(prob_values):
            for j_g, gamma in enumerate(gamma_values):
                f.write(f"{gamma:.6f},{prob:.6f},{lcoe_grid[i_p, j_g]:.9f}\n")
    print(f"[INFO] grid results saved to {out_csv}")
    
        # ---- 4. 画 3D 曲面图 ----
        # ====== 找到 LCOE 最小的参数组合 ======
    min_idx = np.unravel_index(np.nanargmin(lcoe_grid), lcoe_grid.shape)
    best_prob = prob_values[min_idx[0]]
    best_gamma = gamma_values[min_idx[1]]
    best_lcoe = lcoe_grid[min_idx]
    
    print("\n================ GRID SEARCH BEST RESULT ================")
    print(f"Best LCOE = {best_lcoe:.9f}")
    print(f"Best gamma_fb  = {best_gamma:.3f}")
    print(f"Best diag_prob = {best_prob:.3f}")
    print("==========================================================\n")
    
    # ====== 定义“不可行点”的判定，并构造用于绘图的 Z ======

    
    # ====== 定义不可行阈值 ======
    penalty_cutoff = 1e5

    # ====== 将 3 个数组打平，准备过滤 ======
    G, P = np.meshgrid(gamma_values, prob_values)

    G_flat = G.ravel()               # gamma_fb
    P_flat = P.ravel()               # diag_prob
    L_flat = lcoe_grid.ravel()       # 对应 LCOE

    # ====== 可行性掩码（LCOE < penalty_cutoff）=====
    feasible_mask = L_flat < penalty_cutoff

    # ====== 过滤掉不可行点（只保留可行点）=====
    G_f = G_flat[feasible_mask]
    P_f = P_flat[feasible_mask]
    L_f = L_flat[feasible_mask]

    print(f"[INFO] feasible points kept = {len(L_f)} / {len(L_flat)}")

    # ====== 找出最优点（可行点中 LCOE 最小）=====
    if len(L_f) == 0:
        raise RuntimeError("No feasible SA solutions found in grid search!")

    best_idx = np.argmin(L_f)
    best_gamma = G_f[best_idx]
    best_prob = P_f[best_idx]
    best_lcoe = L_f[best_idx]

    print("\n================ GRID SEARCH BEST RESULT ================")
    print(f"Best LCOE = {best_lcoe:.9f}")
    print(f"Best gamma_fb  = {best_gamma:.3f}")
    print(f"Best diag_prob = {best_prob:.3f}")
    print("==========================================================\n")

    # ====== 保存 CSV（只保存可行点）=====
    out_csv = "sa_gamma_diag_lcoe_feasible_only.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("gamma_fb,diag_prob,lcoe\n")
        for g, p, l in zip(G_f, P_f, L_f):
            f.write(f"{g:.6f},{p:.6f},{l:.12f}\n")
    print(f"[INFO] feasible-only grid saved to {out_csv}")

    # ====== 2D 平面图（无不可行点，仅可行点 + 最优点）=====
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

    # 你之前说不想要颜色条，这里就注释掉
    # plt.colorbar(sc, ax=ax, label="Best LCOE")

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("sa_gamma_diag_feasible_2d.png", dpi=300, bbox_inches="tight")
    print("[INFO] figure saved to sa_gamma_diag_feasible_2d.png")

    plt.show()



if __name__ == "__main__":
    main()
