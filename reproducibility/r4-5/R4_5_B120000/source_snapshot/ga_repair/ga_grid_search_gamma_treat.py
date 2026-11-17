# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:53:59 2025

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
ga_grid_search_gamma_treat.py
对 Classic Guided GA 的 (parent_treat_prob, parent_treat_strength=gamma_feedback)
做网格搜索，绘制 LCOE 平面图

使用：
    python ga_grid_search_gamma_treat.py

依赖：
    - config_ga.get_config
    - microgrid_env.MicrogridEnv
    - classic_ga_guided.run_classic_ga_guided
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config_ga import get_config
from microgrid_env import MicrogridEnv
from classic_ga_guided import run_classic_ga_guided


# ========== 工具 ==========
def load_typical_days_data(data_folder="data"):
    import pandas as pd
    typical_days_data = []
    required = ["GHI", "Tair", "Vwind", "Load"]

    for i in range(5):
        fp = os.path.join(data_folder, f"typicalday{i}.csv")
        df = pd.read_csv(fp)
        for k in required:
            if k not in df.columns:
                raise ValueError(f"Missing {k} in {fp}")
        typical_days_data.append(df)

    print(f"[INFO] Loaded {len(typical_days_data)} typical days.")
    return typical_days_data


# ========== 单个组合下运行一次 Guided GA ==========
def run_single_ga(env, config, treat_prob, gamma_fb, seed=0):
    """
    固定 parent_treat_prob 和 parent_treat_strength 运行一次 Classic Guided GA。
    """
    import numpy as np
    np.random.seed(seed)

    # 修改 config，使 GA 采用指定参数
    cfg = config
    cfg["ga_params"]["classic_params"]["guided"]["parent_treat_prob"] = treat_prob
    cfg["ga_params"]["classic_params"]["guided"]["parent_treat_strength"] = gamma_fb

    total_iters = cfg["simulation_params"]["total_iterations"]

    res = run_classic_ga_guided(
        cfg, env,
        total_iterations=total_iters,
        run_seed=seed
    )

    return float(res["best_fitness"]), res["fitness_history"]


# ========== 网格搜索主函数 ==========
def main():
    config = get_config()
    total_iters = config["simulation_params"]["total_iterations"]

    typical_days = load_typical_days_data("data")
    env = MicrogridEnv(config, typical_days)

    # 定义网格 (你可改范围)
    TREAT_MIN, TREAT_MAX, TREAT_STEP = 0.01, 0.21, 0.01
    GAMMA_MIN, GAMMA_MAX, GAMMA_STEP = 0.1, 3, 0.1

    treat_values = np.arange(TREAT_MIN, TREAT_MAX + 1e-9, TREAT_STEP)
    gamma_values = np.arange(GAMMA_MIN, GAMMA_MAX + 1e-9, GAMMA_STEP)

    P = len(treat_values)
    G = len(gamma_values)
    print(f"[INFO] Grid size: {P} x {G} = {P*G} runs")

    # 存 best LCOE
    lcoe_grid = np.full((P, G), np.nan)

    base_seed = 42

    # --------------- 双重循环执行 ----------------
    for i, prob in enumerate(treat_values):
        print(f"[ROW {i+1}/{P}] treat_prob = {prob:.2f}")
        for j, gamma in enumerate(gamma_values):
            best_f, _ = run_single_ga(
                env,
                config,
                treat_prob=prob,
                gamma_fb=gamma,
                seed=base_seed
            )
            lcoe_grid[i, j] = best_f

    # 找全局最优
    idx = np.unravel_index(np.nanargmin(lcoe_grid), lcoe_grid.shape)
    best_prob = treat_values[idx[0]]
    best_gamma = gamma_values[idx[1]]
    best_lcoe = lcoe_grid[idx]

    print("\n================ GRID SEARCH BEST RESULT ================")
    print(f"Best LCOE       = {best_lcoe:.6f}")
    print(f"Best treat_prob = {best_prob:.3f}")
    print(f"Best gamma_fb   = {best_gamma:.3f}")
    print("=========================================================\n")

    # ========== 保存 CSV ==========
    out_csv = "ga_gamma_treat_lcoe_grid.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("treat_prob,gamma_fb,lcoe\n")
        for i, prob in enumerate(treat_values):
            for j, gamma in enumerate(gamma_values):
                f.write(f"{prob:.6f},{gamma:.6f},{lcoe_grid[i,j]:.12f}\n")
    print(f"[INFO] Grid saved → {out_csv}")

    # ========== 画 2D 散点图 ==========
    TT, GG = np.meshgrid(treat_values, gamma_values)
    X = GG.ravel()
    Y = TT.ravel()
    Z = lcoe_grid.ravel()

    fig, ax = plt.subplots(figsize=(7, 5))

    sc = ax.scatter(
        X, Y, c=Z,
        cmap="viridis",
        s=35,
        edgecolors="none"
    )

    ax.scatter(
        best_gamma,
        best_prob,
        marker="*",
        s=160,
        color="red",
        label="Best"
    )

    ax.set_xlabel("gamma_feedback")
    ax.set_ylabel("parent_treat_prob")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("ga_gamma_treat_2d.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("[INFO] Figure saved → ga_gamma_treat_2d.png")


if __name__ == "__main__":
    main()
