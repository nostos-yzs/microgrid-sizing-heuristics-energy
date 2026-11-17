# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:44:55 2025

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 16:xx:xx 2025

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
pso_grid_search_gamma_treat.py
对 Classic Guided PSO 的 (treat_prob, gamma_feedback)
做网格搜索，绘制 LCOE 平面图

使用：
    python pso_grid_search_gamma_treat.py

依赖：
    - config_sac.get_config
    - microgrid_env.MicrogridEnv
    - classic_pso_guided.run_classic_pso_guided
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config_sac import get_config
from microgrid_env import MicrogridEnv
from classic_pso_guided import run_classic_pso_guided


# ========== 工具 ==========
def load_typical_days_data(data_folder="data"):
    """
    与 main_comparison.py 中保持一致的典型日加载逻辑。
    """
    import pandas as pd

    typical_days_data = []
    required_columns = ["GHI", "Tair", "Vwind", "Load"]

    for i in range(5):
        file_path = os.path.join(data_folder, f"typicalday{i}.csv")
        df = pd.read_csv(file_path)

        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"!!! FATAL ERROR: In file '{file_path}', "
                f"missing columns: {missing_cols}"
            )
        typical_days_data.append(df)

    print(f"[INFO] Loaded {len(typical_days_data)} typical days from '{data_folder}'.")
    return typical_days_data


# ========== 单个组合下运行一次 Guided PSO ==========
def run_single_pso(env, config, treat_prob, gamma_fb, seed=0):
    """
    固定 treat_prob 和 gamma_feedback 运行一次 Classic Guided PSO。
    """
    import numpy as np
    np.random.seed(seed)

    # 修改 config，使 PSO 采用指定参数
    cfg = config
    cfg["classic_guided_params"]["treat_prob"] = treat_prob
    cfg["classic_guided_params"]["gamma_feedback"] = gamma_fb

    total_iters = cfg["simulation_params"]["total_iterations"]

    res = run_classic_pso_guided(
        cfg, env,
        total_iterations=total_iters,
        run_seed=seed
    )

    return float(res["best_fitness"]), res["fitness_history"]


# ========== 网格搜索主函数 ==========
def main():
    # 1) 读取配置与环境
    config = get_config()
    total_iters = config["simulation_params"]["total_iterations"]
    print(f"[INFO] total_iterations = {total_iters}")

    typical_days = load_typical_days_data("data")
    env = MicrogridEnv(config, typical_days)

    # 2) 定义网格（范围可以按需改）
    #   这里沿用 GA 版本的范围设置：
    #   - treat_prob: 0.1 ~ 1.0
    #   - gamma_feedback: 0.5 ~ 10.5
    TREAT_MIN, TREAT_MAX, TREAT_STEP = 0.01, 0.4, 0.01
    GAMMA_MIN, GAMMA_MAX, GAMMA_STEP = 6, 17, 0.5

    treat_values = np.arange(TREAT_MIN, TREAT_MAX + 1e-9, TREAT_STEP)
    gamma_values = np.arange(GAMMA_MIN, GAMMA_MAX + 1e-9, GAMMA_STEP)

    P = len(treat_values)
    G = len(gamma_values)
    print(f"[INFO] Grid size: {P} x {G} = {P * G} runs")

    # 存 best LCOE
    lcoe_grid = np.full((P, G), np.nan)

    base_seed = 42

    # 3) 双重循环执行
    for i, prob in enumerate(treat_values):
        print(f"[ROW {i + 1}/{P}] treat_prob = {prob:.2f}")
        for j, gamma in enumerate(gamma_values):
            best_f, _ = run_single_pso(
                env,
                config,
                treat_prob=prob,
                gamma_fb=gamma,
                seed=base_seed
            )
            lcoe_grid[i, j] = best_f
            print(f"   gamma_fb = {gamma:.2f} → LCOE = {best_f:.6f}")

    # 4) 找全局最优
    idx = np.unravel_index(np.nanargmin(lcoe_grid), lcoe_grid.shape)
    best_prob = treat_values[idx[0]]
    best_gamma = gamma_values[idx[1]]
    best_lcoe = lcoe_grid[idx]

    print("\n================ GRID SEARCH BEST RESULT ================")
    print(f"Best LCOE        = {best_lcoe:.6f}")
    print(f"Best treat_prob  = {best_prob:.3f}")
    print(f"Best gamma_fb    = {best_gamma:.3f}")
    print("=========================================================\n")

    # 5) 保存 CSV
    out_csv = "pso_gamma_treat_lcoe_grid.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("treat_prob,gamma_fb,lcoe\n")
        for i, prob in enumerate(treat_values):
            for j, gamma in enumerate(gamma_values):
                f.write(f"{prob:.6f},{gamma:.6f},{lcoe_grid[i, j]:.12f}\n")
    print(f"[INFO] Grid saved → {out_csv}")

    # 6) 画 2D 散点图：X=gamma_feedback, Y=treat_prob, 颜色=LCOE
    TT, GG = np.meshgrid(treat_values, gamma_values)
    X = GG.ravel()
    Y = TT.ravel()
    Z = lcoe_grid.ravel()

    fig, ax = plt.subplots(figsize=(7, 5))

    sc = ax.scatter(
        X, Y,
        c=Z,
        cmap="viridis",
        s=35,
        edgecolors="none"
    )

    # 标注最优点
    ax.scatter(
        best_gamma,
        best_prob,
        marker="*",
        s=160,
        color="red",
        label="Best"
    )

    ax.set_xlabel("gamma_feedback")
    ax.set_ylabel("treat_prob")
    ax.legend(loc="upper right")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("LCOE")

    plt.tight_layout()
    plt.savefig("pso_gamma_treat_2d.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("[INFO] Figure saved → pso_gamma_treat_2d.png")


if __name__ == "__main__":
    main()
