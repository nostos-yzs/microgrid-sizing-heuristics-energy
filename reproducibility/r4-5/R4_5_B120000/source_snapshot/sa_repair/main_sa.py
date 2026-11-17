# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 11:04:05 2025

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
main_sa.py
三种 SA 模式（SAC-Guided / Classic-Guided / Classic-Simple）的统一入口。
与 main_ga_comparison.py 的结构保持一致，直接可用于统计对比与作图。
"""
import os
import time
import random
import numpy as np
import pandas as pd
import torch

from config_sa import get_config
from microgrid_env import MicrogridEnv
from sac_agent import SACAgent
from sa_core import initialize_sa, run_sa_iteration

from utils import plot_statistical_results, plot_final_fitness_boxplot

# —— 状态编码映射，与 GA 对齐 ——
STATUS_MAP = {"success": 0, "fail_supply": 1, "fail_storage_overdischarge": 2,
              "fail_storage_overcharge": 3, "simulation_error": 4}

# —— 典型日数据加载（复用 GA 写法） ——
def load_typical_days_data(data_folder="data"):
    typical_days_data = []
    required_columns = ['GHI', 'Tair', 'Vwind', 'Load']
    try:
        for i in range(5):
            file_path = os.path.join(data_folder, f"typicalday{i}.csv")
            df = pd.read_csv(file_path)
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"!!! FATAL ERROR: In file '{file_path}', missing columns: {missing_cols}")
                return None
            typical_days_data.append(df)
        print(f"Successfully loaded {len(typical_days_data)} typical days from '{data_folder}'.")
        return typical_days_data
    except FileNotFoundError:
        print(f"Error: Data folder '{data_folder}' or its files not found.")
        return None

# —— 与 GA 契合的状态构造（无种群，用 recent_pool 近似 diversity） ——
def get_state_sa(sa_state: dict, total_iterations: int) -> np.ndarray:
    """
    state = [iteration_ratio, diversity, convergence_rate, status_code]
      - iteration_ratio: 迭代进度
      - diversity: 利用 recent_pool 的分量方差均值来近似多样性
      - convergence_rate: 最近若干步的 gbest 改善率（这里用 f_best 序列需外部传；为便捷用 recent_pool 差分近似）
      - status_code: 取 diag_best['status'] 的编码 / |STATUS_MAP|
    """
    iter_ratio = sa_state["iter"] / max(1, total_iterations)

    # diversity：最近解窗的分量标准差的均值（归一到 ~[0,1]，以 100 为经验尺度）
    rp = np.array(sa_state.get("recent_pool", []))
    if rp.ndim == 2 and len(rp) >= 2:
        diversity = float(np.std(rp, axis=0).mean())
    else:
        diversity = 0.0
    normalized_diversity = min(diversity / 100.0, 1.0)

    # convergence_rate：用最近两点 best 的差分近似（简化版）
    # 为避免额外依赖，这里使用 current 与 best 的差值做一个粗略指标
    f_best = float(sa_state["f_best"])
    f_cur = float(sa_state["f_cur"])
    improvement = max(0.0, (f_cur - f_best))  # 若当前劣于 best，则改进为 0
    convergence_rate = 0.0 if f_best <= 0 or f_best >= 1e6 else (1.0 - min(1.0, f_best / (f_cur + 1e-9)))

    # status 编码
    status_str = (sa_state.get("diag_best") or {}).get("status", "success")
    status_code = STATUS_MAP.get(status_str, 0) / max(1, len(STATUS_MAP))

    return np.array([iter_ratio, normalized_diversity, convergence_rate, status_code], dtype=float)

# —— 与 GA 一致的奖励口径（可直接复用 GA 的实现；这里给出自洽实现） ——
def calculate_reward_sa(old_gbest: float, new_gbest: float, diagnostics: dict,
                        first_feasible: bool, config: dict) -> float:
    params = config["sac_agent_params"]
    if first_feasible:
        return params["rewards"]["first_feasible_bonus"]

    reward = 0.0
    if new_gbest < old_gbest:
        improvement = (old_gbest - new_gbest) / old_gbest if 0 < old_gbest < 1e6 else 0.0
        reward += params["rewards"]["gbest_improvement"] * np.tanh(improvement * 10.0)
    else:
        reward += params["rewards"]["stagnation"]

    status = (diagnostics or {}).get("status", "success")
    if status != "success":
        reward += params["penalties"].get(status, 0.0)
    return float(reward)

# ======================= 三种运行模式 =======================

def _map_action_bounds(raw_action: np.ndarray, bounds_dict: dict) -> np.ndarray:
    """
    将 SAC 的原始动作（若在 [-1,1]）映射到物理区间。
    这里默认 SACAgent.choose_action() 已经做了映射；如未做，则可调用本函数。
    """
    # 占位：直接返回；若需线性映射，可在此按 bounds 处理
    return raw_action

def run_sac_guided_sa(config, env, total_iterations, run_seed):
    print(f"\n--- Running SAC-Guided SA (Run #{run_seed+1}) ---")
    np.random.seed(run_seed); random.seed(run_seed); torch.manual_seed(run_seed)

    agent = SACAgent(config)  # 要求其 action_dim=8，与 config_sa 中一致
    sa = initialize_sa(config, env)

    fit_hist = []
    state = get_state_sa(sa, total_iterations)
    first_feasible = (sa["f_best"] < 1e6)

    # 提前取 action 边界（若你在 SACAgent 内部已处理，可忽略）
    action_bounds = config["sac_agent_params"]["action_bounds"]

    for t in range(total_iterations):
        raw_action = agent.choose_action(state)  # 8 维
        action = _map_action_bounds(raw_action, action_bounds)

        old_best = sa["f_best"]
        sa = run_sa_iteration(sa, action, env)

        # 首次可行奖励
        if (not first_feasible) and sa["f_best"] < 1e6:
            first_feasible = True

        reward = calculate_reward_sa(old_best, sa["f_best"], sa["diag_best"], first_feasible, config)
        next_state = get_state_sa(sa, total_iterations)

        done = (t == total_iterations - 1)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.learn()

        state = next_state
        fit_hist.append(sa["f_best"])

    return {"best_solution": sa["x_best"], "best_fitness": sa["f_best"], "fitness_history": fit_hist}

def run_classic_guided_sa(config, env, total_iterations, run_seed):
    print(f"\n--- Running Classic Guided SA (Run #{run_seed+1}) ---")
    np.random.seed(run_seed); random.seed(run_seed)

    sa_cfg = config["sa_params"]["classic_guided"]
    # 将固定参数拼成与 SAC 相同的 8 维动作（每步相同）
    fixed_action = np.array([
        sa_cfg["T0"], sa_cfg.get("alpha", config["sa_params"]["alpha"]),
        sa_cfg["Tmin"], sa_cfg["p_guided"],
        sa_cfg["s_success"], sa_cfg["s_overdis"], sa_cfg["s_overchg"], sa_cfg["s_supply"]
    ], dtype=float)

    sa = initialize_sa(config, env)
    fit_hist = []

    for t in range(total_iterations):
        sa = run_sa_iteration(sa, fixed_action, env)
        fit_hist.append(sa["f_best"])

    return {"best_solution": sa["x_best"], "best_fitness": sa["f_best"], "fitness_history": fit_hist}

def run_classic_sa_simple(config, env, total_iterations, run_seed):
    print(f"\n--- Running Classic Simple SA (Run #{run_seed+1}) ---")
    np.random.seed(run_seed); random.seed(run_seed)

    sa_cfg = config["sa_params"]["classic_simple"]
    # 简化：无诊断引导，p_guided=0；随机步使用 random_step
    fixed_action = np.array([
        sa_cfg["T0"], sa_cfg["alpha"], sa_cfg["Tmin"], 0.0,
        sa_cfg["random_step"], sa_cfg["random_step"], sa_cfg["random_step"], sa_cfg["random_step"]
    ], dtype=float)

    sa = initialize_sa(config, env)
    fit_hist = []

    for t in range(total_iterations):
        sa = run_sa_iteration(sa, fixed_action, env)
        fit_hist.append(sa["f_best"])

    return {"best_solution": sa["x_best"], "best_fitness": sa["f_best"], "fitness_history": fit_hist}

# ======================= 统计主程序 =======================

def run_sa_experiments():
    config = get_config()
    typical_days = load_typical_days_data("data")
    if typical_days is None:
        print("!!! ERROR: failed to load typical days, abort.")
        return

    env = MicrogridEnv(config, typical_days)
    total_iterations = config["simulation_params"]["total_iterations"]
    num_runs = config["simulation_params"]["num_independent_runs"]

    results = {"SAC-Guided SA": [], "Classic Guided SA": [], "Classic Simple SA": []}

    start_time = time.time()
    for i in range(num_runs):
        print(f"\n{'='*25} Starting Statistical Run #{i+1}/{num_runs} {'='*25}")
        results["SAC-Guided SA"].append(run_sac_guided_sa(config, env, total_iterations, i))
        results["Classic Guided SA"].append(run_classic_guided_sa(config, env, total_iterations, i))
        results["Classic Simple SA"].append(run_classic_sa_simple(config, env, total_iterations, i))

    print(f"\n\n{'='*20} STATISTICAL ANALYSIS COMPLETE {'='*20}")
    print(f"Total time for {num_runs} runs: {time.time() - start_time:.2f} seconds")

    # 整理数据绘图（与 GA 一致）
    histories = {name: [res["fitness_history"] for res in res_list] for name, res_list in results.items()}
    final_fits = {name: [h[-1] for h in h_list] for name, h_list in histories.items()}

    plot_statistical_results(
        histories_list=list(histories.values()),
        labels=list(histories.keys())
    )
    plot_final_fitness_boxplot(final_fits)

if __name__ == "__main__":
    run_sa_experiments()
