# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:54:48 2025

@author: Lenovo
"""

# ts/main_tabu_comparison.py
import os, time, random, collections
import numpy as np
import pandas as pd

from config_tabu import get_config
from microgrid_env import MicrogridEnv
from sac_agent import SACAgent
from utils import plot_statistical_results, plot_final_fitness_boxplot

import ts_core
from classic_tabu_simple import run_classic_tabu_simple
from classic_tabu_guided import run_classic_tabu_guided

STATUS_MAP = {"success": 0, "fail_supply": 1, "fail_storage_overdischarge": 2,
              "fail_storage_overcharge": 3, "simulation_error": 4}

def set_global_seed(seed):
    import numpy as np
    import random
    import torch

    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass


def load_typical_days_data(data_folder="data"):
    typical_days_data = []
    required = ['GHI', 'Tair', 'Vwind', 'Load']
    for i in range(5):
        fp = os.path.join(data_folder, f"typicalday{i}.csv")
        df = pd.read_csv(fp)
        miss = [c for c in required if c not in df.columns]
        if miss:
            print(f"!!! FATAL: '{fp}' missing {miss}"); return None
        typical_days_data.append(df)
    print(f"Loaded {len(typical_days_data)} typical days from '{data_folder}'.")
    return typical_days_data

def get_state_ts(state, iteration, total_iterations, gbest_hist):
    iter_ratio = iteration / total_iterations
    # 邻域多样性用最近若干步的“区域哈希”数量估算；这里简化：用 sigma 映射
    normalized_sigma = min(max(state['sigma_ratio'] / 0.5, 0.0), 1.0)
    convergence_rate = 0
    hist = list(gbest_hist)
    if len(hist) > 10 and hist[0] < 1e6:
        improvement = hist[0] - hist[-1]
        convergence_rate = improvement / max(hist[0], 1e-9)
    status = state.get("best_diag", {}).get("status", "success")
    status_code = STATUS_MAP.get(status, 0) / len(STATUS_MAP)
    return np.array([iter_ratio, normalized_sigma, convergence_rate, status_code])

def calculate_reward(old_best, new_best, diagnostics, first_feasible, config):
    p = config["sac_agent_params"]
    if first_feasible: return p["rewards"]["first_feasible_bonus"]
    r = 0
    if new_best < old_best:
        imp = (old_best - new_best) / max(old_best, 1e-9) if old_best < 1e6 else 0
        r += p["rewards"]["gbest_improvement"] * np.tanh(10 * imp)
    else:
        r += p["rewards"]["stagnation"]
    status = diagnostics.get("status", "success") if diagnostics else "success"
    if status != "success": r += p["penalties"].get(status, 0)
    return r

import csv

def run_sac_guided_ts(config, env, total_iterations, run_seed):
    print(f"\n--- Running SAC-Guided-TS (Run #{run_seed+1}) ---")
    np.random.seed(run_seed); random.seed(run_seed)

    agent = SACAgent(config)
    state = ts_core.initialize_tabu_state(config, env, seed=run_seed)
    fit_hist, gbest_hist = [], collections.deque(maxlen=20)
    found_feasible = (state["best_f"] < 1e6)

    # ====== 1) 为当前 run 准备日志文件 ======
    base_log_dir = os.path.join(".", "runs_ts", "sac_guided")
    run_log_dir = os.path.join(base_log_dir, f"run_{run_seed}")
    os.makedirs(run_log_dir, exist_ok=True)

    log_path = os.path.join(run_log_dir, "iter_log.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["iter", "reward", "diag_prob", "gamma_fb",
                         "lcoe", "state0", "state1", "state2", "state3"])

    s = get_state_ts(state, 0, total_iterations, gbest_hist)

    for i in range(total_iterations):
        act = agent.choose_action(s)      # [diag_prob, gamma_feedback]
        old_best = state["best_f"]

        state = ts_core.run_tabu_iteration_guided(state, act, env)

        is_first_feasible = (not found_feasible) and (state["best_f"] < 1e6)
        if is_first_feasible:
            found_feasible = True

        r = calculate_reward(old_best, state["best_f"], state["best_diag"], is_first_feasible, config)
        s2 = get_state_ts(state, i + 1, total_iterations, gbest_hist)

        agent.replay_buffer.add(s, act, r, s2, (i == total_iterations - 1))
        agent.learn()

        # ====== 2) 写一行日志 ======
        diag_prob = float(act[0])
        gamma_fb  = float(act[1])
        lcoe      = float(state["best_f"])
        log_writer.writerow([
            i,
            float(r),
            diag_prob,
            gamma_fb,
            lcoe,
            float(s2[0]),
            float(s2[1]),
            float(s2[2]),
            float(s2[3]),
        ])

        s = s2
        gbest_hist.append(state["best_f"])
        fit_hist.append(state["best_f"])

    # ====== 3) 关闭日志文件 ======
    log_file.close()
    print(f"[SAC-Guided TS] iter_log saved to: {log_path}")

    return {
        "best_solution": state["best_x"],
        "best_fitness": state["best_f"],
        "fitness_history": fit_hist
    }

def run_tabu_experiments():


    config = get_config()
    seed = config.get("random_seed", 42)
    set_global_seed(seed)
    total_iterations = config["simulation_params"]["total_iterations"]
    num_runs = config["simulation_params"]["num_independent_runs"]

    tdata = load_typical_days_data()
    if not tdata: return
    env = MicrogridEnv(config, tdata)

    results = collections.defaultdict(list)
    start = time.time()
    for i in range(num_runs):
        print(f"\n{'='*24} Statistical Run #{i+1}/{num_runs} {'='*24}")
        results["Guided TS with SAC"].append(run_sac_guided_ts(config, env, total_iterations, seed))
        results["Guided TS"].append(run_classic_tabu_guided(config, env, total_iterations, seed))
        results["TS"].append(run_classic_tabu_simple(config, env, total_iterations, seed))

    print(f"\n{'='*20} STATISTICS COMPLETE {'='*20}")
    print(f"Total time for {num_runs} runs: {time.time() - start:.2f}s")

    histories = {name: [res["fitness_history"] for res in res_list] for name, res_list in results.items()}
    finals    = {name: [h[-1] for h in h_list] for name, h_list in histories.items()}
    
    histories = {name: [res["fitness_history"] for res in res_list] for name, res_list in results.items()}
    finals    = {name: [h[-1] for h in h_list] for name, h_list in histories.items()}
    
    # 打印三种算法在所有独立运行中的最优解与装机配置
    print("\nBest solutions over all runs:")
    for name, res_list in results.items():
        if not res_list:
            continue
    
        # 在 num_runs 次独立运行中，找到该算法 LCOE 最小的那一次
        best_idx = int(np.argmin([res["best_fitness"] for res in res_list]))
        best_res = res_list[best_idx]
    
        # 连续解
        x_cont = np.array(best_res["best_solution"], dtype=float)
    
        # 取整后的装机配置（不小于 0）
        x_int = np.maximum(np.round(x_cont).astype(int), 0)
    
        print(f"\n[{name}] 最优结果来自 Run #{best_idx+1}:")
        try:
            print(f"  最优目标值（LCOE） = {best_res['best_fitness']:.6f}")
        except Exception:
            # 防止 best_fitness 不是 float 的情况
            print(f"  最优目标值（LCOE） = {best_res['best_fitness']}")
        print(f"  连续变量 x_best   = {x_cont}")
    
        # 按 [PV, WIND, EC, FC, TANK, BESS] 的顺序打印
        labels_cn = [
            "PV 组件数量",
            "WIND 机组数量",
            "EC 台数",
            "FC 台数",
            "H2 TANK 数量",
            "BESS 簇数量",
        ]
        print("  装机配置（取整后）：")
        for label, val in zip(labels_cn, x_int):
            print(f"    {label:<10} = {val}")
    
    plot_statistical_results(list(histories.values()), list(histories.keys()))
    plot_final_fitness_boxplot(finals)


    plot_statistical_results(list(histories.values()), list(histories.keys()))
    plot_final_fitness_boxplot(finals)

if __name__ == "__main__":
    run_tabu_experiments()
