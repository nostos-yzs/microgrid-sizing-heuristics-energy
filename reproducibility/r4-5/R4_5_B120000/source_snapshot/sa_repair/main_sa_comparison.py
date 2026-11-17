# -*- coding: utf-8 -*-
"""
main_sa_comparison.py  —— 三种 SA 模式对比（两层退火）
  1) SA 传统（可近似“无引导”）
  2) SA + 诊断引导（固定参数）
  3) SA + 诊断引导 + SAC 学两维 [gamma_fb, diag_prob]
依赖：
- config_sa.get_config
- microgrid_env.MicrogridEnv
- sac_agent.SACAgent   （action_dim=2）
- sa_core.initialize_sa_state, sa_core.run_sa_iteration
- utils.plot_statistical_results / plot_final_fitness_boxplot
"""

import os
import time
import numpy as np
import pandas as pd
import torch

from config_sa import get_config
from microgrid_env import MicrogridEnv
from sac_agent import SACAgent
from sa_core import initialize_sa_state as initialize_sa, run_sa_iteration
from utils import plot_statistical_results, plot_final_fitness_boxplot

STATUS_MAP = {
    "success": 0,
    "fail_supply": 1,
    "fail_storage_overdischarge": 2,
    "fail_storage_overcharge": 3,
    "simulation_error": 4
}

# ---------- 典型日数据 ----------
def load_typical_days_data(data_folder="data"):
    typical_days_data = []
    required_columns = ['GHI', 'Tair', 'Vwind', 'Load']
    for i in range(5):
        fp = os.path.join(data_folder, f"typicalday{i}.csv")
        if not os.path.exists(fp):
            print(f"[ERROR] not found: {fp}")
            return None
        df = pd.read_csv(fp)
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            print(f"[ERROR] missing columns in {fp}: {missing}")
            return None
        typical_days_data.append(df)
    print(f"[INFO] loaded {len(typical_days_data)} typical days from '{data_folder}'")
    return typical_days_data

# ---------- RL 状态 / 奖励 ----------
def get_state_sa(sa_state: dict, total_iterations: int) -> np.ndarray:
    iter_ratio = float(sa_state.get("iter", 0)) / max(1, total_iterations)

    diversity = 0.0
    normalized_diversity = 0.0

    f_best = float(sa_state.get("f_best", 1e9))
    f_cur  = float(sa_state.get("f", 1e9))
    if f_cur <= 0 or f_best >= 1e8:
        conv = 0.0
    else:
        conv = 1.0 - min(1.0, f_best / (f_cur + 1e-9))

    status_str = (sa_state.get("diag_best") or {}).get("status", "success")
    status_code = STATUS_MAP.get(status_str, 0) / max(1, len(STATUS_MAP))

    return np.array([iter_ratio, normalized_diversity, conv, status_code], dtype=float)

# def calculate_reward_sa(old_gbest: float, new_gbest: float, diagnostics: dict,
#                         first_feasible: bool, sac_cfg: dict) -> float:
#     p = sac_cfg
#     rewards = p.get("rewards", {"gbest_improvement": 10, "stagnation": -1, "first_feasible_bonus": 100})
#     penalties = p.get("penalties", {})

#     if first_feasible:
#         return float(rewards.get("first_feasible_bonus", 100))

#     reward = 0.0
#     if new_gbest < old_gbest:
#         improvement = (old_gbest - new_gbest) / old_gbest if 0 < old_gbest < 1e6 else 0.0
#         reward += rewards.get("gbest_improvement", 10) * np.tanh(improvement * 10.0)
#     else:
#         reward += rewards.get("stagnation", -1)

#     status = (diagnostics or {}).get("status", "success")
#     if status != "success":
#         reward += penalties.get(status, 0.0)
#     return float(reward)

def calculate_reward_sa(old_gbest, new_gbest, diagnostics, first_feasible, config):
    params = config
    reward = 0.0

    # (A) gbest 改善 / 停滞
    if new_gbest < old_gbest:
        improvement = (old_gbest - new_gbest) / old_gbest if 0 < old_gbest < 1e6 else 0.0
        reward += params["rewards"]["gbest_improvement"] * np.tanh(improvement * 10.0)
    else:
        reward += params["rewards"]["stagnation"]

    # (B) 一次性“首次可行”奖励：仅当本轮第一次进入可行域时叠加
    if first_feasible:
        reward = params["rewards"]["first_feasible_bonus"]

    # (C) 失败类型罚分（用“本轮”的 diagnostics，见步骤 3 的传参）
    status = diagnostics.get("status", "success") if diagnostics else "success"
    if status != "success":
        reward += params["penalties"].get(status, 0)

    return float(reward)

# ---------- 工具：读取两层参数 ----------
def _read_layers(cfg):
    sim = cfg.get("simulation_params", {})
    n_layers = int(sim.get("n_layers", 30))
    inner_loops = int(sim.get("inner_loops", 1))
    return n_layers, inner_loops

# ---------- 三种模式 ----------
def run_sa_traditional(config, env, total_iterations, run_seed):
    print(f"\n--- SA (traditional) Run #{run_seed+1} ---")
    np.random.seed(run_seed); torch.manual_seed(run_seed)

    min_b = config["min_bounds"].astype(float)
    max_b = config["max_bounds"].astype(float)
    sched = config["sa_params"]["schedule"]
    seed = int(config["experiments"]["classic_simple"]["random_seed"])

    sa = initialize_sa(
        env=env, min_b=min_b, max_b=max_b, schedule=sched,
        seed=seed, rw_cfg={"rewards": {}, "penalties": {}},
    )
    # 两层参数
    n_layers, inner_loops = _read_layers(config)
    T0, alpha, Tmin = float(sched["T0"]), float(sched["alpha"]), float(sched["Tmin"])

    fixed_action = np.array([0.0, 0.0], dtype=float)
    fit_hist = []
    for layer in range(n_layers):
        sa["T"] = max(T0 * (alpha ** layer), Tmin)  # 外层降温
        for _ in range(inner_loops):
            sa = run_sa_iteration(sa, fixed_action, env)
            fit_hist.append(float(sa["f_best"]))
    return {"best_solution": sa["x_best"], "best_fitness": float(sa["f_best"]), "fitness_history": fit_hist}

def run_sa_guided(config, env, total_iterations, run_seed):
    print(f"\n--- SA (guided-fixed) Run #{run_seed+1} ---")
    np.random.seed(run_seed); torch.manual_seed(run_seed)

    min_b = config["min_bounds"].astype(float)
    max_b = config["max_bounds"].astype(float)
    sched = config["sa_params"]["schedule"]

    exp = config["experiments"]["classic_guided"]
    gamma_fb = float(exp["gamma_fb"])
    diag_prob = float(exp["diag_prob"])
    seed = int(exp["random_seed"])

    sa = initialize_sa(
        env=env, min_b=min_b, max_b=max_b, schedule=sched,
        seed=seed, rw_cfg={"rewards": {}, "penalties": {}},
    )
    n_layers, inner_loops = _read_layers(config)
    T0, alpha, Tmin = float(sched["T0"]), float(sched["alpha"]), float(sched["Tmin"])

    fixed_action = np.array([gamma_fb, diag_prob], dtype=float)
    fit_hist = []
    for layer in range(n_layers):
        sa["T"] = max(T0 * (alpha ** layer), Tmin)
        for _ in range(inner_loops):
            sa = run_sa_iteration(sa, fixed_action, env)
            fit_hist.append(float(sa["f_best"]))
    return {"best_solution": sa["x_best"], "best_fitness": float(sa["f_best"]), "fitness_history": fit_hist}

def _map_to_bounds(action01: np.ndarray, bounds: dict) -> np.ndarray:
    out = []
    keys = ["gamma_fb", "diag_prob"]
    for i, k in enumerate(keys):
        lo, hi = bounds[k]
        a = float(action01[i])
        if -1.2 <= a <= 1.2:
            a01 = np.clip(a, -1.0, 1.0)
            val = lo + (a01 + 1.0) * 0.5 * (hi - lo)
        else:
            val = np.clip(a, lo, hi)
        out.append(val)
    return np.array(out, dtype=float)

def run_sa_rl_guided(config, env, total_iterations, run_seed):
    print(f"\n--- SA (guided + SAC) Run #{run_seed+1} ---")
    np.random.seed(run_seed); torch.manual_seed(run_seed)

    min_b = config["min_bounds"].astype(float)
    max_b = config["max_bounds"].astype(float)
    sched = config["sa_params"]["schedule"]

    sac_cfg = config["experiments"]["sac_guided"]["sac_agent_params"]
    bounds = sac_cfg["action_bounds"]
    seed = int(config["experiments"]["sac_guided"]["random_seed"])

    cfg_for_agent = dict(config)
    cfg_for_agent["sac_agent_params"] = sac_cfg
    agent = SACAgent(cfg_for_agent)  # action_dim=2

    sa = initialize_sa(
        env=env, min_b=min_b, max_b=max_b, schedule=sched,
        seed=seed, rw_cfg=sac_cfg
    )

    n_layers, inner_loops = _read_layers(config)
    total_iterations = n_layers * inner_loops  # 仍用于 RL 状态比例、终止条件
    T0, alpha, Tmin = float(sched["T0"]), float(sched["alpha"]), float(sched["Tmin"])

    fit_hist = []
    t = 0
    state = get_state_sa(sa, total_iterations)

    # 记录“历史上是否出现过可行解”
    ever_feasible = ((sa.get("diag_best") or {}).get("status", "") == "success")

    log_rows = []
    for layer in range(n_layers):
        sa["T"] = max(T0 * (alpha ** layer), Tmin)
        for _ in range(inner_loops):
            raw_action = agent.choose_action(state)
            if len(raw_action) != 2:
                raise RuntimeError(f"SACAgent.choose_action() must return 2 dims, got {len(raw_action)}")
            action = _map_to_bounds(np.asarray(raw_action, dtype=float), bounds)

            gamma_fb = float(action[0])
            diag_prob = float(action[1])

            old_best = float(sa["f_best"])
            sa = run_sa_iteration(sa, action, env)

            # 当前这一代的诊断状态
            status_now = (sa.get("diag_best") or {}).get("status", "")

            # 只在“从未可行 → 本代第一次可行”时触发一次
            if (not ever_feasible) and (status_now == "success"):
                just_became_feasible = True
                ever_feasible = True
            else:
                just_became_feasible = False

            reward = calculate_reward_sa(
                old_best,
                float(sa["f_best"]),
                sa.get("diag_best", {}),
                just_became_feasible,   # 注意这里传的是 just_became_feasible
                sac_cfg
            )

            next_state = get_state_sa(sa, total_iterations)
            done = (t == total_iterations - 1)

            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            fit_hist.append(float(sa["f_best"]))

            last_reward = float(sa.get("last_reward", reward))
            status = (sa.get("diag") or {}).get("status", "unknown")

            log_rows.append({
                "iter": int(sa.get("iter", t)),
                "best_f": float(sa["f_best"]),
                "f": float(sa["f"]),
                "gamma_fb": gamma_fb,
                "diag_prob": diag_prob,
                "T": float(sa["T"]),
                "reward": reward,
                "status": status,
            })
            t += 1

    # ========= 新增：把日志写成 CSV =========
    log_root = os.path.join("runs_sa", "sac_guided")
    os.makedirs(log_root, exist_ok=True)
    csv_path = os.path.join(log_root, f"iter_log.csv")
    pd.DataFrame(log_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[LOG] SA iter log saved to: {csv_path}")
            

    return {"best_solution": sa["x_best"], "best_fitness": float(sa["f_best"]), "fitness_history": fit_hist}

# ---------- 统计主程序 ----------
def run_sa_experiments():
    config = get_config()

    # 典型日
    typical_days = load_typical_days_data("data")
    if typical_days is None:
        print("[FATAL] typical day CSVs missing. Abort.")
        return

    # Env
    env = MicrogridEnv(config, typical_days)

    # 两层 → 总步数 = n_layers * inner_loops
    n_layers, inner_loops = _read_layers(config)
    total_iterations = n_layers * inner_loops

    num_runs = 1

    results = {"Guided SA with SAC": [],"Guided SA": [],"SA": []  }
    # results = {"SA (traditional)": []}

    t0 = time.time()
    random_seed=42
    for i in range(num_runs):
        print(f"\n{'='*24} Run {i+1}/{num_runs} {'='*24}")
        results["SA"].append(run_sa_traditional(config, env, total_iterations, random_seed))
        results["Guided SA"].append(run_sa_guided(config, env, total_iterations, random_seed))
        results["Guided SA with SAC"].append(run_sa_rl_guided(config, env, total_iterations, random_seed))

    print(f"\n[INFO] Done. total time: {time.time()-t0:.2f}s")

    histories = {name: [r["fitness_history"] for r in lst] for name, lst in results.items()}
    finals    = {name: [h[-1] for h in hs] for name, hs in histories.items() if len(hs)>0 and len(hs[0])>0}

    plot_statistical_results(
        histories_list=[histories[k] for k in results.keys()],
        labels=list(results.keys())
    )
    plot_final_fitness_boxplot(finals)
    # ===== 打印每种 SA 在所有 runs 中的全局最优解及装机配置 =====
    for method_name, res_list in results.items():
        if not res_list:
            continue
    
        # 选出这一种算法中 best_fitness 最小的那个 run
        best_run = min(res_list, key=lambda r: r["best_fitness"])
        x_best = np.array(best_run["best_solution"], dtype=float)
        f_best = float(best_run["best_fitness"])
    
        # 解释为装机数量（向上取整，<=0 视为 0）
        n_PV, n_WIND, n_EC, n_FC, n_TANK, n_BESS = (
            int(np.ceil(x)) if x > 0 else 0 for x in x_best
        )
    
        print(f"\n[{method_name}] 全部 {num_runs} 次实验中的最优结果：")
        print(f"  最优目标值（LCOE） = {f_best:.6f}")
        print(f"  连续变量 x_best   = {x_best}")
        print("  装机配置（取整后）：")
        print(f"    PV   组件数量 = {n_PV}")
        print(f"    WIND 机组数量 = {n_WIND}")
        print(f"    EC   台数     = {n_EC}")
        print(f"    FC   台数     = {n_FC}")
        print(f"    H2 TANK 数量  = {n_TANK}")
        print(f"    BESS 簇数量   = {n_BESS}")


if __name__ == "__main__":
    run_sa_experiments()
