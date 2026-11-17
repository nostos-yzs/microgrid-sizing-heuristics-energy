# main_comparison.py
import pandas as pd
import os
import numpy as np
import collections
import torch
import time
import csv


# 引入所有需要的模块
# 注意: 我们现在导入的是新的绘图函数
from config_sac import get_config
from microgrid_env import MicrogridEnv
from sac_agent import SACAgent
import pso_core_sac
from classic_pso_guided import run_classic_pso_guided
from classic_pso_simple import run_classic_pso_simple
from utils import plot_statistical_results, plot_final_fitness_boxplot


class EvalCounterEnv:
    """Wrap env to count evaluate_solution calls + time."""

    def __init__(self, env):
        self.env = env
        self.reset_counters()

    def reset_counters(self):
        self.ems_eval_count = 0
        self.ems_eval_time_s = 0.0

    def evaluate_solution(self, x):
        t0 = time.perf_counter()
        out = self.env.evaluate_solution(x)
        self.ems_eval_time_s += (time.perf_counter() - t0)
        self.ems_eval_count += 1
        return out

    def __getattr__(self, name):
        return getattr(self.env, name)


def find_time_to_target_from_history(fitness_history, eps=0.01):
    """
    time-to-target: first iteration where fitness <= (1+eps)*final_best.
    Returns: (tgt_iter, final_best, threshold)
    """
    if not fitness_history:
        return None, None, None
    final_best = float(fitness_history[-1])
    threshold = final_best * (1.0 + float(eps))
    tgt_iter = None
    for k, f in enumerate(fitness_history):
        if float(f) <= threshold:
            tgt_iter = k
            break
    if tgt_iter is None:
        tgt_iter = len(fitness_history) - 1
    return int(tgt_iter), float(final_best), float(threshold)


def attach_full_and_target_metrics(out, wall_full_s, ems_eval_count_full, ems_eval_time_full_s, eps=0.01):
    """Attach full-run metrics + time-to-target metrics (proportional approximation)."""
    out["wall_time_s_full"] = float(wall_full_s)
    out["ems_eval_count_full"] = int(ems_eval_count_full)
    out["ems_eval_time_s_full"] = float(ems_eval_time_full_s)

    hist = out.get("fitness_history", []) or []
    tgt_iter, best_final, threshold = find_time_to_target_from_history(hist, eps=eps)

    out["tgt_iter"] = tgt_iter
    out["best_final"] = best_final
    out["tgt_threshold"] = threshold

    if tgt_iter is None or best_final is None or len(hist) <= 1:
        out["wall_time_s_tgt"] = float(wall_full_s)
        out["ems_eval_count_tgt"] = int(ems_eval_count_full)
        out["ems_eval_time_s_tgt"] = float(ems_eval_time_full_s)
        return out

    frac = float(tgt_iter) / float(max(1, len(hist) - 1))
    out["wall_time_s_tgt"] = float(wall_full_s * frac)
    out["ems_eval_count_tgt"] = int(round(ems_eval_count_full * frac))
    out["ems_eval_time_s_tgt"] = float(ems_eval_time_full_s * frac)
    return out

# --- 数据加载函数 ---
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
        print(f"Successfully loaded {len(typical_days_data)} typical days from '{data_folder}' folder.")
        return typical_days_data
    except FileNotFoundError:
        print(f"Error: Data folder '{data_folder}' or its files not found.")
        return None

import pandas as pd
import os
import numpy as np
import collections
import torch
import time

# 这里插入一个辅助函数
def print_best_config(all_results, algo_name):
    """
    给定某个算法的多次运行结果，打印：
    - 全部 run 中 LCOE 最小的那一次
    - 对应的配置（各设备台数）
    """
    if not all_results:
        print(f"[{algo_name}] 没有结果可以打印。")
        return

    # 选出 best_fitness 最小的那一个 run
    best_idx = int(np.argmin([res["best_fitness"] for res in all_results]))
    best = all_results[best_idx]
    sol = best["best_solution"]
    if sol is None:
        print(f"[{algo_name}] 没有找到可行解（best_solution 为 None）。")
        return

    # 按你目前的约定：solution = [n_PV, n_WIND, n_EC, n_FC, n_TANK, n_BESS]
    n_PV, n_WIND, n_EC, n_FC, n_TANK, n_BESS = np.ceil(sol).astype(int)

    print(f"\n>>> {algo_name} 在 {len(all_results)} 次运行中的最优结果（run #{best_idx+1}）")
    print(f"    最优 LCOE: {best['best_fitness']:.6f}")
    print("    对应配置：")
    print(f"        n_PV   = {n_PV}")
    print(f"        n_WIND = {n_WIND}")
    print(f"        n_EC   = {n_EC}")
    print(f"        n_FC   = {n_FC}")
    print(f"        n_TANK = {n_TANK}")
    print(f"        n_BESS = {n_BESS}")


# --- SAC专用的辅助函数 ---
STATUS_MAP = {
    "success": 0, "fail_supply": 1, "fail_storage_overdischarge": 2,
    "fail_storage_overcharge": 3, "simulation_error": 4
}
def get_state(swarm_state, iteration, total_iterations, gbest_history):
    iter_ratio = iteration / total_iterations
    diversity = np.std(swarm_state["particles"], axis=0).mean()
    normalized_diversity = min(diversity / 100.0, 1.0)
    convergence_rate = 0
    if len(gbest_history) > 1 and gbest_history[0] < 1e6:
        improvement = gbest_history[0] - gbest_history[-1]
        convergence_rate = improvement / gbest_history[0] if gbest_history[0] > 0 else 0
    diagnostics = swarm_state["gbest_diagnostics"]
    status_str = diagnostics.get("status", "success") if diagnostics else "success"
    status_code = STATUS_MAP.get(status_str, 0) / len(STATUS_MAP)
    return np.array([iter_ratio, normalized_diversity, convergence_rate, status_code])

# def calculate_reward(old_gbest_fitness, new_gbest_fitness, diagnostics, is_first_feasible, config):
#     sac_params = config["sac_agent_params"]
#     reward = 0
#     if is_first_feasible: return sac_params["rewards"]["first_feasible_bonus"] 
#     old_fitness_comp = old_gbest_fitness if old_gbest_fitness < 1e6 else 2e6
#     new_fitness_comp = new_gbest_fitness if new_gbest_fitness < 1e6 else 2e6
#     if new_fitness_comp < old_fitness_comp:
#         improvement_reward = (old_fitness_comp - new_fitness_comp) / old_fitness_comp if old_fitness_comp > 0 else 1
#         reward += sac_params["rewards"]["gbest_improvement"] * np.tanh(improvement_reward * 10)
#     else:
#         reward += sac_params["rewards"]["stagnation"]
#     status = diagnostics.get("status", "success") if diagnostics else "success"
#     if status != "success": reward += sac_params["penalties"].get(status, 0) / 10.0
#     return reward

def calculate_reward(old_gbest, new_gbest, diagnostics, first_feasible, config):
    params = config["sac_agent_params"]
    reward = 0.0

    # (A) gbest 改善 / 停滞
    if new_gbest < old_gbest:
        improvement = (old_gbest - new_gbest) / old_gbest if 0 < old_gbest < 1e6 else 0.0
        reward += params["rewards"]["gbest_improvement"] * np.tanh(improvement * 10.0)
    else:
        reward += params["rewards"]["stagnation"]

    # (B) 一次性“首次可行”奖励：仅当本轮第一次进入可行域时叠加
    if first_feasible:
        reward += params["rewards"]["first_feasible_bonus"]

    # (C) 失败类型罚分（用“本轮”的 diagnostics，见步骤 3 的传参）
    status = diagnostics.get("status", "success") if diagnostics else "success"
    if status != "success":
        reward += params["penalties"].get(status, 0)

    return float(reward)

# --- 实验运行函数 ---
# def run_sac_guided_pso(config, env, total_iterations, run_seed):
#     print(f"\n--- Running SAC-Guided-PSO (Run #{run_seed+1}) ---")
#     np.random.seed(run_seed); torch.manual_seed(run_seed)
#     agent = SACAgent(config)
#     swarm_state = pso_core_sac.initialize_swarm(config)
#     gbest_fitness, fitness_history, gbest_history = swarm_state["gbest_fitness"], [], collections.deque(maxlen=10)
#     found_first_feasible = False
#     state = get_state(swarm_state, 0, total_iterations, gbest_history)
#     for i in range(total_iterations):
#         action_vector = agent.choose_action(state)
#         old_gbest_fitness = gbest_fitness
#         swarm_state, iter_results = pso_core_sac.run_pso_iteration_guided(swarm_state, action_vector, env)
#         new_gbest_fitness = iter_results["gbest_fitness"]
#         is_first_feasible_now = not found_first_feasible and new_gbest_fitness < 1e6
#         if is_first_feasible_now: found_first_feasible = True
#         reward = calculate_reward(old_gbest_fitness, new_gbest_fitness, iter_results["gbest_diagnostics"], is_first_feasible_now, config)
#         next_state = get_state(swarm_state, i + 1, total_iterations, gbest_history)
#         done = (i == total_iterations - 1)
#         agent.replay_buffer.add(state, action_vector, reward, next_state, done)
#         agent.learn()
#         state, gbest_fitness = next_state, new_gbest_fitness
#         gbest_history.append(gbest_fitness)
#         fitness_history.append(gbest_fitness)
#     return {"best_solution": swarm_state["gbest_position"], "best_fitness": gbest_fitness, "fitness_history": fitness_history}

def run_sac_guided_pso(config, env, total_iterations, run_seed):
    print(f"\n--- Running SAC-Guided-PSO (Run #{run_seed+1}) ---")
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    agent = SACAgent(config)
    swarm_state = pso_core_sac.initialize_swarm(config)

    gbest_fitness = swarm_state["gbest_fitness"]
    fitness_history = []
    gbest_history = collections.deque(maxlen=10)
    found_first_feasible = False

    # 初始状态
    state = get_state(swarm_state, 0, total_iterations, gbest_history)

    # ========= 准备本次 run 的日志目录和 CSV =========
    # 你可以按自己习惯改路径；这里放在项目下 runs_pso_sac/ 里
    run_dir = os.path.join("runs_pso_sac", f"sac_pso_seed{run_seed}")
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "iter_log.csv")

    # 我们记录：迭代数、reward、LCOE、动作两个维度、状态四个维度
    fieldnames = [
        "iter",
        "reward",
        "lcoe",
        "treat_prob",
        "gamma_fb",
        "state0",
        "state1",
        "state2",
        "state3",
    ]

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(total_iterations):
            # 1) SAC 选择动作：action = [treat_prob, gamma_feedback]
            action_vector = agent.choose_action(state)

            old_gbest_fitness = gbest_fitness

            # 2) PSO 执行一次 guided 迭代
            swarm_state, iter_results = pso_core_sac.run_pso_iteration_guided(
                swarm_state, action_vector, env
            )
            new_gbest_fitness = iter_results["gbest_fitness"]

            # 3) 判定是否首次找到可行解
            is_first_feasible_now = (not found_first_feasible) and (new_gbest_fitness < 1e6)
            if is_first_feasible_now:
                found_first_feasible = True

            # 4) 计算奖励
            reward = calculate_reward(
                old_gbest_fitness,
                new_gbest_fitness,
                iter_results["gbest_diagnostics"],
                is_first_feasible_now,
                config,
            )

            # 5) 下一个状态
            next_state = get_state(swarm_state, i + 1, total_iterations, gbest_history)
            done = (i == total_iterations - 1)

            # 6) SAC 经验回放 + 学习
            agent.replay_buffer.add(state, action_vector, reward, next_state, done)
            agent.learn()

            # 7) 记录一条 CSV
            # 这里的 state 是“当前代的状态”，reward 是从 state→next_state 的回报
            writer.writerow({
                "iter": i,
                "reward": float(reward),
                "lcoe": float(new_gbest_fitness),
                "treat_prob": float(action_vector[0]),
                "gamma_fb": float(action_vector[1]),
                "state0": float(state[0]),
                "state1": float(state[1]),
                "state2": float(state[2]),
                "state3": float(state[3]),
            })

            # 8) 更新循环变量
            state = next_state
            gbest_fitness = new_gbest_fitness
            gbest_history.append(gbest_fitness)
            fitness_history.append(gbest_fitness)

    print(f"[SAC-Guided-PSO] log saved to: {csv_path}")

    return {
        "best_solution": swarm_state["gbest_position"],
        "best_fitness": gbest_fitness,
        "fitness_history": fitness_history,
    }

# def run_full_comparison():
#     NUM_INDEPENDENT_RUNS = 1
#     config = get_config()
#     total_iterations = config["simulation_params"]["total_iterations"]
    
#     typical_days_data = load_typical_days_data()
#     if typical_days_data is None: return
#     env = MicrogridEnv(config, typical_days_data)

#     # 存储所有运行的结果
#     all_sac_results, all_guided_results, all_simple_results = [], [], []
    
#     start_time = time.time()
#     for i in range(NUM_INDEPENDENT_RUNS):
#         print(f"\n{'='*25} Starting Statistical Run #{i+1}/{NUM_INDEPENDENT_RUNS} {'='*25}")
#         all_sac_results.append(run_sac_guided_pso(config, env, total_iterations, i))
#         all_guided_results.append(run_classic_pso_guided(config, env, total_iterations, i))
#         all_simple_results.append(run_classic_pso_simple(config, env, total_iterations, i))
#     end_time = time.time()
    
#     print(f"\n\n{'='*20} STATISTICAL ANALYSIS COMPLETE {'='*20}")
#     print(f"Total time for {NUM_INDEPENDENT_RUNS} runs: {end_time - start_time:.2f} seconds")

#     # 整理数据
#     sac_histories = [res["fitness_history"] for res in all_sac_results]
#     guided_histories = [res["fitness_history"] for res in all_guided_results]
#     simple_histories = [res["fitness_history"] for res in all_simple_results]
    
#     final_fitness_data = {
#         "SAC-Guided PSO": [h[-1] for h in sac_histories],
#         "Classic Guided PSO": [h[-1] for h in guided_histories],
#         "Simplest Classic PSO": [h[-1] for h in simple_histories]
#     }

#     # 绘图
#     plot_statistical_results(
#         histories_list=[sac_histories, guided_histories, simple_histories],
#         labels=['Guided PSO with SAC', 'Guided PSO', ' PSO']
#     )
#     plot_final_fitness_boxplot(final_fitness_data)
#     # 绘图

#         # 打印三种算法整体最优配置
#     print_best_config(all_sac_results,    "SAC-Guided PSO (Proposed)")
#     print_best_config(all_guided_results, "Classic Guided PSO")
#     print_best_config(all_simple_results, "Simplest Classic PSO")

def run_full_comparison():
    config = get_config()
    total_iterations = config["simulation_params"]["total_iterations"]
    num_runs = int(config["simulation_params"].get("num_independent_runs", 1))
    eps = float(config.get("simulation_params", {}).get("time_to_target_eps", 0.01))
    base_seed = int(config.get("simulation_params", {}).get("base_seed", 42))

    typical_days_data = load_typical_days_data()
    if not typical_days_data:
        return

    run_dir = os.path.join("runs", time.strftime("pso_%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    timing_csv = os.path.join(run_dir, "timing_stats.csv")
    print("Logs to:", os.path.abspath(run_dir))

    results = collections.defaultdict(list)
    start_time = time.time()

    for i in range(num_runs):
        print(f"\n{'='*25} Starting Statistical Run #{i+1}/{num_runs} {'='*25}")
        seed = base_seed + i

        # Fresh env per run + wrap for EMS counters
        base_env = MicrogridEnv(config, typical_days_data)
        env = EvalCounterEnv(base_env)

        timing_rows = []

        def _run_one(method_name, fn):
            env.reset_counters()
            t0 = time.perf_counter()
            res = fn(config, env, total_iterations, seed)
            wall_full = time.perf_counter() - t0

            res = attach_full_and_target_metrics(
                res,
                wall_full_s=wall_full,
                ems_eval_count_full=env.ems_eval_count,
                ems_eval_time_full_s=env.ems_eval_time_s,
                eps=eps,
            )
            results[method_name].append(res)

            row = {
                "run": i,
                "seed": seed,
                "method": method_name,
                "wall_time_s_full": res.get("wall_time_s_full"),
                "ems_eval_count_full": res.get("ems_eval_count_full"),
                "ems_eval_time_s_full": res.get("ems_eval_time_s_full"),
                "tgt_iter": res.get("tgt_iter"),
                "best_final": res.get("best_final"),
                "tgt_threshold": res.get("tgt_threshold"),
                "wall_time_s_tgt": res.get("wall_time_s_tgt"),
                "ems_eval_count_tgt": res.get("ems_eval_count_tgt"),
                "ems_eval_time_s_tgt": res.get("ems_eval_time_s_tgt"),
            }
            return row

        # 1) SAC-Guided PSO (proposed RL-adjusted operator)
        timing_rows.append(_run_one("PSO with proposed RL-adjusted operator", run_sac_guided_pso))

        # 2) Classic Guided PSO (proposed static operator)
        timing_rows.append(_run_one("PSO with proposed static operator", run_classic_pso_guided))

        # 3) PSO baseline
        timing_rows.append(_run_one("PSO", run_classic_pso_simple))

        # append timing rows
        df_timing = pd.DataFrame(timing_rows)
        if not os.path.exists(timing_csv):
            df_timing.to_csv(timing_csv, index=False)
        else:
            df_timing.to_csv(timing_csv, mode="a", header=False, index=False)

    print(f"\n\n{'='*20} STATISTICAL ANALYSIS COMPLETE {'='*20}")
    print(f"Total time for {num_runs} runs: {time.time() - start_time:.2f} seconds")
    print(f"Timing stats saved → {os.path.abspath(timing_csv)}")

    # Keep plotting behavior
    histories = {name: [res.get("fitness_history", []) for res in res_list] for name, res_list in results.items()}
    final_fits = {name: [h[-1] for h in h_list if len(h) > 0] for name, h_list in histories.items()}

    plot_statistical_results(histories_list=list(histories.values()), labels=list(histories.keys()))
    plot_final_fitness_boxplot(final_fits)

    # Optional: print best config for each method (与你现在的函数保持一致)
    print_best_config(results.get("PSO with proposed RL-adjusted operator", []), "SAC-Guided PSO (Proposed)")
    print_best_config(results.get("PSO with proposed static operator", []), "Classic Guided PSO")
    print_best_config(results.get("PSO", []), "Simplest Classic PSO")

if __name__ == "__main__":
    run_full_comparison()
