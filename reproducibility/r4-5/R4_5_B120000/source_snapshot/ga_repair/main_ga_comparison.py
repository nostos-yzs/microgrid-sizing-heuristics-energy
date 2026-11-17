# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 10:10:38 2025

@author: U2019
"""
#存在的问题 pso诊断的参数都统一了而不是分开 classic和guide参数要分开在用sac 要把参数统一基本统一至少是 总之需要把算法流程过一遍
# main_ga_comparison.py
import pandas as pd
import os
import numpy as np
import collections
import torch
import time
import random

# 引入所有需要的模块
from config_ga import get_config
from microgrid_env import MicrogridEnv
from sac_agent import SACAgent
import ga_core
from classic_ga_simple import run_classic_ga_simple
from classic_ga_guided import run_classic_ga_guided
# from utils import plot_statistical_results, plot_final_fitness_boxplot

# --- main_ga_comparison.py 顶部 imports 旁边 ---
from utils import plot_statistical_results, plot_final_fitness_boxplot, ExperimentLogger,BestConfigWriter

# --- SAC专用的辅助函数 ---
STATUS_MAP = {"success": 0, "fail_supply": 1, "fail_storage_overdischarge": 2, 
              "fail_storage_overcharge": 3, "simulation_error": 4}

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
    
def get_state_ga(population, iteration, total_iterations, gbest_history, gbest_diagnostics):
    iter_ratio = iteration / total_iterations
    diversity = np.std(np.array(population), axis=0).mean()
    normalized_diversity = min(diversity / 100.0, 1.0)
    convergence_rate = 0
    if len(gbest_history) > 10 and gbest_history[0] < 1e6:
        improvement = gbest_history[0] - gbest_history[-1]
        convergence_rate = improvement / gbest_history[0] if gbest_history[0] > 0 else 0
    status_str = gbest_diagnostics.get("status", "success") if gbest_diagnostics else "success"
    status_code = STATUS_MAP.get(status_str, 0) / len(STATUS_MAP)
    return np.array([iter_ratio, normalized_diversity, convergence_rate, status_code])

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

# # --- 实验运行函数 ---
# def run_sac_guided_ga(config, env, total_iterations, run_seed):
    
    
#     print(f"\n--- Running SAC-Guided-GA (Run #{run_seed+1}) ---")
#     np.random.seed(run_seed); torch.manual_seed(run_seed); random.seed(run_seed)
    
#     agent = SACAgent(config) # SACAgent现在默认使用config中的action_bounds
    
#     pop, min_b, max_b = ga_core.initialize_population(config)
#     gbest_fit, gbest_sol, gbest_diag = float('inf'), None, None
#     fit_hist, gbest_hist = [], collections.deque(maxlen=20)
#     found_feasible = False

#     state = get_state_ga(pop, 0, total_iterations, list(gbest_hist), gbest_diag)

#     for i in range(total_iterations):
#         action_vector = agent.choose_action(state)
#         old_gbest_fit = gbest_fit
        
#         eval_res = ga_core.evaluate_population(pop, env)
#         curr_best = min(eval_res, key=lambda x: x['fitness'])
        
#         is_first_feasible = not found_feasible and curr_best['fitness'] < 1e6
#         if is_first_feasible: found_feasible = True
        
#         if curr_best['fitness'] < gbest_fit:
#             gbest_fit, gbest_sol, gbest_diag = curr_best['fitness'], curr_best['individual'].copy(), curr_best['diagnostics']

#         reward = calculate_reward(old_gbest_fit, gbest_fit, gbest_diag, is_first_feasible, config)
        
#         pop = ga_core.run_ga_iteration_guided(pop, eval_res, action_vector, min_b, max_b)
        
#         next_state = get_state_ga(pop, i + 1, total_iterations, list(gbest_hist), gbest_diag)
#         agent.replay_buffer.add(state, action_vector, reward, next_state, (i == total_iterations - 1))
#         agent.learn()
        
#         state = next_state
#         gbest_hist.append(gbest_fit)
#         fit_hist.append(gbest_fit)
        
#     return {"best_solution": gbest_sol, "best_fitness": gbest_fit, "fitness_history": fit_hist}

# --- 实验运行函数（可整段替换） ---
def run_sac_guided_ga(config, env, total_iterations, run_seed):
    import numpy as np, torch, random, collections
    try:
        from utils import ExperimentLogger
        _has_logger = True
    except Exception:
        ExperimentLogger = None
        _has_logger = False

    print(f"\n--- Running SAC-Guided-GA (Run #{run_seed+1}) ---")
    np.random.seed(run_seed); torch.manual_seed(run_seed); random.seed(run_seed)

    agent = SACAgent(config)

    pop, min_b, max_b = ga_core.initialize_population(config)
    gbest_fit, gbest_sol, gbest_diag = float('inf'), None, None
    fit_hist, gbest_hist = [], collections.deque(maxlen=20)
    found_feasible = False

    logger = ExperimentLogger(outdir=f"runs/ga_sac_seed{run_seed}") if _has_logger else None
    best_tag_counter = 0

    # 初始状态
    state = get_state_ga(pop, 0, total_iterations, list(gbest_hist), gbest_diag)
    
    first_bonus_paid = False


    for i in range(total_iterations):
        # 1) SAC 动作
        action_vector = agent.choose_action(state)

        # 2) 评估父代
        eval_res_old = ga_core.evaluate_population(pop, env)
        curr_best_old = min(eval_res_old, key=lambda x: x['fitness'])

        old_gbest_fit = gbest_fit
        # 仅当 success 且 fitness<1e6 才能更新 gbest
        if (not found_feasible) and (curr_best_old['fitness'] < 1e6) and (curr_best_old['diagnostics'].get("status") == "success"):
            found_feasible = True
        if (curr_best_old['fitness'] < gbest_fit) and (curr_best_old['fitness'] < 1e6) and (curr_best_old['diagnostics'].get("status") == "success"):
            gbest_fit = curr_best_old['fitness']
            gbest_sol = curr_best_old['individual'].copy()
            gbest_diag = curr_best_old['diagnostics']

        # 3) 产生子代
        #3) 产生子代（← 这里把 env 作为最后一个参数传进去）
        pop = ga_core.run_ga_iteration_guided(pop, eval_res_old, action_vector, min_b, max_b, env)

        # 4) 评估子代（动作效果）
        eval_res_new = ga_core.evaluate_population(pop, env)
        curr_best_new = min(eval_res_new, key=lambda x: x['fitness'])

        if (curr_best_new['fitness'] < gbest_fit) and (curr_best_new['fitness'] < 1e6) and (curr_best_new['diagnostics'].get("status") == "success"):
            gbest_fit = curr_best_new['fitness']
            gbest_sol = curr_best_new['individual'].copy()
            gbest_diag = curr_best_new['diagnostics']

        # 5) 奖励
        
        # 计算“本轮是否第一次进入可行域”（瞬时布尔）
        is_feasible_now = (
            (curr_best_new['fitness'] < 1e6) and
            (curr_best_new['diagnostics'].get("status") == "success")
        )
        is_first_feasible = (not first_bonus_paid) and is_feasible_now
        if is_first_feasible:
            first_bonus_paid = True

        # 建议改为（用最新一轮的诊断，而不是 gbest 的诊断）：
        reward = calculate_reward(
            old_gbest_fit,
            gbest_fit,
            curr_best_new['diagnostics'],   # ← 关键修改
            is_first_feasible,
            config
        )

        # 6) 经验 & 学习
        next_state = get_state_ga(pop, i + 1, total_iterations, list(gbest_hist), gbest_diag)
        done_flag = (i == total_iterations - 1)
        agent.replay_buffer.add(state, action_vector, reward, next_state, done_flag)
        agent.learn()

        # 7) 历史
        state = next_state
        gbest_hist.append(gbest_fit)
        fit_hist.append(gbest_fit)

        # 8) 日志与 trace（仅 success 的 gbest 保存）
        if logger is not None:
            conv_rate = 0.0
            if len(gbest_hist) > 1 and gbest_hist[0] < 1e6:
                base0 = gbest_hist[0]
                imp = base0 - gbest_hist[-1]
                conv_rate = (imp / base0) if base0 > 0 else 0.0
            try:
                logger.log_iter(
                    iteration=i, total_iterations=total_iterations,
                    action_vec=action_vector,
                    gbest=gbest_fit, curr_best=curr_best_new['fitness'],
                    reward=reward, ev_res=eval_res_new,
                    gbest_diag=gbest_diag, conv_rate=conv_rate
                )
            except Exception:
                pass

            if (gbest_fit < old_gbest_fit) and (gbest_fit < 1e6) and (gbest_sol is not None) and (gbest_diag and gbest_diag.get("status") == "success"):
                try:
                    df_trace = env.simulate_day_trace(gbest_sol, day_index=0, stochastic=False)
                    logger.save_trace_df(df_trace, tag=f"{i:04d}_{best_tag_counter}")
                    best_tag_counter += 1
                except Exception:
                    pass

    return {"best_solution": gbest_sol, "best_fitness": gbest_fit, "diagnostics": gbest_diag, "fitness_history": fit_hist}



def run_ga_experiments():
    config = get_config()
    total_iterations = config["simulation_params"]["total_iterations"]
    num_runs = config["simulation_params"]["num_independent_runs"]

    typical_days_data = load_typical_days_data()
    if not typical_days_data: return
    env = MicrogridEnv(config, typical_days_data)

    # 新增：每次统计实验输出到一个时间戳目录
    run_dir = os.path.join("runs", time.strftime("ga_%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print("Logs to:", os.path.abspath(run_dir))
    best_writer = BestConfigWriter(outdir=run_dir)

    results = collections.defaultdict(list)
    start_time = time.time()
    base_seed=42
    for i in range(num_runs):
        print(f"\n{'='*25} Starting Statistical Run #{i+1}/{num_runs} {'='*25}")

        # 1) SAC-Guided
        sac_res = run_sac_guided_ga(config, env, total_iterations, base_seed)
        results["Guided GA with SAC"].append(sac_res)
        # 记录最优解配置（若缺 diagnostics，用复评补齐）
        sac_sol, sac_fit, sac_diag = sac_res.get("best_solution"), sac_res.get("best_fitness"), sac_res.get("diagnostics")
        if sac_diag is None and sac_sol is not None:
            try:
                _, sac_diag = env._run_single_simulation(sac_sol, stochastic=False)
            except Exception:
                sac_diag = {"status":"unknown"}
        best_writer.append("Guided GA with SAC", i, (sac_diag or {}).get("status","unknown"), sac_fit if sac_fit is not None else float("inf"), sac_sol)

        # 2) Classic Guided
        cg_res = run_classic_ga_guided(config, env, total_iterations, base_seed)
        results["Guided GA"].append(cg_res)
        cg_sol, cg_fit, cg_diag = cg_res.get("best_solution"), cg_res.get("best_fitness"), cg_res.get("diagnostics")
        if cg_diag is None and cg_sol is not None:
            try:
                _, cg_diag = env._run_single_simulation(cg_sol, stochastic=False)
            except Exception:
                cg_diag = {"status":"unknown"}
        best_writer.append("Guided GA", i, (cg_diag or {}).get("status","unknown"), cg_fit if cg_fit is not None else float("inf"), cg_sol)

        # 3) Classic Simple
        cs_res = run_classic_ga_simple(config, env, total_iterations, base_seed)
        results["GA"].append(cs_res)
        cs_sol, cs_fit, cs_diag = cs_res.get("best_solution"), cs_res.get("best_fitness"), cs_res.get("diagnostics")
        if cs_diag is None and cs_sol is not None:
            try:
                _, cs_diag = env._run_single_simulation(cs_sol, stochastic=False)
            except Exception:
                cs_diag = {"status":"unknown"}
        best_writer.append("GA", i, (cs_diag or {}).get("status","unknown"), cs_fit if cs_fit is not None else float("inf"), cs_sol)

    print(f"\n\n{'='*20} STATISTICAL ANALYSIS COMPLETE {'='*20}")
    print(f"Total time for {num_runs} runs: {time.time() - start_time:.2f} seconds")
    print(f"Best-configs saved → {os.path.abspath(os.path.join(run_dir,'best_configs.csv'))}")

    # 整理数据绘图（原样）
    histories = {name: [res.get("fitness_history",[]) for res in res_list] for name, res_list in results.items()}
    final_fits = {name: [h[-1] for h in h_list if len(h)>0] for name, h_list in histories.items()}

    plot_statistical_results(histories_list=list(histories.values()), labels=list(histories.keys()))
    plot_final_fitness_boxplot(final_fits)


if __name__ == "__main__":
    run_ga_experiments()