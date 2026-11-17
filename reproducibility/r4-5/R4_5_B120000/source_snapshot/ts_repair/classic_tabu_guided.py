# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:54:17 2025

@author: Lenovo
"""

# ts/classic_tabu_guided.py
import numpy as np, random, ts_core

def run_classic_tabu_guided(config, env, total_iterations, run_seed):
    """
    经典 TS + 固定诊断参数（不接 SAC），用于和 simple/guided/PSO/GA 做对照。
    使用 config['ts_params']['guided'] 中的 diag_prob 与 gamma_feedback。
    """
    print(f"\n--- Running Classic Guided TS (Run #{run_seed+1}) ---")
    np.random.seed(run_seed); random.seed(run_seed)

    gcfg = config["ts_params"].get("guided", {})
    diag_prob    = float(gcfg.get("diag_prob", 0.6))
    gamma_fb     = float(gcfg.get("gamma_feedback", 30.0))
    action_vec   = [diag_prob, gamma_fb]

    state = ts_core.initialize_tabu_state(config, env, seed=run_seed)
    fit_hist = [state["best_f"]]

    for _ in range(total_iterations):
        state = ts_core.run_tabu_iteration_guided(state, action_vec, env)
        fit_hist.append(state["best_f"])

    print(f"--- Classic Guided TS (Run #{run_seed+1}) Finished ---")
    return {"best_solution": state["best_x"], "best_fitness": state["best_f"], "fitness_history": fit_hist}
