# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:53:52 2025

@author: Lenovo
"""

# ts/classic_tabu_simple.py
import numpy as np, random, ts_core

def run_classic_tabu_simple(config, env, total_iterations, run_seed):
    print(f"\n--- Running Classic Simple TS (Run #{run_seed+1}) ---")
    np.random.seed(run_seed); random.seed(run_seed)

    state = ts_core.initialize_tabu_state(config, env, seed=run_seed)
    fit_hist = [state["best_f"]]

    for _ in range(total_iterations):
        state = ts_core.run_tabu_iteration_simple(state, env)
        fit_hist.append(state["best_f"])

    print(f"--- Classic Simple TS (Run #{run_seed+1}) Finished ---")
    return {"best_solution": state["best_x"], "best_fitness": state["best_f"], "fitness_history": fit_hist}
