# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 10:09:28 2025

@author: U2019
"""

# classic_ga_simple.py
import numpy as np
import random
import ga_core

def run_classic_ga_simple(config, env, total_iterations, run_seed):
    print(f"\n--- Running Simplest Classic GA (Run #{run_seed+1}) ---")
    np.random.seed(run_seed); random.seed(run_seed)

    params = {
        "crossover_rate": config["ga_params"]["classic_params"]["simple"]["crossover_rate"],
        "mutation_rate": config["ga_params"]["classic_params"]["simple"]["mutation_rate"],
        "mutation_strength": config["ga_params"]["classic_params"]["simple"]["mutation_strength"],
    }
    pop_size = config["ga_params"]["population_size"]
    
    pop, min_b, max_b = ga_core.initialize_population(config)
    gbest_fit, gbest_sol, fit_hist = float('inf'), None, []

    for _ in range(total_iterations):
        eval_res = ga_core.evaluate_population(pop, env)
        curr_best = min(eval_res, key=lambda x: x['fitness'])
        if curr_best['fitness'] < gbest_fit:
            gbest_fit, gbest_sol = curr_best['fitness'], curr_best['individual'].copy()
        fit_hist.append(gbest_fit)

        next_gen = [gbest_sol]
        while len(next_gen) < pop_size:
            p1, p2 = ga_core.tournament_selection(eval_res), ga_core.tournament_selection(eval_res)
            c1, c2 = ga_core.simulated_binary_crossover(p1['individual'], p2['individual'], min_b=min_b, max_b=max_b) \
                if random.random() < params["crossover_rate"] else (p1['individual'].copy(), p2['individual'].copy())
            
            c1 = ga_core.gaussian_mutation(c1, params["mutation_rate"], params["mutation_strength"], min_b, max_b)
            c2 = ga_core.gaussian_mutation(c2, params["mutation_rate"], params["mutation_strength"], min_b, max_b)
            next_gen.extend([c1, c2])
        pop = next_gen[:pop_size]
    
    print(f"--- Simplest Classic GA (Run #{run_seed+1}) Finished ---")
    return {"best_solution": gbest_sol, "best_fitness": gbest_fit, "fitness_history": fit_hist}