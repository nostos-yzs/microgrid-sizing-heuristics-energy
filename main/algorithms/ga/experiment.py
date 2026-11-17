"""Online search/training functions extracted from the archived comparison entry.
Only progress printing, iteration logging and trace exports have been removed.
Search, state, action mapping, reward and learning statements are preserved.
"""
import collections
import random
import numpy as np
import torch
from sac_agent import SACAgent
import ga_core
STATUS_MAP = {'success': 0, 'fail_supply': 1, 'fail_storage_overdischarge': 2, 'fail_storage_overcharge': 3, 'simulation_error': 4}

def get_state_ga(population, iteration, total_iterations, gbest_history, gbest_diagnostics):
    iter_ratio = iteration / total_iterations
    diversity = np.std(np.array(population), axis=0).mean()
    normalized_diversity = min(diversity / 100.0, 1.0)
    convergence_rate = 0
    if len(gbest_history) > 10 and gbest_history[0] < 1000000.0:
        improvement = gbest_history[0] - gbest_history[-1]
        convergence_rate = improvement / gbest_history[0] if gbest_history[0] > 0 else 0
    status_str = gbest_diagnostics.get('status', 'success') if gbest_diagnostics else 'success'
    status_code = STATUS_MAP.get(status_str, 0) / len(STATUS_MAP)
    return np.array([iter_ratio, normalized_diversity, convergence_rate, status_code])

def calculate_reward(old_gbest, new_gbest, diagnostics, first_feasible, config):
    params = config['sac_agent_params']
    reward = 0.0
    if new_gbest < old_gbest:
        improvement = (old_gbest - new_gbest) / old_gbest if 0 < old_gbest < 1000000.0 else 0.0
        reward += params['rewards']['gbest_improvement'] * np.tanh(improvement * 10.0)
    else:
        reward += params['rewards']['stagnation']
    if first_feasible:
        reward += params['rewards']['first_feasible_bonus']
    status = diagnostics.get('status', 'success') if diagnostics else 'success'
    if status != 'success':
        reward += params['penalties'].get(status, 0)
    return float(reward)

def run_sac_guided_ga(config, env, total_iterations, run_seed):
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    random.seed(run_seed)
    agent = SACAgent(config)
    pop, min_b, max_b = ga_core.initialize_population(config)
    gbest_fit, gbest_sol, gbest_diag = (float('inf'), None, None)
    fit_hist, gbest_hist = ([], collections.deque(maxlen=20))
    state = get_state_ga(pop, 0, total_iterations, list(gbest_hist), gbest_diag)
    first_bonus_paid = False
    for i in range(total_iterations):
        action_vector = agent.choose_action(state)
        eval_res_old = ga_core.evaluate_population(pop, env)
        curr_best_old = min(eval_res_old, key=lambda x: x['fitness'])
        old_gbest_fit = gbest_fit
        if curr_best_old['fitness'] < gbest_fit and curr_best_old['fitness'] < 1000000.0 and (curr_best_old['diagnostics'].get('status') == 'success'):
            gbest_fit = curr_best_old['fitness']
            gbest_sol = curr_best_old['individual'].copy()
            gbest_diag = curr_best_old['diagnostics']
        pop = ga_core.run_ga_iteration_guided(pop, eval_res_old, action_vector, min_b, max_b, env)
        eval_res_new = ga_core.evaluate_population(pop, env)
        curr_best_new = min(eval_res_new, key=lambda x: x['fitness'])
        if curr_best_new['fitness'] < gbest_fit and curr_best_new['fitness'] < 1000000.0 and (curr_best_new['diagnostics'].get('status') == 'success'):
            gbest_fit = curr_best_new['fitness']
            gbest_sol = curr_best_new['individual'].copy()
            gbest_diag = curr_best_new['diagnostics']
        is_feasible_now = curr_best_new['fitness'] < 1000000.0 and curr_best_new['diagnostics'].get('status') == 'success'
        is_first_feasible = not first_bonus_paid and is_feasible_now
        if is_first_feasible:
            first_bonus_paid = True
        reward = calculate_reward(old_gbest_fit, gbest_fit, curr_best_new['diagnostics'], is_first_feasible, config)
        next_state = get_state_ga(pop, i + 1, total_iterations, list(gbest_hist), gbest_diag)
        done_flag = i == total_iterations - 1
        agent.replay_buffer.add(state, action_vector, reward, next_state, done_flag)
        agent.learn()
        state = next_state
        gbest_hist.append(gbest_fit)
        fit_hist.append(gbest_fit)
    return {'best_solution': gbest_sol, 'best_fitness': gbest_fit, 'diagnostics': gbest_diag, 'fitness_history': fit_hist}
