"""Online search/training functions extracted from the archived comparison entry.
Only progress printing, iteration logging and trace exports have been removed.
Search, state, action mapping, reward and learning statements are preserved.
"""
import collections
import random
import numpy as np
import torch
from sac_agent import SACAgent
import ts_core
STATUS_MAP = {'success': 0, 'fail_supply': 1, 'fail_storage_overdischarge': 2, 'fail_storage_overcharge': 3, 'simulation_error': 4}

def set_global_seed(seed: int) -> None:
    import torch
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def get_state_ts(state, iteration, total_iterations, gbest_hist):
    iter_ratio = iteration / total_iterations
    normalized_sigma = min(max(state.get('sigma_ratio', 0.0) / 0.5, 0.0), 1.0)
    convergence_rate = 0.0
    hist = list(gbest_hist)
    if len(hist) > 10 and hist[0] < 1000000.0:
        improvement = hist[0] - hist[-1]
        convergence_rate = improvement / max(hist[0], 1e-09)
    status = state.get('best_diag', {}).get('status', 'success')
    status_code = STATUS_MAP.get(status, 0) / len(STATUS_MAP)
    return np.array([iter_ratio, normalized_sigma, convergence_rate, status_code], dtype=float)

def calculate_reward(old_best, new_best, diagnostics, first_feasible, config):
    p = config['sac_agent_params']
    if first_feasible:
        return p['rewards']['first_feasible_bonus']
    r = 0.0
    if new_best < old_best:
        imp = (old_best - new_best) / max(old_best, 1e-09) if old_best < 1000000.0 else 0.0
        r += p['rewards']['gbest_improvement'] * np.tanh(10 * imp)
    else:
        r += p['rewards']['stagnation']
    status = diagnostics.get('status', 'success') if diagnostics else 'success'
    if status != 'success':
        r += p['penalties'].get(status, 0)
    return float(r)

def run_sac_guided_ts(config, env, total_iterations, run_seed):
    np.random.seed(run_seed)
    random.seed(run_seed)
    agent = SACAgent(config)
    state = ts_core.initialize_tabu_state(config, env, seed=run_seed)
    fit_hist, gbest_hist = ([], collections.deque(maxlen=20))
    found_feasible = state.get('best_f', float('inf')) < 1000000.0
    s = get_state_ts(state, 0, total_iterations, gbest_hist)
    for i in range(total_iterations):
        act = agent.choose_action(s)
        old_best = float(state.get('best_f', float('inf')))
        state = ts_core.run_tabu_iteration_guided(state, act, env)
        is_first_feasible = not found_feasible and float(state.get('best_f', float('inf'))) < 1000000.0
        if is_first_feasible:
            found_feasible = True
        r = calculate_reward(old_best, float(state.get('best_f', float('inf'))), state.get('best_diag'), is_first_feasible, config)
        s2 = get_state_ts(state, i + 1, total_iterations, gbest_hist)
        agent.replay_buffer.add(s, act, r, s2, i == total_iterations - 1)
        agent.learn()
        diag_prob = float(act[0])
        gamma_fb = float(act[1])
        lcoe = float(state.get('best_f', float('inf')))
        s = s2
        gbest_hist.append(lcoe)
        fit_hist.append(lcoe)
    return {'best_solution': state.get('best_x'), 'best_fitness': state.get('best_f'), 'diagnostics': state.get('best_diag'), 'fitness_history': fit_hist}
