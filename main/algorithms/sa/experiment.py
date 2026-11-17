"""Online search/training functions extracted from the archived comparison entry.
Only progress printing, iteration logging and trace exports have been removed.
Search, state, action mapping, reward and learning statements are preserved.
"""
import collections
import random
import numpy as np
import torch
from sac_agent import SACAgent
from sa_core import initialize_sa_state as initialize_sa, run_sa_iteration
STATUS_MAP = {'success': 0, 'fail_supply': 1, 'fail_storage_overdischarge': 2, 'fail_storage_overcharge': 3, 'simulation_error': 4}

def get_state_sa(sa_state: dict, total_iterations: int) -> np.ndarray:
    iter_ratio = float(sa_state.get('iter', 0)) / max(1, total_iterations)
    normalized_diversity = 0.0
    f_best = float(sa_state.get('f_best', 1000000000.0))
    f_cur = float(sa_state.get('f', 1000000000.0))
    if f_cur <= 0 or f_best >= 100000000.0:
        conv = 0.0
    else:
        conv = 1.0 - min(1.0, f_best / (f_cur + 1e-09))
    status_str = (sa_state.get('diag_best') or {}).get('status', 'success')
    status_code = STATUS_MAP.get(status_str, 0) / max(1, len(STATUS_MAP))
    return np.array([iter_ratio, normalized_diversity, conv, status_code], dtype=float)

def calculate_reward_sa(old_gbest, new_gbest, diagnostics, first_feasible, cfg_rewards_penalties):
    params = cfg_rewards_penalties
    reward = 0.0
    if new_gbest < old_gbest:
        improvement = (old_gbest - new_gbest) / old_gbest if 0 < old_gbest < 1000000.0 else 0.0
        reward += params['rewards']['gbest_improvement'] * np.tanh(improvement * 10.0)
    else:
        reward += params['rewards']['stagnation']
    if first_feasible:
        reward = params['rewards']['first_feasible_bonus']
    status = diagnostics.get('status', 'success') if diagnostics else 'success'
    if status != 'success':
        reward += params['penalties'].get(status, 0)
    return float(reward)

def _read_layers(cfg):
    sim = cfg.get('simulation_params', {})
    n_layers = int(sim.get('n_layers', 30))
    inner_loops = int(sim.get('inner_loops', 1))
    return (n_layers, inner_loops)

def run_sa_traditional(config, env, total_iterations, run_seed):
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    random.seed(run_seed)
    min_b = config['min_bounds'].astype(float)
    max_b = config['max_bounds'].astype(float)
    sched = config['sa_params']['schedule']
    seed = int(config.get('experiments', {}).get('classic_simple', {}).get('random_seed', run_seed))
    sa = initialize_sa(env=env, min_b=min_b, max_b=max_b, schedule=sched, seed=seed, rw_cfg={'rewards': {}, 'penalties': {}})
    n_layers, inner_loops = _read_layers(config)
    T0, alpha, Tmin = (float(sched['T0']), float(sched['alpha']), float(sched['Tmin']))
    fixed_action = np.array([0.0, 0.0], dtype=float)
    fit_hist = []
    for layer in range(n_layers):
        sa['T'] = max(T0 * alpha ** layer, Tmin)
        for _ in range(inner_loops):
            sa = run_sa_iteration(sa, fixed_action, env)
            fit_hist.append(float(sa['f_best']))
    return {'best_solution': sa['x_best'], 'best_fitness': float(sa['f_best']), 'diagnostics': sa.get('diag_best', None), 'fitness_history': fit_hist}

def run_sa_guided(config, env, total_iterations, run_seed):
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    random.seed(run_seed)
    min_b = config['min_bounds'].astype(float)
    max_b = config['max_bounds'].astype(float)
    sched = config['sa_params']['schedule']
    exp = config['experiments']['classic_guided']
    gamma_fb = float(exp['gamma_fb'])
    diag_prob = float(exp['diag_prob'])
    seed = int(exp.get('random_seed', run_seed))
    sa = initialize_sa(env=env, min_b=min_b, max_b=max_b, schedule=sched, seed=seed, rw_cfg={'rewards': {}, 'penalties': {}})
    n_layers, inner_loops = _read_layers(config)
    T0, alpha, Tmin = (float(sched['T0']), float(sched['alpha']), float(sched['Tmin']))
    fixed_action = np.array([gamma_fb, diag_prob], dtype=float)
    fit_hist = []
    for layer in range(n_layers):
        sa['T'] = max(T0 * alpha ** layer, Tmin)
        for _ in range(inner_loops):
            sa = run_sa_iteration(sa, fixed_action, env)
            fit_hist.append(float(sa['f_best']))
    return {'best_solution': sa['x_best'], 'best_fitness': float(sa['f_best']), 'diagnostics': sa.get('diag_best', None), 'fitness_history': fit_hist}

def _map_to_bounds(action01: np.ndarray, bounds: dict) -> np.ndarray:
    out = []
    keys = ['gamma_fb', 'diag_prob']
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
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    random.seed(run_seed)
    min_b = config['min_bounds'].astype(float)
    max_b = config['max_bounds'].astype(float)
    sched = config['sa_params']['schedule']
    sac_cfg = config['experiments']['sac_guided']['sac_agent_params']
    bounds = sac_cfg['action_bounds']
    seed = int(config['experiments']['sac_guided'].get('random_seed', run_seed))
    cfg_for_agent = dict(config)
    cfg_for_agent['sac_agent_params'] = sac_cfg
    agent = SACAgent(cfg_for_agent)
    sa = initialize_sa(env=env, min_b=min_b, max_b=max_b, schedule=sched, seed=seed, rw_cfg=sac_cfg)
    n_layers, inner_loops = _read_layers(config)
    total_iterations = n_layers * inner_loops
    T0, alpha, Tmin = (float(sched['T0']), float(sched['alpha']), float(sched['Tmin']))
    fit_hist = []
    t = 0
    state = get_state_sa(sa, total_iterations)
    ever_feasible = (sa.get('diag_best') or {}).get('status', '') == 'success'
    for layer in range(n_layers):
        sa['T'] = max(T0 * alpha ** layer, Tmin)
        for _ in range(inner_loops):
            raw_action = agent.choose_action(state)
            if len(raw_action) != 2:
                raise RuntimeError(f'SACAgent.choose_action() must return 2 dims, got {len(raw_action)}')
            action = _map_to_bounds(np.asarray(raw_action, dtype=float), bounds)
            gamma_fb = float(action[0])
            diag_prob = float(action[1])
            old_best = float(sa['f_best'])
            sa = run_sa_iteration(sa, action, env)
            status_now = (sa.get('diag_best') or {}).get('status', '')
            if not ever_feasible and status_now == 'success':
                just_became_feasible = True
                ever_feasible = True
            else:
                just_became_feasible = False
            reward = calculate_reward_sa(old_best, float(sa['f_best']), sa.get('diag_best', {}), just_became_feasible, sac_cfg)
            next_state = get_state_sa(sa, total_iterations)
            done = t == total_iterations - 1
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            fit_hist.append(float(sa['f_best']))
            status = (sa.get('diag') or {}).get('status', 'unknown')
            t += 1
    return {'best_solution': sa['x_best'], 'best_fitness': float(sa['f_best']), 'diagnostics': sa.get('diag_best', None), 'fitness_history': fit_hist}
