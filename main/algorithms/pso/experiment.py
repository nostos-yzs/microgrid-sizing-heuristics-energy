"""Online search/training functions extracted from the archived comparison entry.
Only progress printing, iteration logging and trace exports have been removed.
Search, state, action mapping, reward and learning statements are preserved.
"""
import collections
import random
import numpy as np
import torch
from sac_agent import SACAgent
import pso_core_sac
STATUS_MAP = {'success': 0, 'fail_supply': 1, 'fail_storage_overdischarge': 2, 'fail_storage_overcharge': 3, 'simulation_error': 4}

def get_state(swarm_state, iteration, total_iterations, gbest_history):
    iter_ratio = iteration / total_iterations
    diversity = np.std(swarm_state['particles'], axis=0).mean()
    normalized_diversity = min(diversity / 100.0, 1.0)
    convergence_rate = 0
    if len(gbest_history) > 1 and gbest_history[0] < 1000000.0:
        improvement = gbest_history[0] - gbest_history[-1]
        convergence_rate = improvement / gbest_history[0] if gbest_history[0] > 0 else 0
    diagnostics = swarm_state['gbest_diagnostics']
    status_str = diagnostics.get('status', 'success') if diagnostics else 'success'
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

def run_sac_guided_pso(config, env, total_iterations, run_seed):
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    agent = SACAgent(config)
    swarm_state = pso_core_sac.initialize_swarm(config)
    gbest_fitness = swarm_state['gbest_fitness']
    fitness_history = []
    gbest_history = collections.deque(maxlen=10)
    found_first_feasible = False
    state = get_state(swarm_state, 0, total_iterations, gbest_history)
    for i in range(total_iterations):
        action_vector = agent.choose_action(state)
        old_gbest_fitness = gbest_fitness
        swarm_state, iter_results = pso_core_sac.run_pso_iteration_guided(swarm_state, action_vector, env)
        new_gbest_fitness = iter_results['gbest_fitness']
        is_first_feasible_now = not found_first_feasible and new_gbest_fitness < 1000000.0
        if is_first_feasible_now:
            found_first_feasible = True
        reward = calculate_reward(old_gbest_fitness, new_gbest_fitness, iter_results['gbest_diagnostics'], is_first_feasible_now, config)
        next_state = get_state(swarm_state, i + 1, total_iterations, gbest_history)
        done = i == total_iterations - 1
        agent.replay_buffer.add(state, action_vector, reward, next_state, done)
        agent.learn()
        state = next_state
        gbest_fitness = new_gbest_fitness
        gbest_history.append(gbest_fitness)
        fitness_history.append(gbest_fitness)
    return {'best_solution': swarm_state['gbest_position'], 'best_fitness': gbest_fitness, 'fitness_history': fitness_history}
