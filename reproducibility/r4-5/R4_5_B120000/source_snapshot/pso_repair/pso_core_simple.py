# pso_core_simple.py
import numpy as np

def initialize_swarm(config):
    """初始化6维粒子群。"""
    pop_size = config["pso_core_params"]["population_size"]
    pso_cfg = config["pso_core_params"]
    pop_size = pso_cfg["population_size"]

    min_bounds = np.array(pso_cfg["min_bounds"], dtype=float)
    max_bounds = np.array(pso_cfg["max_bounds"], dtype=float)
    particles = min_bounds + (max_bounds - min_bounds) * np.random.rand(pop_size, 6)
    velocities = (max_bounds - min_bounds) * 0.1 * (np.random.rand(pop_size, 6) - 0.5)
    pbest_positions = particles.copy()
    pbest_fitness = np.array([float('inf')] * pop_size)
    gbest_position, gbest_fitness = None, float('inf')
    return {
        "particles": particles, "velocities": velocities,
        "pbest_positions": pbest_positions, "pbest_fitness": pbest_fitness,
        "gbest_position": gbest_position, "gbest_fitness": gbest_fitness
    }

def run_pso_iteration_simple(swarm_state, pso_params, env):
    """
    执行一轮最经典的PSO迭代，完全忽略环境返回的diagnostics。
    """
    particles = swarm_state["particles"]
    velocities = swarm_state["velocities"]
    pbest_positions = swarm_state["pbest_positions"]
    pbest_fitness = swarm_state["pbest_fitness"]
    gbest_position = swarm_state["gbest_position"]
    gbest_fitness = swarm_state["gbest_fitness"]

    w, c1, c2 = pso_params["w"], pso_params["c1"], pso_params["c2"]
    pop_size = len(particles)

    for i in range(pop_size):
        # 只关心fitness，忽略diagnostics
        fitness, _ = env.evaluate_solution(particles[i].copy())
        
        if fitness < pbest_fitness[i]:
            pbest_fitness[i] = fitness
            pbest_positions[i] = particles[i].copy()
        if fitness < gbest_fitness:
            gbest_fitness = fitness
            gbest_position = particles[i].copy()

    if gbest_position is not None:
        r1, r2 = np.random.rand(2)
        cognitive_velocity = c1 * r1 * (pbest_positions - particles)
        social_velocity = c2 * r2 * (gbest_position - particles)
        velocities = w * velocities + cognitive_velocity + social_velocity
        particles += velocities
        particles = np.maximum(0, particles)

    updated_swarm_state = {
        "particles": particles, "velocities": velocities,
        "pbest_positions": pbest_positions, "pbest_fitness": pbest_fitness,
        "gbest_position": gbest_position, "gbest_fitness": gbest_fitness
    }
    iteration_results = {"gbest_fitness": gbest_fitness}
    return updated_swarm_state, iteration_results
