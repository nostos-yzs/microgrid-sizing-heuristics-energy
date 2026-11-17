# classic_pso_guided.py
import pso_core_sac as pso_core
import numpy as np

def run_classic_pso_guided(config, env, total_iterations, run_seed):
    print(f"\n--- Running Classic Guided PSO (Run #{run_seed+1}) ---")
    np.random.seed(run_seed)

    guided_cfg = config["classic_guided_params"]
    classic_op_vector = np.array([
        guided_cfg["treat_prob"],
        guided_cfg["gamma_feedback"],
    ])

    swarm_state = pso_core.initialize_swarm(config)
    fitness_history = []
    
    for i in range(total_iterations):
        swarm_state, iter_results = pso_core.run_pso_iteration_guided(
            swarm_state, classic_op_vector, env
        )
        current_gbest_fitness = iter_results["gbest_fitness"]
        fitness_history.append(current_gbest_fitness)

    print(f"--- Classic Guided PSO (Run #{run_seed+1}) Finished ---")
    return {
        "best_solution": swarm_state["gbest_position"],
        "best_fitness": swarm_state["gbest_fitness"],
        "fitness_history": fitness_history
    }
