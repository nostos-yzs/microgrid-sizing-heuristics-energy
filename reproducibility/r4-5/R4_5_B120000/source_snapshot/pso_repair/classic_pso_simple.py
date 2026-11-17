# classic_pso_simple.py
import pso_core_simple as pso_core
import numpy as np

def run_classic_pso_simple(config, env, total_iterations, run_seed):
    """
    运行最简化版的经典PSO。
    新增了 run_seed 参数以匹配主程序的调用。
    """
    print(f"\n--- Running Simplest Classic PSO (Run #{run_seed+1}) ---")
    np.random.seed(run_seed) # 设置随机种子

    # 使用一组固定的、经典的 w, c1, c2 参数
    pso_cfg = config["pso_core_params"]
    pso_params = {
        "w":  pso_cfg["w"],
        "c1": pso_cfg["c1"],
        "c2": pso_cfg["c2"],
    }
        
    swarm_state = pso_core.initialize_swarm(config)
    fitness_history = []
    
    for i in range(total_iterations):
        swarm_state, iter_results = pso_core.run_pso_iteration_simple(
            swarm_state, pso_params, env
        )
        fitness_history.append(iter_results["gbest_fitness"])
        # 减少打印频率
        # if (i + 1) % 20 == 0:
        #     print(f"  Run {run_seed+1}, Iter {i+1}/{total_iterations} | G-Best: {iter_results['gbest_fitness']:.4f}")

    print(f"--- Simplest Classic PSO (Run #{run_seed+1}) Finished ---")
    return {
        "best_solution": swarm_state["gbest_position"],
        "best_fitness": swarm_state["gbest_fitness"],
        "fitness_history": fitness_history
    }
