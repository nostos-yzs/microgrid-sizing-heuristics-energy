import numpy as np
STATUS_MAP = {
    "success": 0, "fail_supply": 1, "fail_storage_overdischarge": 2,
    "fail_storage_overcharge": 3, "simulation_error": 4
}

def get_state(swarm_state, iteration, total_iterations, gbest_history):
    iter_ratio = iteration / total_iterations
    diversity = np.std(swarm_state["particles"], axis=0).mean()
    normalized_diversity = min(diversity / 100.0, 1.0)
    convergence_rate = 0
    if len(gbest_history) > 1 and gbest_history[0] < 1e6:
        improvement = gbest_history[0] - gbest_history[-1]
        convergence_rate = improvement / gbest_history[0] if gbest_history[0] > 0 else 0
    diagnostics = swarm_state["gbest_diagnostics"]
    status_str = diagnostics.get("status", "success") if diagnostics else "success"
    status_code = STATUS_MAP.get(status_str, 0) / len(STATUS_MAP)
    return np.array([iter_ratio, normalized_diversity, convergence_rate, status_code])

def calculate_reward(old_gbest, new_gbest, diagnostics, first_feasible, config):
    params = config["sac_agent_params"]
    reward = 0.0

    # (A) gbest 改善 / 停滞
    if new_gbest < old_gbest:
        improvement = (old_gbest - new_gbest) / old_gbest if 0 < old_gbest < 1e6 else 0.0
        reward += params["rewards"]["gbest_improvement"] * np.tanh(improvement * 10.0)
    else:
        reward += params["rewards"]["stagnation"]

    # (B) 一次性“首次可行”奖励：仅当本轮第一次进入可行域时叠加
    if first_feasible:
        reward += params["rewards"]["first_feasible_bonus"]

    # (C) 失败类型罚分（用“本轮”的 diagnostics，见步骤 3 的传参）
    status = diagnostics.get("status", "success") if diagnostics else "success"
    if status != "success":
        reward += params["penalties"].get(status, 0)

    return float(reward)
