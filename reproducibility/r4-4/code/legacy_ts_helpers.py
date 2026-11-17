import numpy as np
STATUS_MAP = {
    "success": 0,
    "fail_supply": 1,
    "fail_storage_overdischarge": 2,
    "fail_storage_overcharge": 3,
    "simulation_error": 4,
}

def get_state_ts(state, iteration, total_iterations, gbest_hist):
    iter_ratio = iteration / total_iterations
    normalized_sigma = min(max(state.get("sigma_ratio", 0.0) / 0.5, 0.0), 1.0)
    convergence_rate = 0.0
    hist = list(gbest_hist)
    if len(hist) > 10 and hist[0] < 1e6:
        improvement = hist[0] - hist[-1]
        convergence_rate = improvement / max(hist[0], 1e-9)
    status = state.get("best_diag", {}).get("status", "success")
    status_code = STATUS_MAP.get(status, 0) / len(STATUS_MAP)
    return np.array([iter_ratio, normalized_sigma, convergence_rate, status_code], dtype=float)

def calculate_reward(old_best, new_best, diagnostics, first_feasible, config):
    p = config["sac_agent_params"]
    if first_feasible:
        return p["rewards"]["first_feasible_bonus"]
    r = 0.0
    if new_best < old_best:
        imp = (old_best - new_best) / max(old_best, 1e-9) if old_best < 1e6 else 0.0
        r += p["rewards"]["gbest_improvement"] * np.tanh(10 * imp)
    else:
        r += p["rewards"]["stagnation"]
    status = diagnostics.get("status", "success") if diagnostics else "success"
    if status != "success":
        r += p["penalties"].get(status, 0)
    return float(r)
