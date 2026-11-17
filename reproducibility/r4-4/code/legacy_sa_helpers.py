import numpy as np
STATUS_MAP = {
    "success": 0,
    "fail_supply": 1,
    "fail_storage_overdischarge": 2,
    "fail_storage_overcharge": 3,
    "simulation_error": 4,
}

def get_state_sa(sa_state: dict, total_iterations: int) -> np.ndarray:
    iter_ratio = float(sa_state.get("iter", 0)) / max(1, total_iterations)

    # 目前不使用多样性项（占位保持维度一致）
    normalized_diversity = 0.0

    f_best = float(sa_state.get("f_best", 1e9))
    f_cur = float(sa_state.get("f", 1e9))
    if f_cur <= 0 or f_best >= 1e8:
        conv = 0.0
    else:
        conv = 1.0 - min(1.0, f_best / (f_cur + 1e-9))

    status_str = (sa_state.get("diag_best") or {}).get("status", "success")
    status_code = STATUS_MAP.get(status_str, 0) / max(1, len(STATUS_MAP))

    return np.array([iter_ratio, normalized_diversity, conv, status_code], dtype=float)

def calculate_reward_sa(old_gbest, new_gbest, diagnostics, first_feasible, cfg_rewards_penalties):
    params = cfg_rewards_penalties
    reward = 0.0

    # (A) gbest 改善 / 停滞
    if new_gbest < old_gbest:
        improvement = (old_gbest - new_gbest) / old_gbest if 0 < old_gbest < 1e6 else 0.0
        reward += params["rewards"]["gbest_improvement"] * np.tanh(improvement * 10.0)
    else:
        reward += params["rewards"]["stagnation"]

    # (B) 一次性“首次可行”奖励：仅当本轮第一次进入可行域时叠加
    if first_feasible:
        reward = params["rewards"]["first_feasible_bonus"]

    # (C) 失败类型罚分
    status = diagnostics.get("status", "success") if diagnostics else "success"
    if status != "success":
        reward += params["penalties"].get(status, 0)

    return float(reward)

def _map_to_bounds(action01: np.ndarray, bounds: dict) -> np.ndarray:
    out = []
    keys = ["gamma_fb", "diag_prob"]
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
