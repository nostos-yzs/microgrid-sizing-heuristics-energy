# -*- coding: utf-8 -*-
"""
sa_core.py —— SA 主循环（支持 classic_simple / classic_guided / sac_guided）
- 两层退火：外层降温，内层固定温度采样（当 state["two_layer"]=True）
- 诊断护栏：避免动作贴边导致坍缩
- 每步 reward 写入 state["last_reward"]（供 SAC 使用）
"""

from typing import Dict, Tuple, Optional
import numpy as np
import math

# 维度顺序：PV, WIND, EC, FC, TANK, BESS
PV, WIND, EC, FC, TANK, BESS = 0, 1, 2, 3, 4, 5

# 随机邻域（高斯）
def _gaussian_neighbor(x: np.ndarray,
                       min_b: np.ndarray,
                       max_b: np.ndarray,
                       sigma_ratio: float,
                       rng: np.random.RandomState) -> np.ndarray:
    span = max_b - min_b
    sigma = sigma_ratio * span
    step = rng.normal(loc=0.0, scale=sigma, size=x.shape)
    x_new = np.clip(x + step, min_b, max_b)
    return x_new


def _guided_step_paper_style(
    x: np.ndarray,
    diagnostics: Dict,
    gamma_fb: float,
    min_b: np.ndarray,
    max_b: np.ndarray,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    PSO-style 的诊断引导步（完全对齐 diag_feedback_term 的 dx 逻辑）
    - success:                 dx = -γ_fb * |delta_E| * rand(D) * 0.01 * x
    - fail_supply:             dx =  γ_fb * delta_P * U(0.5,1.5)^D
    - fail_storage_overcharge: dx =  γ_fb * [-1,-1,+1,+1,+1,+1] ⊙ rand(D) * 5
    - fail_storage_overdischarge:
                               dx =  γ_fb * [+1,+1,-1,-1,-1,-1] ⊙ rand(D) * 5
    - 其他/未知:               dx = 0（不调整）
    """
    # 保持与 diag_feedback_term 同步的核心逻辑，只是随机数用传入 rng
    x = x.astype(float, copy=True)
    D = x.size
    gamma_fb = float(gamma_fb)
    status = str(diagnostics.get("status", "success"))

    randD = rng.rand(D)
    uniD = rng.uniform(0.5, 1.5, size=D)

    if status == "success":
        dE = abs(float(diagnostics.get("delta_E", 0.0)))
        dx = - gamma_fb * dE * randD * 0.01 * x

    elif status == "fail_supply":
        dP = float(diagnostics.get("delta_P", 1.0))
        dx = gamma_fb * dP * uniD

    elif status == "fail_storage_overcharge":
        mask = np.array([-1, -1, +1, +1, +1, +1], dtype=float)
        if mask.shape[0] != D:
            m = np.zeros(D, float)
            m[:min(D, mask.shape[0])] = mask[:min(D, mask.shape[0])]
            mask = m
        dx = gamma_fb * mask * randD * 5.0

    elif status == "fail_storage_overdischarge":
        mask = np.array([+1, +1, -1, -1, -1, -1], dtype=float)
        if mask.shape[0] != D:
            m = np.zeros(D, float)
            m[:min(D, mask.shape[0])] = mask[:min(D, mask.shape[0])]
            mask = m
        dx = gamma_fb * mask * randD * 5.0

    else:
        # 未知诊断：不给反馈，保持原值
        dx = np.zeros_like(x)

    x += dx
    np.clip(x, min_b, max_b, out=x)
    return x



def _compute_reward(diag: Dict, info: Dict, rw_cfg: Dict) -> float:
    r = 0.0
    R = (rw_cfg or {}).get("rewards", {}) or {}
    P = (rw_cfg or {}).get("penalties", {}) or {}

    if info.get("gbest_improved", False):
        r += float(R.get("gbest_improvement", 0.0))

    stag_k = int(info.get("stagnation", 0))
    if stag_k > 0:
        r += float(R.get("stagnation", 0.0)) * float(stag_k)

    if info.get("first_feasible", False):
        r += float(R.get("first_feasible_bonus", 0.0))

    st = (diag or {}).get("status", None)
    if st in P:
        r += float(P[st])

    return float(r)


def initialize_sa_state(env=None,
                        min_b: Optional[np.ndarray] = None,
                        max_b: Optional[np.ndarray] = None,
                        schedule: Optional[Dict] = None,
                        seed: int = 42,
                        rw_cfg: Optional[Dict] = None,
                        p_guided_bounds: Optional[Tuple[float, float]] = None) -> Dict:
    rng = np.random.RandomState(seed)
    if min_b is None or max_b is None:
        raise ValueError("initialize_sa_state: min_b / max_b 不能为空")

    min_b = np.asarray(min_b, dtype=float)
    max_b = np.asarray(max_b, dtype=float)

    sched = schedule or {"T0": 1e-1, "alpha": 0.97, "Tmin": 1e-6}
    T0, alpha, Tmin = float(sched["T0"]), float(sched["alpha"]), float(sched["Tmin"])

    x0 = min_b + (max_b - min_b) * rng.rand(min_b.size)
    if env is not None:
        f0, d0 = env.evaluate_solution(x0.copy())
    else:
        f0, d0 = float("inf"), {}


    state = {
        "x": x0, "f": float(f0), "diag": d0,
        "x_best": x0.copy(), "f_best": float(f0), "diag_best": d0,
        "T": T0, "iter": 0,
        "min_b": min_b, "max_b": max_b,
        "schedule": {"T0": T0, "alpha": alpha, "Tmin": Tmin},
        "rng": rng,
        "rw_cfg": rw_cfg or {"rewards": {}, "penalties": {}},
        "stagnation": 0,
        "first_feasible_done": False,
        # "p_guided_bounds": (float(p_guided_bounds[0]), float(p_guided_bounds[1])),
        "last_reward": 0.0,
        # —— 双层开关（外层降温/内层恒温）——
        "two_layer": True,
    }
    return state


def run_sa_iteration(state: Dict,
                     action_vec: np.ndarray,
                     env) -> Dict:
    """
    执行一轮 SA：
    - action_vec: [gamma_fb, diag_prob]
    - 产生候选（guided / gaussian）、Metropolis 验收、更新 gbest、计算 reward
    - reward 写入 state["last_reward"]，供 SAC 训练循环读取
    """
    rng = state["rng"]
    gamma_fb, diag_prob = float(action_vec[0]), float(action_vec[1])

    x = state["x"].astype(float, copy=True)
    f = float(state["f"])
    min_b, max_b = state["min_b"], state["max_b"]

    # —— 退火温度更新（双层：仅读取外层设定；单层：每步递减）——
    sched = state["schedule"]
    if state.get("two_layer", False):
        T = float(state.get("T", sched.get("T0", 1e-1)))
    else:
        it = int(state.get("iter", 0))
        if it == 0:
            T = float(sched["T0"])
        else:
            T = max(float(sched["Tmin"]), float(sched["alpha"]) * float(state["T"]))

    # 评估当前解的诊断
    f_cur, diag_cur = env.evaluate_solution(x.copy())

    # 诊断概率护栏
    p_eff = float(np.clip(diag_prob, 0.0, 1.0))

    # 采样候选
    if rng.rand() < p_eff:
        x_new = _guided_step_paper_style(x, diag_cur, gamma_fb, min_b, max_b, rng=rng)
    else:
        x_new = _gaussian_neighbor(x, min_b, max_b, sigma_ratio=0.02, rng=rng)

    # 评估新解与 Metropolis 验收
    def _finite(v):
        return isinstance(v, (float, int)) and np.isfinite(v)

    f_new, d_new = env.evaluate_solution(x_new)
    state["proposal_f"] = float(f_new)
    state["proposal_diag"] = d_new
    state["proposal_x"] = x_new.copy()
    if not _finite(f_new):
        accept = False
    else:
        f_cur_safe = float(f_cur) if _finite(f_cur) else 1e12
        delta = float(f_new) - f_cur_safe
        accept = (f_new <= f_cur_safe) or (rng.rand() < math.exp(-delta / max(T, 1e-12)))

    if accept:
        x_acc, f_acc, d_acc = x_new, float(f_new), d_new
    else:
        x_acc, f_acc, d_acc = x, float(f_cur), diag_cur

    # 更新全局最优与停滞
    gbest_improved = False
    if _finite(f_acc) and f_acc < state["f_best"]:
        state["x_best"] = x_acc.copy()
        state["f_best"] = float(f_acc)
        state["diag_best"] = d_acc
        state["stagnation"] = 0
        gbest_improved = True
    else:
        state["stagnation"] = int(state.get("stagnation", 0)) + 1

    # 可行性定义：status=="success" 视作首次可行
    first_feasible = False
    if (d_acc or {}).get("status", "") == "success" and not state.get("first_feasible_done", False):
        first_feasible = True
        state["first_feasible_done"] = True

    # 计算 reward
    info = {
        "gbest_improved": gbest_improved,
        "stagnation": state["stagnation"],
        "first_feasible": first_feasible,
        "accepted": bool(accept),
        "p_eff": p_eff,
        "status": (d_acc or {}).get("status", "unknown"),
    }
    rw = _compute_reward(d_acc, info, state.get("rw_cfg", {"rewards": {}, "penalties": {}}))
    state["last_reward"] = float(rw)

    # 写回
    state["x"] = x_acc
    state["f"] = float(f_acc)
    state["diag"] = d_acc
    state["T"] = float(T)  # 双层时：T 由外层赋值；此处只记录
    state["iter"] = int(state.get("iter", 0)) + 1
    return state
