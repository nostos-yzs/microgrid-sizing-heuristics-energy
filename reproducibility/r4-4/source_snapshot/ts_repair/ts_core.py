# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:40:56 2025

@author: Lenovo
"""

# ts/ts_core.py
import numpy as np, random, hashlib

# 变量下标（与你环境中的6维保持一致）
PV, WIND, EC, FC, TANK, BESS = 0,1,2,3,4,5

# 诊断引导的概率裁剪

def _clip(x, lo, hi): 
    return np.clip(x, lo, hi)

def _hash_region(x, min_b, max_b, grid_ratio=0.01):
    span = (max_b - min_b)
    grid = np.floor((x - min_b) / (grid_ratio * span + 1e-12)).astype(int)
    return hashlib.md5(grid.tobytes()).hexdigest()

def _hash_move(dx, min_b, max_b, tau_ratio=0.005):
    span = (max_b - min_b)
    sig = (np.abs(dx) > (tau_ratio * span)).astype(int)
    sgn = np.sign(dx).astype(int)
    feat = np.stack([sig, sgn], axis=1).astype(np.int8).tobytes()
    return hashlib.md5(feat).hexdigest()


def _diag_feedback_ts(x, diagnostics, gamma_fb, rng):
    x = x.astype(float).copy()
    D = x.shape[-1]
    status = diagnostics.get("status", "success")

    # --- RNG 兼容：RandomState / Generator 都能用 ---
    if hasattr(rng, "rand"):  # 旧 numpy.random.RandomState
        randD = rng.rand(D)
        uniD  = rng.uniform(0.5, 1.5, D)
    else:  # 新 numpy.random.Generator
        randD = rng.random(D)
        uniD  = rng.uniform(0.5, 1.5, size=D)

    dE = abs(float(diagnostics.get("delta_E", 0.0)))
    dP = float(diagnostics.get("delta_P", 1.0))

    if status == "success":
        dx = - gamma_fb * dE * randD * 0.01 * x

    elif status == "fail_supply":
        dx = gamma_fb * dP * uniD

    elif status == "fail_storage_overcharge":
        mask = np.array([-1, -1, +1, +1, +1, +1], float)
        if mask.shape[0] != D:
            m = np.zeros(D, float)
            m[:min(D, mask.shape[0])] = mask[:min(D, mask.shape[0])]
            mask = m
        dx = gamma_fb * mask * randD * 5.0

    elif status == "fail_storage_overdischarge":
        mask = np.array([+1, +1, -1, -1, -1, -1], float)
        if mask.shape[0] != D:
            m = np.zeros(D, float)
            m[:min(D, mask.shape[0])] = mask[:min(D, mask.shape[0])]
            mask = m
        dx = gamma_fb * mask * randD * 5.0

    else:
        dx = np.zeros_like(x)

    return dx

def _guided_step_paper_style(x, diag, gamma_fb, min_b, max_b, rng=None):
    """
    与 GA/PSO 同构的“诊断步”模板（只改当前解 x），
    这里直接复用 PSO 的 diag_feedback_term 逻辑。
    """
    rng = rng or np.random
    x = x.copy().astype(float)

    # 1) 用和 PSO 一致的规则算出 dx
    dx = _diag_feedback_ts(x, diag or {}, gamma_fb, rng)

    # 2) 在 TS 里直接把 dx 当作一步的“引导步长”
    u = x + dx

    # 3) 仍然在边界内裁剪
    return _clip(u, min_b, max_b)
    """
    与 GA/PSO 同构的“诊断步”模板（只改当前解 x），保持论文一致性。
    """
    rng = rng or np.random
    x = x.copy().astype(float)
    status = (diag or {}).get("status", "success")
    dE = abs(float((diag or {}).get("delta_E", 0.0)))
    dP = float((diag or {}).get("delta_P", 0.0))
    D  = x.shape[0]
    randD = rng.uniform(0.9, 1.1, D)
    span = (max_b - min_b)

    if status == "success":
        # 与 PSO/GA 完全对齐：- γ_fb * |ΔE| * rand * scale * x
        # 取 ΔE 与 γ_feedback
        delta_E = dE
        # 随机因子 & 缩放系数（PSO/GA 用 0.01；若想保持你原来的 0.003~0.012，可把 scale 换成 rng.uniform(0.003, 0.012)）
        randv = rng.rand(x.size) if hasattr(rng, "rand") else np.random.rand(x.size)
        scale = 1
    
        delta = - gamma_fb * abs(delta_E) * randv * scale * x
        u = x + delta


    elif status == "fail_supply":
        k = rng.uniform(0.5, 1.5) * max(0.0, dP) * gamma_fb
        v = np.ones_like(x, float)
        step = 0.01 * k * v * randD * span
        u = x + step

    elif status == "fail_storage_overcharge":
        v = np.array([-1, -1, +1, 0, +1, +1], float)
        if v.shape[0] != D:
            w = np.zeros(D, float); w[:min(D,6)] = v[:min(D,6)]; v = w
        k = rng.uniform(0.005, 0.010) * gamma_fb
        step = k * v * randD * span
        u = x + step

    elif status == "fail_storage_overdischarge":
        v = np.ones_like(x, float)
        k = rng.uniform(0.02, 0.05) * gamma_fb
        step = k * v * randD * span
        u = x + step

    else:
        # 未知状态：不引导
        return x

    return _clip(u, min_b, max_b)

def _random_neighbor(x, min_b, max_b, sigma_ratio, rng):
    span = (max_b - min_b)
    step = rng.normal(0.0, sigma_ratio, size=x.shape) * span
    return _clip(x + step, min_b, max_b), step

def initialize_tabu_state(config, env, seed=None):
    if seed is None:
        seed = config.get("random_seed", 42)

    rng = np.random.default_rng(seed)
    min_b = np.array(config["ts_params"]["min_bounds"], float)
    max_b = np.array(config["ts_params"]["max_bounds"], float)
    D = min_b.size
    x0 = min_b + (max_b - min_b) * rng.random(D)
    f0, d0 = env.evaluate_solution(x0)
    state = {
        "x": x0, "f": float(f0), "diag": d0,
        "best_x": x0.copy(), "best_f": float(f0), "best_diag": d0,
        "min_b": min_b, "max_b": max_b,
        "rng": rng, "iter": 0,
        "tabu_moves": [], "tabu_regions": [],
        "tabu_len": int(config["ts_params"].get("tabu_len", 32)),
        "sigma_ratio": float(config["ts_params"].get("sigma_random", 0.15)),
        "grid_ratio": float(config["ts_params"].get("grid_ratio", 0.01)),
        "tau_ratio": float(config["ts_params"].get("tau_ratio", 0.005)),
        "K": int(config["ts_params"].get("neighbors", 16)),
    }
    return state

def _is_tabu(u, dx, state):
    # 区域禁忌
    h_region = _hash_region(u, state["min_b"], state["max_b"], state["grid_ratio"])
    if h_region in state["tabu_regions"]:
        return True
    # 移动禁忌
    h_move = _hash_move(dx, state["min_b"], state["max_b"], state["tau_ratio"])
    if h_move in state["tabu_moves"]:
        return True
    return False

def _update_tabu(u, dx, state):
    h_region = _hash_region(u, state["min_b"], state["max_b"], state["grid_ratio"])
    h_move   = _hash_move(dx, state["min_b"], state["max_b"], state["tau_ratio"])
    state["tabu_regions"].append(h_region)
    state["tabu_moves"].append(h_move)
    # 控制长度
    L = state["tabu_len"]
    if len(state["tabu_regions"]) > L: state["tabu_regions"].pop(0)
    if len(state["tabu_moves"])   > L: state["tabu_moves"].pop(0)

def run_tabu_iteration_simple(state, env):
    """
    纯 TS：全随机邻域，选最优非禁忌（破禁=刷新历史最优）。
    """
    rng = state["rng"]
    x, f = state["x"], state["f"]
    min_b, max_b = state["min_b"], state["max_b"]
    K, sigma = state["K"], state["sigma_ratio"]

    cands, metas = [], []
    for _ in range(K):
        u, dx = _random_neighbor(x, min_b, max_b, sigma, rng)
        cands.append(u); metas.append(dx)

    # 选择
    best_u, best_fu, best_du, best_dx = None, None, None, None
    for u, dx in zip(cands, metas):
        fu, du = env.evaluate_solution(u)
        tabu = _is_tabu(u, dx, state)
        # 破禁：只要刷新历史最优
        if tabu and fu >= state["best_f"]:
            continue
        if (best_fu is None) or (fu < best_fu):
            best_u, best_fu, best_du, best_dx = u, float(fu), du, dx

    # fallback：若全被禁且不破禁，就选最不差的（仍要更新一次）
    if best_u is None:
        scores = []
        for u, dx in zip(cands, metas):
            fu, du = env.evaluate_solution(u)
            scores.append((fu, u, du, dx))
        fu, u, du, dx = min(scores, key=lambda z: z[0])
        best_u, best_fu, best_du, best_dx = u, float(fu), du, dx

    # 更新
    state["x"], state["f"], state["diag"] = best_u, best_fu, best_du
    _update_tabu(best_u, best_dx, state)

    if best_fu < state["best_f"]:
        state["best_f"] = best_fu
        state["best_x"] = best_u.copy()
        state["best_diag"] = best_du

    state["iter"] += 1
    return state

def run_tabu_iteration_guided(state, action_vec, env):
    """
    TS + 诊断邻域组合：
      action_vec = [diag_prob, gamma_feedback]
    在 K 个邻域中，按 diag_prob 比例使用“诊断步邻域”，其余用随机邻域；
    然后执行标准 TS 的“最优可接受”选择与禁忌更新。
    """
    diag_prob   = float(action_vec[0])
    gamma_fb    = float(action_vec[1])
    p_eff       = diag_prob

    rng = state["rng"]
    x, f, diag = state["x"], state["f"], state["diag"]
    min_b, max_b = state["min_b"], state["max_b"]
    K, sigma = state["K"], state["sigma_ratio"]

    cands, metas = [], []
    guided_quota = int(np.floor(p_eff * K))

    # 诊断邻域
    for _ in range(guided_quota):
        u = _guided_step_paper_style(x, diag, gamma_fb, min_b, max_b, rng=rng)
        dx = u - x
        cands.append(u); metas.append(dx)

    # 随机邻域
    for _ in range(K - guided_quota):
        u, dx = _random_neighbor(x, min_b, max_b, sigma, rng)
        cands.append(u); metas.append(dx)

    # 选择最优可接受（带破禁）
    best_u, best_fu, best_du, best_dx = None, None, None, None
    for u, dx in zip(cands, metas):
        fu, du = env.evaluate_solution(u)
        tabu = _is_tabu(u, dx, state)
        if tabu and fu >= state["best_f"]:
            continue
        if (best_fu is None) or (fu < best_fu):
            best_u, best_fu, best_du, best_dx = u, float(fu), du, dx

    if best_u is None:
        scores = []
        for u, dx in zip(cands, metas):
            fu, du = env.evaluate_solution(u)
            scores.append((fu, u, du, dx))
        fu, u, du, dx = min(scores, key=lambda z: z[0])
        best_u, best_fu, best_du, best_dx = u, float(fu), du, dx

    # 更新
    state["x"], state["f"], state["diag"] = best_u, best_fu, best_du
    _update_tabu(best_u, best_dx, state)

    if best_fu < state["best_f"]:
        state["best_f"] = best_fu
        state["best_x"] = best_u.copy()
        state["best_diag"] = best_du

    state["iter"] += 1
    return state
