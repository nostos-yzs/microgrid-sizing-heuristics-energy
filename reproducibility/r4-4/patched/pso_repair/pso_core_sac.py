# pso_core_sac.py
import numpy as np

def initialize_swarm(config):
    pop_size = config["pso_core_params"]["population_size"]
    pso_cfg = config["pso_core_params"]
    pop_size = pso_cfg["population_size"]
    min_bounds = np.array(pso_cfg["min_bounds"], dtype=float)
    max_bounds = np.array(pso_cfg["max_bounds"], dtype=float)
    particles = min_bounds + (max_bounds - min_bounds) * np.random.rand(pop_size, 6)
    velocities = (max_bounds - min_bounds) * 0.1 * (np.random.rand(pop_size, 6) - 0.5)
    pbest_positions = particles.copy()
    pbest_fitness = np.array([float('inf')] * pop_size)
    gbest_position, gbest_fitness, gbest_diagnostics = None, float('inf'), None
    return {"particles": particles, "velocities": velocities, "pbest_positions": pbest_positions, "pbest_fitness": pbest_fitness, "gbest_position": gbest_position, "gbest_fitness": gbest_fitness, "gbest_diagnostics": gbest_diagnostics}

def diag_feedback_term(particle, diagnostics, gamma_params):
    """
    基于诊断信息生成一个 PSO 的反馈速度项（与旧版 dx 完全对齐）
    """
    x = particle.astype(float).copy()
    D = x.shape[-1]
    gamma_fb = float(gamma_params["gamma_feedback"])
    status = diagnostics.get("status", "success")

    randD = np.random.rand(D)
    uniD  = np.random.uniform(0.5, 1.5, D)

    if status == "success":
        dE = abs(float(diagnostics.get("delta_E", 0.0)))
        dx = - gamma_fb * dE * randD * 0.01 * x

    elif status == "fail_supply":
        dP = float(diagnostics.get("delta_P", 1.0))
        dx = gamma_fb * dP * uniD

    elif status == "fail_storage_overcharge":
        # 和旧逻辑保持一致的 mask
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
        # 未知诊断：不给反馈（相当于旧代码的“return x 不改”）
        dx = np.zeros_like(x)

    return dx

def run_pso_iteration_guided(swarm_state, op_vector, env):
    """
    引导式 PSO 一次迭代。

    op_vector: [treat_prob, gamma_feedback]
        treat_prob     ∈ [0, 1]，对单个粒子应用诊断反馈的概率；
        gamma_feedback ≥ 0，诊断反馈强度。

    PSO 的惯性和学习因子在函数内部设为固定常数，
    不再由 SAC 控制。
    """
    particles = swarm_state["particles"]
    velocities = swarm_state["velocities"]
    pbest_positions = swarm_state["pbest_positions"]
    pbest_fitness = swarm_state["pbest_fitness"]
    gbest_position = swarm_state["gbest_position"]
    gbest_fitness = swarm_state["gbest_fitness"]

    pop_size = len(particles)

    # 固定的 PSO 参数（你可以以后移到 config 里）
    pso_cfg = env.config["pso_core_params"]
    w = pso_cfg["w"]
    c1 = pso_cfg["c1"]
    c2 = pso_cfg["c2"]

    # SAC 或经典策略给出的两个参数
    treat_prob = float(np.clip(op_vector[0], 0.0, 1.0))
    gamma_fb = float(max(op_vector[1], 0.0))

    gamma_params = {
        "gamma_inertia": w,
        "gamma_pbest": c1,
        "gamma_gbest": c2,
        "gamma_feedback": gamma_fb,
    }

    new_velocities = np.zeros_like(velocities)
    current_gbest_diagnostics = swarm_state["gbest_diagnostics"]

    # 先评估并更新 pbest / gbest
    diagnostics_list = []
    for i in range(pop_size):
        fitness, diagnostics = env.evaluate_solution(particles[i].copy())
        diagnostics_list.append(diagnostics)

        if fitness < pbest_fitness[i]:
            pbest_fitness[i] = fitness
            pbest_positions[i] = particles[i].copy()
        if fitness < gbest_fitness:
            gbest_fitness = fitness
            gbest_position = particles[i].copy()
            current_gbest_diagnostics = diagnostics

    # 再根据 PSO 公式 + 诊断反馈更新速度和位置
    for i in range(pop_size):
        r1, r2 = np.random.rand(2)
        cognitive_term = gamma_params["gamma_pbest"] * r1 * (pbest_positions[i] - particles[i])
        social_term = gamma_params["gamma_gbest"] * r2 * (gbest_position - particles[i]) if gbest_position is not None else 0.0

        # 默认不加反馈
        feedback_term = np.zeros_like(particles[i], dtype=float)

        # 以 treat_prob 的概率应用诊断反馈
        if np.random.rand() < treat_prob:
            diagnostics = diagnostics_list[i]
            feedback_term = diag_feedback_term(particles[i], diagnostics, gamma_params)

        new_velocities[i] = (
            gamma_params["gamma_inertia"] * velocities[i]
            + cognitive_term + social_term
            + feedback_term
        )

    velocities = new_velocities
    particles = particles + new_velocities
    particles = np.maximum(0.0, particles)

    updated_swarm_state = {
        "particles": particles,
        "velocities": velocities,
        "pbest_positions": pbest_positions,
        "pbest_fitness": pbest_fitness,
        "gbest_position": gbest_position,
        "gbest_fitness": gbest_fitness,
        "gbest_diagnostics": current_gbest_diagnostics,
    }
    iteration_results = {
        "gbest_fitness": gbest_fitness,
        "gbest_diagnostics": current_gbest_diagnostics,
        "population_diversity": np.std(particles, axis=0).mean(),
    }
    return updated_swarm_state, iteration_results


def evaluate_moved_swarm(state, env):
    """Evaluate x_(t+1) before computing reward for action a_t. No second motion."""
    representative = None
    for i, x in enumerate(state["particles"]):
        f, d = env.evaluate_solution(x.copy())
        if representative is None or f < representative[0]:
            representative = (float(f), d)
        if f < state["pbest_fitness"][i]:
            state["pbest_fitness"][i] = f
            state["pbest_positions"][i] = x.copy()
        if f < state["gbest_fitness"]:
            state["gbest_fitness"] = float(f)
            state["gbest_position"] = x.copy()
            state["gbest_diagnostics"] = d
    return representative
