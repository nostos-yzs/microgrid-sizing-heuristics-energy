# ga_core.py
import numpy as np
import random

# --- 核心遗传算子 ---

def tournament_selection(eval_results, tournament_size=3):
    contenders = random.sample(eval_results, tournament_size)
    winner = min(contenders, key=lambda x: x['fitness'])
    return winner

def simulated_binary_crossover(p1, p2, eta=15, min_b=None, max_b=None):
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if random.random() > 0.5: continue
        u = random.random()
        beta = (2 * u)**(1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u)))**(1 / (eta + 1))
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    if min_b is not None and max_b is not None:
        np.clip(c1, min_b, max_b, out=c1)
        np.clip(c2, min_b, max_b, out=c2)
    return c1, c2

def gaussian_mutation(ind, mut_rate, mut_strength, min_b, max_b):
    mut_ind = ind.copy()
    gene_range = max_b - min_b
    for i in range(len(ind)):
        if random.random() < mut_rate:
            mut_ind[i] += np.random.normal(0, gene_range[i] * mut_strength)
    np.clip(mut_ind, min_b, max_b, out=mut_ind)
    return mut_ind

# def heuristic_mutation(ind, diag, strength, min_b, max_b):
#     # 与 PSO 的 feedback_term 同构：直接用 delta_P / delta_E 等能量量纲驱动
#     import numpy as np, random
#     status = diag.get('status', 'unknown')
#     x = ind.copy().astype(float)

#     # 噪声
#     u05_15 = np.random.uniform(0.5, 1.5, size=x.shape[0])
#     u01 = np.random.rand(x.shape[0])

#     # 方向掩码（与 PSO 的数组一致）
#     if   status == 'fail_supply':
#         # 扩 PV/WIND/TANK/BESS，力度与缺口 delta_P 成正比
#         dp = float(diag.get('delta_P', 1.0))
#         print("dp"+str(dp))
#         mask = np.array([1, 1, 0, 0, 1, 1], dtype=float)
#         delta = dp * u05_15 * mask

#     elif status == 'fail_storage_overdischarge':
#         # 全面补能，常数级强度（与 PSO 中 *5 一致）
#         mask = np.ones_like(x, dtype=float)
#         delta = 5.0 * u01 * mask

#     elif status == 'fail_storage_overcharge':
#         # 降源增存；FC 不动（与 PSO 的 [-1,-1,+1,0,+1,+1] 一致）
#         mask = np.array([-1, -1, +1, 0, +1, +1], dtype=float)
#         delta = 5.0 * u01 * mask

#     # elif status == 'success':
#         # 降本微调：只动 PV/WIND，力度与 |delta_E| 成正比，且按当前规模相对缩放（× ind）
#         # dE = abs(float(diag.get('delta_E', 0.0)))
#         # mask = np.array([1, 1, 0, 0, 0, 0], dtype=float)
#         # delta = - dE * 0.01 * u01 * (x * mask)
#     elif status == 'success':
#         delta_E  = float(diag.get('delta_E', 0.0))
#         print("delta_E"+str(delta_E))
#         # 与 PSO 完全对齐：-γ * |ΔE| * rand * 0.01 * 个体
#         randv = np.random.rand(x.size)  # 或 np.random.rand(x.size)
#         delta = - abs(delta_E) * randv * 0.01 * x
#         x_new = x + delta


#     else:
#         # 不识别的诊断：退回高斯
#         return gaussian_mutation(ind, 1.0, strength, min_b, max_b)

#     # 应用统一强度系数，再裁剪到边界
#     x += strength * delta
#     np.clip(x, min_b, max_b, out=x)
#     return x



# --- 辅助函数与迭代控制器 (最终版) ---

def initialize_population(config):
    pop_size = config["ga_params"]["population_size"]
    min_bounds = config["ga_params"]["min_bounds"]
    max_bounds = config["ga_params"]["max_bounds"]
    num_genes = len(min_bounds)
    population = [min_bounds + np.random.rand(num_genes) * (max_bounds - min_bounds) for _ in range(pop_size)]
    return population, min_bounds, max_bounds

def evaluate_population(population, env):
    return [{"individual": ind, "fitness": lcoe, "diagnostics": diag} 
            for ind, (lcoe, diag) in [(i, env.evaluate_solution(i)) for i in population]]

# --- NEW: 这是与最终版 config 和 heuristic_mutation 协同工作的最终版 run_ga_iteration_guided ---
def run_ga_iteration_guided(pop, ev_res, act_vec, min_b, max_b, env):
    """
    仅用 SAC 的 guide 参数（treat_prob, gamma_feedback）：
      1) 先按诊断对父代做一次 PSO-style 校正（位置直接加反馈项）；
      2) 再进行经典 GA（锦标赛 + SBX + 纯高斯变异）。
    改动：校正后对 treated_pop 进行一次【重评】，并用新 fitness 参与精英/选择。
    """
    import numpy as np, random

    # === 仅解包 guide 参数 ===
    # act_vec = [treat_prob, gamma_feedback]
    treat_prob   = float(act_vec[0])
    gamma_fb     = float(act_vec[1])

    # === 固定的 GA 参数（也可以改成从 config 里读取） ===
    cx_rate      = 0.9
    mut_rate     = 0.4
    rand_mut_s   = 0.1

    pop_size = len(pop)
    D = pop[0].shape[0]

    # ---- PSO-style 父代校正（使用已有 diagnostics，避免校正阶段重复评估） ----
    def _pso_style_adjust_with_diag(x, diag):
        x = x.copy().astype(float)
        status = diag.get("status", "success")
        randD = np.random.rand(D)
        uniD  = np.random.uniform(0.5, 1.5, D)

        if status == "success":
            dE = abs(float(diag.get("delta_E", 0.0)))
            dx = - gamma_fb * dE * randD * 0.01 * x
        # elif status == "fail_supply":
        #     dP = float(diag.get("delta_P", 1.0))
        #     dx = gamma_fb * dP * uniD
        # elif status == "fail_storage_overcharge":
        #     # mask = np.array([-1, -1, +1, 0, +1, +1], dtype=float)
        #     mask = np.array([-1, -1, +1, +1, +1, +1], dtype=float)
        #     if mask.shape[0] != D:
        #         m = np.zeros(D, float); m[:min(D, mask.shape[0])] = mask[:min(D, mask.shape[0])]
        #         mask = m
        #     dx = gamma_fb * mask * randD * 5.0
        # elif status == "fail_storage_overdischarge":
        #     mask = np.array([+1, +1, -1, -1, -1, -1], dtype=float)
        #     if mask.shape[0] != D:
        #         m = np.zeros(D, float); m[:min(D, mask.shape[0])] = mask[:min(D, mask.shape[0])]
        #         mask = m
        #     dx = gamma_fb * mask * randD * 5.0
        else:
            return x  # 未知诊断：不改

        x += dx
        np.clip(x, min_b, max_b, out=x)
        return x

    # 1) 按概率对父代进行一次校正
    treated_pop = []
    for k in range(pop_size):
        if random.random() < max(0.0, min(1.0, treat_prob)):
            treated_pop.append(_pso_style_adjust_with_diag(pop[k], ev_res[k]['diagnostics']))
        else:
            treated_pop.append(pop[k])

    # 2) 【关键修改】对 treated_pop 进行重评，并用新的 fitness 做精英/选择
    ev_res_treated = evaluate_population(treated_pop, env)  # 需要 env 入参

    # 3) 经典 GA：精英 + 锦标赛 + SBX + 纯高斯变异（全部以“重评后的 fitness”为准）
    best_curr = min(ev_res_treated, key=lambda x: x['fitness'])
    next_gen = [best_curr['individual'].copy()]

    while len(next_gen) < pop_size:
        p1_res = tournament_selection(ev_res_treated)
        p2_res = tournament_selection(ev_res_treated)

        if random.random() < cx_rate:
            c1, c2 = simulated_binary_crossover(
                p1_res['individual'], p2_res['individual'], min_b=min_b, max_b=max_b
            )
        else:
            c1, c2 = p1_res['individual'].copy(), p2_res['individual'].copy()

        if random.random() < mut_rate:
            c1 = gaussian_mutation(c1, 1.0, rand_mut_s, min_b, max_b)
        if random.random() < mut_rate and len(next_gen) + 1 < pop_size:
            c2 = gaussian_mutation(c2, 1.0, rand_mut_s, min_b, max_b)

        next_gen.append(c1)
        if len(next_gen) < pop_size:
            next_gen.append(c2)

    return next_gen[:pop_size]

