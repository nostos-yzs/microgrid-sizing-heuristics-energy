# classic_ga_guided.py
import numpy as np
import random
import ga_core

def run_classic_ga_guided(config, env, total_iterations, run_seed):
    """
    Classic GA (guided, PSO-style):
    - 在“繁殖前”对父代进行一次基于诊断的校正（与 PSO 的 feedback_term 逻辑完全一致）；
    - 之后走经典 GA：精英保留 + 锦标赛选择 + SBX 交叉 + 高斯变异；
    - 不再在变异阶段混入启发式（启发式已前移到父代校正）。
    """
    print(f"\n--- Running Classic Guided GA (Run #{run_seed+1}) ---")
    np.random.seed(run_seed); random.seed(run_seed)

    # === 参数 ===
    params    = config["ga_params"]["classic_params"]["guided"]
    pop_size  = config["ga_params"]["population_size"]
    cx_rate   = params.get("crossover_rate", 0.9)
    mut_rate  = params.get("mutation_rate", 0.15)
    rand_sigma= params.get("random_mutation_strength", 0.1)

    # 与 PSO 一致的反馈强度（等价于 PSO 的 gamma_feedback）
    gamma_fb  = params.get("parent_treat_strength", 0.5)

    # === 初始化种群 ===
    pop, min_b, max_b = ga_core.initialize_population(config)
    gbest_fit, gbest_sol, fit_hist = float('inf'), None, []

    def _pso_style_adjust_parent(ind):
        """
        将 PSO 的 feedback_term 逐式等价为 GA 的位置校正：
            PSO: v += feedback(status, diagnostics)
            GA : x += feedback(status, diagnostics)
        四个分支与 PSO 完全一致（量纲、随机项、掩码/方向）。
        """
        x = ind.copy().astype(float)
        fitness, diagnostics = env.evaluate_solution(x.copy())
        status = diagnostics.get("status", "success")
    
        D = x.shape[0]
        randD = np.random.rand(D)
        uniD  = np.random.uniform(0.5, 1.5, D)
    
        if status == "success":
            # -gamma_fb * |delta_E| * rand(D) * 0.01 * x
            delta_E = abs(float(diagnostics.get("delta_E", 0.0)))
            dx = - gamma_fb * delta_E * randD * 0.01 * x
    
        elif status == "fail_supply":
            #  gamma_fb * delta_P * U(0.5,1.5)^D   （PSO 原版为全维正向）
            delta_P = float(diagnostics.get("delta_P", 1.0))
            dx = gamma_fb * delta_P * uniD
    
        elif status == "fail_storage_overcharge":
            #  gamma_fb * [-1,-1,+1,0,+1,+1] ⊙ rand(D) * 5
            mask = np.array([-1, -1, +1, 0, +1, +1], dtype=float)
            if mask.shape[0] != D:
                m = np.zeros(D, dtype=float)
                m[:min(D, mask.shape[0])] = mask[:min(D, mask.shape[0])]
                mask = m
            dx = gamma_fb * mask * randD * 5.0
    
        elif status == "fail_storage_overdischarge":
            #  gamma_fb * 1^D ⊙ rand(D) * 5
            mask = np.ones(D, dtype=float)
            dx = gamma_fb * mask * randD * 5.0
    
        else:
            return x  # 未知诊断：不改
    
        x += dx
        np.clip(x, min_b, max_b, out=x)  # GA 建议上下界都裁，防越界
        return x
    
    
    # === 迭代 ===
    # 从 config 或 params 里取；若未配置则默认 1.0，保持与旧版完全一致
    parent_treat_prob = float(params.get("parent_treat_prob", 1.0))
    parent_treat_prob = max(0.0, min(1.0, parent_treat_prob))
    
    for _ in range(total_iterations):
        # Step 0: 父代基于诊断的 PSO-style 校正（带概率门控）
        if parent_treat_prob >= 1.0:
            # 与原逻辑一致：对所有父代进行校正
            pop = [_pso_style_adjust_parent(p) for p in pop]
        elif parent_treat_prob <= 0.0:
            # 完全不校正
            pop = [p.copy() for p in pop]
        else:
            # 逐个体按概率校正（未被选中的个体保持不变，避免额外评估）
            new_pop = []
            for p in pop:
                if random.random() < parent_treat_prob:
                    new_pop.append(_pso_style_adjust_parent(p))
                else:
                    new_pop.append(p.copy())
            pop = new_pop
    
        # Step 1: 评估（校正后的父代）
        eval_res = ga_core.evaluate_population(pop, env)
        curr_best = min(eval_res, key=lambda x: x['fitness'])
        if curr_best['fitness'] < gbest_fit:
            gbest_fit, gbest_sol = curr_best['fitness'], curr_best['individual'].copy()
        fit_hist.append(gbest_fit)
    
        # Step 2: 经典 GA 生成后代（精英保留 + 锦标赛 + SBX + 高斯变异）
        next_gen = [gbest_sol]
        while len(next_gen) < pop_size:
            p1 = ga_core.tournament_selection(eval_res)
            p2 = ga_core.tournament_selection(eval_res)
    
            if random.random() < cx_rate:
                c1, c2 = ga_core.simulated_binary_crossover(
                    p1['individual'], p2['individual'], min_b=min_b, max_b=max_b
                )
            else:
                c1, c2 = p1['individual'].copy(), p2['individual'].copy()
    
            if random.random() < mut_rate:
                c1 = ga_core.gaussian_mutation(c1, 1.0, rand_sigma, min_b, max_b)
            if random.random() < mut_rate and len(next_gen) + 1 < pop_size:
                c2 = ga_core.gaussian_mutation(c2, 1.0, rand_sigma, min_b, max_b)
    
            next_gen.append(c1)
            if len(next_gen) < pop_size:
                next_gen.append(c2)
    
        pop = next_gen[:pop_size]
    
    print(f"--- Classic Guided GA (Run #{run_seed+1}) Finished ---")
    return {"best_solution": gbest_sol, "best_fitness": gbest_fit, "fitness_history": fit_hist}
