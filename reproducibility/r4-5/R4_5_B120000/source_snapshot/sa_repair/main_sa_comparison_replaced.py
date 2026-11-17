# -*- coding: utf-8 -*-
"""
main_sa_comparison.py  —— 三种 SA 模式对比（两层退火）

新增（与 main_ga_comparison.py 完全一致的统计 computing time 元素与模式）：
- EvalCounterEnv：统计 env.evaluate_solution 调用次数与耗时（包含 repair/diagnostic 触发的内部评估）
- wall_time_s_full / ems_eval_count_full / ems_eval_time_s_full
- time-to-target（按 fitness_history 近似折算）：tgt_iter / wall_time_s_tgt / ems_eval_count_tgt / ems_eval_time_s_tgt
- 每个统计 run 追加写入 runs/<timestamp>/timing_stats.csv
- 追加写入 best_configs.csv（与 GA 逻辑一致）

依赖：
- config_sa.get_config
- microgrid_env.MicrogridEnv
- sac_agent.SACAgent   （action_dim=2）
- sa_core.initialize_sa_state, sa_core.run_sa_iteration
- utils.plot_statistical_results / plot_final_fitness_boxplot / BestConfigWriter
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch

from config_sa import get_config
from microgrid_env import MicrogridEnv
from sac_agent import SACAgent
from sa_core import initialize_sa_state as initialize_sa, run_sa_iteration

from utils import (
    plot_statistical_results,
    plot_final_fitness_boxplot,
    BestConfigWriter,
)

STATUS_MAP = {
    "success": 0,
    "fail_supply": 1,
    "fail_storage_overdischarge": 2,
    "fail_storage_overcharge": 3,
    "simulation_error": 4,
}


class EvalCounterEnv:
    """Wrap env to count evaluate_solution calls + time (includes diagnostic/repair internal calls)."""

    def __init__(self, env):
        self.env = env
        self.reset_counters()

    def reset_counters(self):
        self.ems_eval_count = 0
        self.ems_eval_time_s = 0.0

    def evaluate_solution(self, x):
        t0 = time.perf_counter()
        out = self.env.evaluate_solution(x)
        self.ems_eval_time_s += (time.perf_counter() - t0)
        self.ems_eval_count += 1
        return out

    def __getattr__(self, name):
        return getattr(self.env, name)


def find_time_to_target_from_history(fitness_history, eps=0.01):
    """
    time-to-target: first iteration where fitness <= (1+eps)*final_best.
    Returns: (tgt_iter, final_best, threshold)
    """
    if not fitness_history:
        return None, None, None
    final_best = float(fitness_history[-1])
    threshold = final_best * (1.0 + float(eps))
    tgt_iter = None
    for k, f in enumerate(fitness_history):
        if float(f) <= threshold:
            tgt_iter = k
            break
    if tgt_iter is None:
        tgt_iter = len(fitness_history) - 1
    return int(tgt_iter), float(final_best), float(threshold)


def attach_full_and_target_metrics(out, wall_full_s, ems_eval_count_full, ems_eval_time_full_s, eps=0.01):
    """Attach full-run metrics + time-to-target metrics (proportional approximation)."""
    out["wall_time_s_full"] = float(wall_full_s)
    out["ems_eval_count_full"] = int(ems_eval_count_full)
    out["ems_eval_time_s_full"] = float(ems_eval_time_full_s)

    hist = out.get("fitness_history", []) or []
    tgt_iter, best_final, threshold = find_time_to_target_from_history(hist, eps=eps)

    out["tgt_iter"] = tgt_iter
    out["best_final"] = best_final
    out["tgt_threshold"] = threshold

    if tgt_iter is None or best_final is None or len(hist) <= 1:
        out["wall_time_s_tgt"] = float(wall_full_s)
        out["ems_eval_count_tgt"] = int(ems_eval_count_full)
        out["ems_eval_time_s_tgt"] = float(ems_eval_time_full_s)
        return out

    frac = float(tgt_iter) / float(max(1, len(hist) - 1))
    out["wall_time_s_tgt"] = float(wall_full_s * frac)
    out["ems_eval_count_tgt"] = int(round(ems_eval_count_full * frac))
    out["ems_eval_time_s_tgt"] = float(ems_eval_time_full_s * frac)
    return out


# ---------- 典型日数据 ----------
def load_typical_days_data(data_folder="data"):
    typical_days_data = []
    required_columns = ["GHI", "Tair", "Vwind", "Load"]
    for i in range(5):
        fp = os.path.join(data_folder, f"typicalday{i}.csv")
        if not os.path.exists(fp):
            print(f"[ERROR] not found: {fp}")
            return None
        df = pd.read_csv(fp)
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            print(f"[ERROR] missing columns in {fp}: {missing}")
            return None
        typical_days_data.append(df)
    print(f"[INFO] loaded {len(typical_days_data)} typical days from '{data_folder}'")
    return typical_days_data


# ---------- RL 状态 / 奖励 ----------
def get_state_sa(sa_state: dict, total_iterations: int) -> np.ndarray:
    iter_ratio = float(sa_state.get("iter", 0)) / max(1, total_iterations)

    diversity = 0.0
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


# ---------- 工具：读取两层参数 ----------
def _read_layers(cfg):
    sim = cfg.get("simulation_params", {})
    n_layers = int(sim.get("n_layers", 30))
    inner_loops = int(sim.get("inner_loops", 1))
    return n_layers, inner_loops


# ---------- 三种模式 ----------
def run_sa_traditional(config, env, total_iterations, run_seed):
    print(f"\n--- SA (traditional) (seed={run_seed}) ---")
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    random.seed(run_seed)

    min_b = config["min_bounds"].astype(float)
    max_b = config["max_bounds"].astype(float)
    sched = config["sa_params"]["schedule"]
    seed = int(config.get("experiments", {}).get("classic_simple", {}).get("random_seed", run_seed))

    sa = initialize_sa(
        env=env,
        min_b=min_b,
        max_b=max_b,
        schedule=sched,
        seed=seed,
        rw_cfg={"rewards": {}, "penalties": {}},
    )

    n_layers, inner_loops = _read_layers(config)
    T0, alpha, Tmin = float(sched["T0"]), float(sched["alpha"]), float(sched["Tmin"])

    fixed_action = np.array([0.0, 0.0], dtype=float)
    fit_hist = []
    for layer in range(n_layers):
        sa["T"] = max(T0 * (alpha ** layer), Tmin)
        for _ in range(inner_loops):
            sa = run_sa_iteration(sa, fixed_action, env)
            fit_hist.append(float(sa["f_best"]))

    return {
        "best_solution": sa["x_best"],
        "best_fitness": float(sa["f_best"]),
        "diagnostics": sa.get("diag_best", None),
        "fitness_history": fit_hist,
    }


def run_sa_guided(config, env, total_iterations, run_seed):
    print(f"\n--- SA (guided-fixed) (seed={run_seed}) ---")
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    random.seed(run_seed)

    min_b = config["min_bounds"].astype(float)
    max_b = config["max_bounds"].astype(float)
    sched = config["sa_params"]["schedule"]

    exp = config["experiments"]["classic_guided"]
    gamma_fb = float(exp["gamma_fb"])
    diag_prob = float(exp["diag_prob"])
    seed = int(exp.get("random_seed", run_seed))

    sa = initialize_sa(
        env=env,
        min_b=min_b,
        max_b=max_b,
        schedule=sched,
        seed=seed,
        rw_cfg={"rewards": {}, "penalties": {}},
    )

    n_layers, inner_loops = _read_layers(config)
    T0, alpha, Tmin = float(sched["T0"]), float(sched["alpha"]), float(sched["Tmin"])

    fixed_action = np.array([gamma_fb, diag_prob], dtype=float)
    fit_hist = []
    for layer in range(n_layers):
        sa["T"] = max(T0 * (alpha ** layer), Tmin)
        for _ in range(inner_loops):
            sa = run_sa_iteration(sa, fixed_action, env)
            fit_hist.append(float(sa["f_best"]))

    return {
        "best_solution": sa["x_best"],
        "best_fitness": float(sa["f_best"]),
        "diagnostics": sa.get("diag_best", None),
        "fitness_history": fit_hist,
    }


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


def run_sa_rl_guided(config, env, total_iterations, run_seed):
    print(f"\n--- SA (guided + SAC) (seed={run_seed}) ---")
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    random.seed(run_seed)

    min_b = config["min_bounds"].astype(float)
    max_b = config["max_bounds"].astype(float)
    sched = config["sa_params"]["schedule"]

    sac_cfg = config["experiments"]["sac_guided"]["sac_agent_params"]
    bounds = sac_cfg["action_bounds"]
    seed = int(config["experiments"]["sac_guided"].get("random_seed", run_seed))

    cfg_for_agent = dict(config)
    cfg_for_agent["sac_agent_params"] = sac_cfg
    agent = SACAgent(cfg_for_agent)  # action_dim=2

    sa = initialize_sa(
        env=env,
        min_b=min_b,
        max_b=max_b,
        schedule=sched,
        seed=seed,
        rw_cfg=sac_cfg,
    )

    n_layers, inner_loops = _read_layers(config)
    total_iterations = n_layers * inner_loops
    T0, alpha, Tmin = float(sched["T0"]), float(sched["alpha"]), float(sched["Tmin"])

    fit_hist = []
    t = 0
    state = get_state_sa(sa, total_iterations)

    # 记录“历史上是否出现过可行解”
    ever_feasible = ((sa.get("diag_best") or {}).get("status", "") == "success")

    log_rows = []
    for layer in range(n_layers):
        sa["T"] = max(T0 * (alpha ** layer), Tmin)
        for _ in range(inner_loops):
            raw_action = agent.choose_action(state)
            if len(raw_action) != 2:
                raise RuntimeError(f"SACAgent.choose_action() must return 2 dims, got {len(raw_action)}")
            action = _map_to_bounds(np.asarray(raw_action, dtype=float), bounds)

            gamma_fb = float(action[0])
            diag_prob = float(action[1])

            old_best = float(sa["f_best"])
            sa = run_sa_iteration(sa, action, env)

            status_now = (sa.get("diag_best") or {}).get("status", "")
            if (not ever_feasible) and (status_now == "success"):
                just_became_feasible = True
                ever_feasible = True
            else:
                just_became_feasible = False

            reward = calculate_reward_sa(
                old_best,
                float(sa["f_best"]),
                sa.get("diag_best", {}),
                just_became_feasible,
                sac_cfg,
            )

            next_state = get_state_sa(sa, total_iterations)
            done = (t == total_iterations - 1)

            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            fit_hist.append(float(sa["f_best"]))

            status = (sa.get("diag") or {}).get("status", "unknown")

            log_rows.append(
                {
                    "iter": int(sa.get("iter", t)),
                    "best_f": float(sa["f_best"]),
                    "f": float(sa["f"]),
                    "gamma_fb": gamma_fb,
                    "diag_prob": diag_prob,
                    "T": float(sa["T"]),
                    "reward": reward,
                    "status": status,
                }
            )
            t += 1

    # 保留原行为：写 iter log
    log_root = os.path.join("runs_sa", "sac_guided")
    os.makedirs(log_root, exist_ok=True)
    csv_path = os.path.join(log_root, "iter_log.csv")
    pd.DataFrame(log_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[LOG] SA iter log saved to: {csv_path}")

    return {
        "best_solution": sa["x_best"],
        "best_fitness": float(sa["f_best"]),
        "diagnostics": sa.get("diag_best", None),
        "fitness_history": fit_hist,
    }


# ---------- 统计主程序 ----------
def run_sa_experiments():
    config = get_config()

    # 典型日
    typical_days = load_typical_days_data("data")
    if typical_days is None:
        print("[FATAL] typical day CSVs missing. Abort.")
        return

    # 两层 → 总步数 = n_layers * inner_loops
    n_layers, inner_loops = _read_layers(config)
    total_iterations = n_layers * inner_loops

    # 与 GA 完全一致的统计参数读取方式（兼容缺省）
    sim = config.get("simulation_params", {})
    num_runs = int(sim.get("num_independent_runs", 1))
    eps = float(sim.get("time_to_target_eps", 0.01))
    base_seed = int(sim.get("base_seed", 42))

    run_dir = os.path.join("runs", time.strftime("sa_%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print("Logs to:", os.path.abspath(run_dir))

    best_writer = BestConfigWriter(outdir=run_dir)
    timing_csv = os.path.join(run_dir, "timing_stats.csv")

    results = {"SA with proposed RL-adjusted operator": [], "SA with proposed static operator": [], "SA": []}

    start_wall = time.time()

    for i in range(num_runs):
        print(f"\n{'='*25} Starting Statistical Run #{i+1}/{num_runs} {'='*25}")
        seed = base_seed + i

        # Fresh env per run（与 GA 一致）
        base_env = MicrogridEnv(config, typical_days)
        env = EvalCounterEnv(base_env)

        timing_rows = []

        def _run_one(method_name, fn):
            env.reset_counters()
            t0 = time.perf_counter()
            res = fn(config, env, total_iterations, seed)
            wall_full = time.perf_counter() - t0

            res = attach_full_and_target_metrics(
                res,
                wall_full_s=wall_full,
                ems_eval_count_full=env.ems_eval_count,
                ems_eval_time_full_s=env.ems_eval_time_s,
                eps=eps,
            )
            results[method_name].append(res)

            # best config logging（与 GA 一致）
            sol, fit, diag = res.get("best_solution"), res.get("best_fitness"), res.get("diagnostics")
            if diag is None and sol is not None:
                try:
                    _, diag = env._run_single_simulation(sol, stochastic=False)
                except Exception:
                    diag = {"status": "unknown"}
            best_writer.append(
                method_name,
                i,
                (diag or {}).get("status", "unknown"),
                fit if fit is not None else float("inf"),
                sol,
            )

            row = {
                "run": i,
                "seed": seed,
                "method": method_name,
                "wall_time_s_full": res.get("wall_time_s_full"),
                "ems_eval_count_full": res.get("ems_eval_count_full"),
                "ems_eval_time_s_full": res.get("ems_eval_time_s_full"),
                "tgt_iter": res.get("tgt_iter"),
                "best_final": res.get("best_final"),
                "tgt_threshold": res.get("tgt_threshold"),
                "wall_time_s_tgt": res.get("wall_time_s_tgt"),
                "ems_eval_count_tgt": res.get("ems_eval_count_tgt"),
                "ems_eval_time_s_tgt": res.get("ems_eval_time_s_tgt"),
            }
            return row

        # 3) Guided SA + SAC
        timing_rows.append(_run_one("SA with proposed RL-adjusted operator", run_sa_rl_guided))

        # 2) Guided SA
        timing_rows.append(_run_one("SA with proposed static operator", run_sa_guided))
        
        # 1) SA baseline
        timing_rows.append(_run_one("SA", run_sa_traditional))

        # write timing rows (append) —— 与 GA 一致
        df_timing = pd.DataFrame(timing_rows)
        if not os.path.exists(timing_csv):
            df_timing.to_csv(timing_csv, index=False)
        else:
            df_timing.to_csv(timing_csv, mode="a", header=False, index=False)

    print(f"\n\n{'='*20} STATISTICAL ANALYSIS COMPLETE {'='*20}")
    print(f"Total time for {num_runs} runs: {time.time() - start_wall:.2f} seconds")
    print(f"Best-configs saved → {os.path.abspath(os.path.join(run_dir, 'best_configs.csv'))}")
    print(f"Timing stats saved → {os.path.abspath(timing_csv)}")

    # 保持原绘图行为：曲线 + boxplot
    histories = {name: [r.get("fitness_history", []) for r in lst] for name, lst in results.items()}
    finals = {name: [h[-1] for h in hs if len(h) > 0] for name, hs in histories.items()}

    plot_statistical_results(
        histories_list=[histories[k] for k in results.keys()],
        labels=list(results.keys()),
    )
    plot_final_fitness_boxplot(finals)

    # 打印每种 SA 在所有 runs 中的最优解及装机配置（保留原逻辑）
    for method_name, res_list in results.items():
        if not res_list:
            continue
        best_run = min(res_list, key=lambda r: r.get("best_fitness", float("inf")))
        x_best = np.array(best_run.get("best_solution", []), dtype=float)
        f_best = float(best_run.get("best_fitness", float("inf")))

        if x_best.size >= 6:
            n_PV, n_WIND, n_EC, n_FC, n_TANK, n_BESS = (int(np.ceil(x)) if x > 0 else 0 for x in x_best[:6])
            print(f"\n[{method_name}] 全部 {num_runs} 次实验中的最优结果：")
            print(f"  最优目标值（LCOE） = {f_best:.6f}")
            print(f"  连续变量 x_best   = {x_best}")
            print("  装机配置（取整后）：")
            print(f"    PV   组件数量 = {n_PV}")
            print(f"    WIND 机组数量 = {n_WIND}")
            print(f"    EC   台数     = {n_EC}")
            print(f"    FC   台数     = {n_FC}")
            print(f"    H2 TANK 数量  = {n_TANK}")
            print(f"    BESS 簇数量   = {n_BESS}")
        else:
            print(f"\n[{method_name}] best_fitness={f_best:.6f}, x_best={x_best}")


if __name__ == "__main__":
    run_sa_experiments()
