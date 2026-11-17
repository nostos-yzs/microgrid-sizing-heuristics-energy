# -*- coding: utf-8 -*-
# main_ga_comparison.py
#
# Patched: add identical computing-time statistics pattern as the reference file:
# - wall-clock time (full run + time-to-target proxy)
# - EMS evaluate_solution call count + cumulative time (via env wrapper)
# - per-run timing_stats.csv appended under timestamped run_dir
#
# NOTE: Algorithmic logic is kept unchanged; only instrumentation + run-loop scaffolding is aligned.

import os
import time
import random
import collections
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config_ga import get_config
from microgrid_env import MicrogridEnv
from sac_agent import SACAgent
import ga_core
from classic_ga_simple import run_classic_ga_simple
from classic_ga_guided import run_classic_ga_guided

from utils import (
    plot_statistical_results,
    plot_final_fitness_boxplot,
    ExperimentLogger,
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
    """Wrap env to count evaluate_solution calls + time."""

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


def find_time_to_target_from_history(fitness_history, eps: float = 0.01) -> Tuple[Optional[int], Optional[float], Optional[float]]:
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


def attach_full_and_target_metrics(
    out: Dict[str, Any],
    wall_full_s: float,
    ems_eval_count_full: int,
    ems_eval_time_full_s: float,
    eps: float = 0.01,
) -> Dict[str, Any]:
    """Attach full-run metrics + time-to-target metrics (proportional approximation)."""
    out["wall_time_s_full"] = float(wall_full_s)
    out["ems_eval_count_full"] = int(ems_eval_count_full)
    out["ems_eval_time_s_full"] = float(ems_eval_time_full_s)

    hist = out.get("fitness_history", []) or []
    tgt_iter, best_final, threshold = find_time_to_target_from_history(hist, eps=eps)

    out["tgt_iter"] = tgt_iter
    out["best_final"] = best_final
    out["tgt_threshold"] = threshold

    # fallback: if history is missing / too short, tgt == full
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


def load_typical_days_data(data_folder: str = "data"):
    typical_days_data = []
    required_columns = ["GHI", "Tair", "Vwind", "Load"]
    try:
        for i in range(5):
            file_path = os.path.join(data_folder, f"typicalday{i}.csv")
            df = pd.read_csv(file_path)
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"FATAL: In '{file_path}', missing columns: {missing_cols}")
                return None
            typical_days_data.append(df)
        print(f"Loaded {len(typical_days_data)} typical days from '{data_folder}'.")
        return typical_days_data
    except FileNotFoundError:
        print(f"Error: Data folder '{data_folder}' or its files not found.")
        return None


def get_state_ga(population, iteration, total_iterations, gbest_history, gbest_diagnostics):
    iter_ratio = iteration / total_iterations
    diversity = np.std(np.array(population), axis=0).mean()
    normalized_diversity = min(diversity / 100.0, 1.0)
    convergence_rate = 0
    if len(gbest_history) > 10 and gbest_history[0] < 1e6:
        improvement = gbest_history[0] - gbest_history[-1]
        convergence_rate = improvement / gbest_history[0] if gbest_history[0] > 0 else 0
    status_str = gbest_diagnostics.get("status", "success") if gbest_diagnostics else "success"
    status_code = STATUS_MAP.get(status_str, 0) / len(STATUS_MAP)
    return np.array([iter_ratio, normalized_diversity, convergence_rate, status_code])


def calculate_reward(old_gbest, new_gbest, diagnostics, first_feasible, config):
    params = config["sac_agent_params"]
    reward = 0.0

    if new_gbest < old_gbest:
        improvement = (old_gbest - new_gbest) / old_gbest if 0 < old_gbest < 1e6 else 0.0
        reward += params["rewards"]["gbest_improvement"] * np.tanh(improvement * 10.0)
    else:
        reward += params["rewards"]["stagnation"]

    if first_feasible:
        reward += params["rewards"]["first_feasible_bonus"]

    status = diagnostics.get("status", "success") if diagnostics else "success"
    if status != "success":
        reward += params["penalties"].get(status, 0)

    return float(reward)


def run_sac_guided_ga(config, env, total_iterations, run_seed):
    print(f"\n--- Running SAC-Guided-GA (Run #{run_seed+1}) ---")
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    random.seed(run_seed)

    agent = SACAgent(config)

    pop, min_b, max_b = ga_core.initialize_population(config)
    gbest_fit, gbest_sol, gbest_diag = float("inf"), None, None
    fit_hist, gbest_hist = [], collections.deque(maxlen=20)

    logger = ExperimentLogger(outdir=f"runs/ga_sac_seed{run_seed}")
    best_tag_counter = 0

    state = get_state_ga(pop, 0, total_iterations, list(gbest_hist), gbest_diag)
    first_bonus_paid = False

    for i in range(total_iterations):
        action_vector = agent.choose_action(state)

        eval_res_old = ga_core.evaluate_population(pop, env)
        curr_best_old = min(eval_res_old, key=lambda x: x["fitness"])

        old_gbest_fit = gbest_fit

        if (curr_best_old["fitness"] < gbest_fit) and (curr_best_old["fitness"] < 1e6) and (
            curr_best_old["diagnostics"].get("status") == "success"
        ):
            gbest_fit = curr_best_old["fitness"]
            gbest_sol = curr_best_old["individual"].copy()
            gbest_diag = curr_best_old["diagnostics"]

        pop = ga_core.run_ga_iteration_guided(pop, eval_res_old, action_vector, min_b, max_b, env)

        eval_res_new = ga_core.evaluate_population(pop, env)
        curr_best_new = min(eval_res_new, key=lambda x: x["fitness"])

        if (curr_best_new["fitness"] < gbest_fit) and (curr_best_new["fitness"] < 1e6) and (
            curr_best_new["diagnostics"].get("status") == "success"
        ):
            gbest_fit = curr_best_new["fitness"]
            gbest_sol = curr_best_new["individual"].copy()
            gbest_diag = curr_best_new["diagnostics"]

        is_feasible_now = (curr_best_new["fitness"] < 1e6) and (curr_best_new["diagnostics"].get("status") == "success")
        is_first_feasible = (not first_bonus_paid) and is_feasible_now
        if is_first_feasible:
            first_bonus_paid = True

        reward = calculate_reward(old_gbest_fit, gbest_fit, curr_best_new["diagnostics"], is_first_feasible, config)

        next_state = get_state_ga(pop, i + 1, total_iterations, list(gbest_hist), gbest_diag)
        done_flag = i == total_iterations - 1
        agent.replay_buffer.add(state, action_vector, reward, next_state, done_flag)
        agent.learn()

        state = next_state
        gbest_hist.append(gbest_fit)
        fit_hist.append(gbest_fit)

        # logging & trace saving (unchanged)
        try:
            conv_rate = 0.0
            if len(gbest_hist) > 1 and gbest_hist[0] < 1e6:
                base0 = gbest_hist[0]
                imp = base0 - gbest_hist[-1]
                conv_rate = (imp / base0) if base0 > 0 else 0.0
            logger.log_iter(
                iteration=i,
                total_iterations=total_iterations,
                action_vec=action_vector,
                gbest=gbest_fit,
                curr_best=curr_best_new["fitness"],
                reward=reward,
                ev_res=eval_res_new,
                gbest_diag=gbest_diag,
                conv_rate=conv_rate,
            )
        except Exception:
            pass

        if (gbest_fit < old_gbest_fit) and (gbest_fit < 1e6) and (gbest_sol is not None) and (
            gbest_diag and gbest_diag.get("status") == "success"
        ):
            try:
                df_trace = env.simulate_day_trace(gbest_sol, day_index=0, stochastic=False)
                logger.save_trace_df(df_trace, tag=f"{i:04d}_{best_tag_counter}")
                best_tag_counter += 1
            except Exception:
                pass

    return {"best_solution": gbest_sol, "best_fitness": gbest_fit, "diagnostics": gbest_diag, "fitness_history": fit_hist}


def run_ga_experiments():
    config = get_config()
    total_iterations = config["simulation_params"]["total_iterations"]
    num_runs = config["simulation_params"]["num_independent_runs"]
    eps = float(config.get("simulation_params", {}).get("time_to_target_eps", 0.01))
    base_seed = int(config.get("simulation_params", {}).get("base_seed", 42))

    typical_days_data = load_typical_days_data()
    if not typical_days_data:
        return

    run_dir = os.path.join("runs", time.strftime("ga_%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print("Logs to:", os.path.abspath(run_dir))

    best_writer = BestConfigWriter(outdir=run_dir)
    timing_csv = os.path.join(run_dir, "timing_stats.csv")

    results = collections.defaultdict(list)
    start_time = time.time()

    for i in range(num_runs):
        print(f"\n{'='*25} Starting Statistical Run #{i+1}/{num_runs} {'='*25}")

        seed = base_seed + i

        # fresh env per run + evaluation counters wrapper
        base_env = MicrogridEnv(config, typical_days_data)
        env = EvalCounterEnv(base_env)

        timing_rows = []

        def _run_one(method_name, fn):
            env.reset_counters()
            t0 = time.perf_counter()
            res = fn(config, env, total_iterations, seed)
            wall_full = time.perf_counter() - t0

            res = attach_full_and_target_metrics(res, wall_full, env.ems_eval_count, env.ems_eval_time_s, eps=eps)
            results[method_name].append(res)

            # best config logging (keep behavior)
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

            return {
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

        # 1) SAC-Guided
        timing_rows.append(_run_one("GA with proposed RL-adjusted operator", run_sac_guided_ga))
        
        # 2) Guided GA
        timing_rows.append(_run_one("GA with proposed static operator", run_classic_ga_guided))
        
        # 3) GA baseline
        timing_rows.append(_run_one("GA", run_classic_ga_simple))


        # append timing csv
        df_timing = pd.DataFrame(timing_rows)
        if not os.path.exists(timing_csv):
            df_timing.to_csv(timing_csv, index=False)
        else:
            df_timing.to_csv(timing_csv, mode="a", header=False, index=False)

    print(f"\n\n{'='*20} STATISTICAL ANALYSIS COMPLETE {'='*20}")
    print(f"Total time for {num_runs} runs: {time.time() - start_time:.2f} seconds")
    print(f"Best-configs saved → {os.path.abspath(os.path.join(run_dir, 'best_configs.csv'))}")
    print(f"Timing stats saved → {os.path.abspath(timing_csv)}")

    # plotting behavior unchanged
    histories = {name: [res.get("fitness_history", []) for res in res_list] for name, res_list in results.items()}
    final_fits = {name: [h[-1] for h in h_list if len(h) > 0] for name, h_list in histories.items()}

    plot_statistical_results(histories_list=list(histories.values()), labels=list(histories.keys()))
    plot_final_fitness_boxplot(final_fits)


if __name__ == "__main__":
    run_ga_experiments()
