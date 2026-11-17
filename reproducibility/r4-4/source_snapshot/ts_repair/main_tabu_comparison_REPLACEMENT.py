# -*- coding: utf-8 -*-
"""
ts/main_tabu_comparison.py

Replacement version (drop-in):
- Add EvalCounterEnv wrapper to count env.evaluate_solution calls and time
- Add wall-clock timing (full run) and optional time-to-target metrics
- Unify curve/legend names for paper:
    1) "TS with proposed RL-adjusted operator"
    2) "TS with proposed static operator"
    3) "TS"
- Make statistical runs use different seeds: seed = base_seed + run_idx
- Write timing_stats.csv and best_configs.csv into a timestamped run folder

Notes:
- Keeps your original algorithm implementations and plotting calls unchanged.
- Does NOT require modifying other modules.
"""


import matplotlib as mpl
import matplotlib.pyplot as plt

# 强制清掉你当前 Python 会话里残留的全局绘图设置（Spyder/Jupyter 必备）
mpl.rcdefaults()
plt.rcdefaults()

# 你论文要的风格（按需改）
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 12

import os
import time
import random
import csv
import collections
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from config_tabu import get_config
from microgrid_env import MicrogridEnv
from sac_agent import SACAgent
from utils import plot_statistical_results, plot_final_fitness_boxplot

import ts_core
from classic_tabu_simple import run_classic_tabu_simple
from classic_tabu_guided import run_classic_tabu_guided

STATUS_MAP = {
    "success": 0,
    "fail_supply": 1,
    "fail_storage_overdischarge": 2,
    "fail_storage_overcharge": 3,
    "simulation_error": 4,
}


# =========================
# Reproducibility
# =========================
def set_global_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# =========================
# Timing + evaluation counter
# =========================
class EvalCounterEnv:
    """Wrap env to count evaluate_solution calls + time."""

    def __init__(self, env: Any):
        self.env = env
        self.reset_counters()

    def reset_counters(self) -> None:
        self.ems_eval_count = 0
        self.ems_eval_time_s = 0.0

    def evaluate_solution(self, x: Any):
        t0 = time.perf_counter()
        out = self.env.evaluate_solution(x)
        self.ems_eval_time_s += (time.perf_counter() - t0)
        self.ems_eval_count += 1
        return out

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def find_time_to_target_from_history(fitness_history: List[float], eps: float = 0.01):
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


# =========================
# Data loader
# =========================
def load_typical_days_data(data_folder: str = "data"):
    typical_days_data = []
    required = ["GHI", "Tair", "Vwind", "Load"]
    for i in range(5):
        fp = os.path.join(data_folder, f"typicalday{i}.csv")
        df = pd.read_csv(fp)
        miss = [c for c in required if c not in df.columns]
        if miss:
            print(f"!!! FATAL: '{fp}' missing {miss}")
            return None
        typical_days_data.append(df)
    print(f"Loaded {len(typical_days_data)} typical days from '{data_folder}'.")
    return typical_days_data


# =========================
# SAC-guided TS core (kept, with only seed handling intact)
# =========================
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


def run_sac_guided_ts(config, env, total_iterations, run_seed):
    print(f"\n--- Running SAC-Guided-TS (seed={run_seed}) ---")
    np.random.seed(run_seed)
    random.seed(run_seed)

    agent = SACAgent(config)
    state = ts_core.initialize_tabu_state(config, env, seed=run_seed)
    fit_hist, gbest_hist = [], collections.deque(maxlen=20)
    found_feasible = (state.get("best_f", float("inf")) < 1e6)

    # log folder (per run)
    base_log_dir = os.path.join(".", "runs_ts", "sac_guided")
    run_log_dir = os.path.join(base_log_dir, f"run_{run_seed}")
    os.makedirs(run_log_dir, exist_ok=True)

    log_path = os.path.join(run_log_dir, "iter_log.csv")
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        ["iter", "reward", "diag_prob", "gamma_fb", "lcoe", "state0", "state1", "state2", "state3"]
    )

    s = get_state_ts(state, 0, total_iterations, gbest_hist)

    for i in range(total_iterations):
        act = agent.choose_action(s)  # [diag_prob, gamma_feedback]
        old_best = float(state.get("best_f", float("inf")))

        state = ts_core.run_tabu_iteration_guided(state, act, env)

        is_first_feasible = (not found_feasible) and (float(state.get("best_f", float("inf"))) < 1e6)
        if is_first_feasible:
            found_feasible = True

        r = calculate_reward(old_best, float(state.get("best_f", float("inf"))), state.get("best_diag"), is_first_feasible, config)
        s2 = get_state_ts(state, i + 1, total_iterations, gbest_hist)

        agent.replay_buffer.add(s, act, r, s2, (i == total_iterations - 1))
        agent.learn()

        # write one row
        diag_prob = float(act[0])
        gamma_fb = float(act[1])
        lcoe = float(state.get("best_f", float("inf")))
        log_writer.writerow([i, float(r), diag_prob, gamma_fb, lcoe, float(s2[0]), float(s2[1]), float(s2[2]), float(s2[3])])

        s = s2
        gbest_hist.append(lcoe)
        fit_hist.append(lcoe)

    log_file.close()
    print(f"[SAC-Guided TS] iter_log saved to: {log_path}")

    return {
        "best_solution": state.get("best_x"),
        "best_fitness": state.get("best_f"),
        "diagnostics": state.get("best_diag"),
        "fitness_history": fit_hist,
    }


# =========================
# Best-config writer (lightweight, no extra deps)
# =========================
class BestConfigWriter:
    def __init__(self, outdir: str):
        self.outdir = outdir
        self.csv_path = os.path.join(outdir, "best_configs.csv")
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["method", "run", "seed", "status", "best_fitness", "best_solution"])

    def append(self, method: str, run_idx: int, seed: int, status: str, best_fitness: Any, best_solution: Any):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([method, run_idx, seed, status, best_fitness, best_solution])


# =========================
# Main experiments
# =========================
def run_tabu_experiments():
    config = get_config()

    total_iterations = int(config["simulation_params"]["total_iterations"])
    num_runs = int(config["simulation_params"]["num_independent_runs"])
    eps = float(config.get("simulation_params", {}).get("time_to_target_eps", 0.01))

    # Use a base seed; each statistical run uses base_seed + i
    base_seed = int(config.get("simulation_params", {}).get("base_seed", config.get("random_seed", 42)))

    set_global_seed(base_seed)

    tdata = load_typical_days_data()
    if not tdata:
        return

    # timestamped run folder
    run_dir = os.path.join("runs_ts", time.strftime("ts_%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print("Logs to:", os.path.abspath(run_dir))

    best_writer = BestConfigWriter(outdir=run_dir)
    timing_csv = os.path.join(run_dir, "timing_stats.csv")

    # unified method names for paper/plots
    NAME_RL = "TS with proposed RL-adjusted operator"
    NAME_STATIC = "TS with proposed static operator"
    NAME_BASE = "TS"

    results = collections.defaultdict(list)
    start_time = time.time()

    for run_idx in range(num_runs):
        print(f"\n{'='*25} Starting Statistical Run #{run_idx+1}/{num_runs} {'='*25}")

        seed_i = base_seed + run_idx

        # fresh env per run
        base_env = MicrogridEnv(config, tdata)
        env = EvalCounterEnv(base_env)

        timing_rows = []

        def _run_one(method_name, fn):
            env.reset_counters()
            t0 = time.perf_counter()
            res = fn(config, env, total_iterations, seed_i)
            wall_full = time.perf_counter() - t0

            res = attach_full_and_target_metrics(res, wall_full, env.ems_eval_count, env.ems_eval_time_s, eps=eps)
            results[method_name].append(res)

            # status
            diag = res.get("diagnostics") or {}
            status = (diag or {}).get("status", "unknown")

            best_writer.append(
                method=method_name,
                run_idx=run_idx,
                seed=seed_i,
                status=status,
                best_fitness=res.get("best_fitness"),
                best_solution=res.get("best_solution"),
            )

            row = {
                "run": run_idx,
                "seed": seed_i,
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

        # 1) SAC-guided (RL-adjusted operator)
        timing_rows.append(_run_one(NAME_RL, run_sac_guided_ts))

        # 2) classic guided TS (static operator)
        timing_rows.append(_run_one(NAME_STATIC, run_classic_tabu_guided))

        # 3) classic TS baseline
        timing_rows.append(_run_one(NAME_BASE, run_classic_tabu_simple))

        # append timing rows
        df_timing = pd.DataFrame(timing_rows)
        if not os.path.exists(timing_csv):
            df_timing.to_csv(timing_csv, index=False)
        else:
            df_timing.to_csv(timing_csv, mode="a", header=False, index=False)

    print(f"\n\n{'='*20} STATISTICS COMPLETE {'='*20}")
    print(f"Total time for {num_runs} runs: {time.time() - start_time:.2f} seconds")
    print(f"Best-configs saved → {os.path.abspath(os.path.join(run_dir, 'best_configs.csv'))}")
    print(f"Timing stats saved → {os.path.abspath(timing_csv)}")

    # Plotting (keep behavior)
    histories = {name: [res.get("fitness_history", []) for res in res_list] for name, res_list in results.items()}
    finals = {name: [h[-1] for h in h_list if len(h) > 0] for name, h_list in histories.items()}

    plot_statistical_results(list(histories.values()), list(histories.keys()))
    plot_final_fitness_boxplot(finals)
    
    plot_statistical_results(list(histories.values()), list(histories.keys()))
    plot_final_fitness_boxplot(finals)

if __name__ == "__main__":
    run_tabu_experiments()
