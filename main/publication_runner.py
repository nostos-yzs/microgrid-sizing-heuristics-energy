"""Run algorithm comparisons with quiet summaries and one results plot each.

The search and environment implementations in algorithms/ are retained.
Configurations, LCOE and curves come directly from the computed results.
"""
import argparse
import contextlib
import csv
import importlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
EQUIPMENT = ("n_PV", "n_WIND", "n_EC", "n_FC", "n_TANK", "n_BESS")
SPECS = {
    "ga": ("config_ga", ("run_sac_guided_ga", "run_classic_ga_guided", "run_classic_ga_simple")),
    "pso": ("config_sac", ("run_sac_guided_pso", "run_classic_pso_guided", "run_classic_pso_simple")),
    "sa": ("config_sa", ("run_sa_rl_guided", "run_sa_guided", "run_sa_traditional")),
    "ts": ("config_tabu", ("run_sac_guided_ts", "run_classic_tabu_guided", "run_classic_tabu_simple")),
}


def decode_solution(solution):
    """Use exactly the positive-ceil/nonpositive-zero rule used by the EMS."""
    if solution is None:
        return None
    values = [float(x) for x in solution]
    if len(values) != 6 or not all(math.isfinite(x) for x in values):
        raise ValueError("A solution must contain six finite equipment variables.")
    return dict(zip(EQUIPMENT, [math.ceil(x) if x > 0 else 0 for x in values]))


class EvalCounterEnv:
    """Count all evaluator calls made by a search, including failed candidates."""
    def __init__(self, env):
        self.env = env
        self.count = 0
        self.elapsed = 0.0

    def evaluate_solution(self, solution):
        start = time.perf_counter()
        result = self.env.evaluate_solution(solution)
        self.elapsed += time.perf_counter() - start
        self.count += 1
        return result

    def __getattr__(self, name):
        return getattr(self.env, name)


def target_metrics(history, wall, count, eval_time, eps=0.01):
    """Retain the archive's proportional estimates, NOT direct target counts."""
    if not history or not math.isfinite(float(history[-1])) or float(history[-1]) >= 1e6:
        return {"target_iteration_index": None, "target_threshold": None,
                "wall_time_to_target_estimate_s": None, "eval_to_target_estimate": None,
                "ems_time_to_target_estimate_s": None}
    threshold = float(history[-1]) * (1.0 + eps)
    index = next(i for i, x in enumerate(history) if float(x) <= threshold)
    fraction = index / (len(history) - 1) if len(history) > 1 else 1.0
    return {"target_iteration_index": index, "target_threshold": threshold,
            "wall_time_to_target_estimate_s": wall * fraction,
            "eval_to_target_estimate": round(count * fraction),
            "ems_time_to_target_estimate_s": eval_time * fraction}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_experiment(algo):
    """Each CMD entry runs in its own process, isolating same-named modules."""
    sys.path.insert(0, str(ROOT / "algorithms" / algo))
    config_module, functions = SPECS[algo]
    config = importlib.import_module(config_module).get_config()
    env_class = importlib.import_module("microgrid_env").MicrogridEnv
    experiment = importlib.import_module("experiment")
    methods = []
    for name in functions:
        if hasattr(experiment, name):
            methods.append(getattr(experiment, name))
        else:
            module = importlib.import_module(name.replace("run_", "", 1))
            methods.append(getattr(module, name))
    return config, env_class, experiment, methods


def make_figure(algo, records, output, show):
    """Plot actual histories with their own x arrays (TS: 200/201 points)."""
    import numpy as np
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    labels = [f"{algo.upper()} with proposed RL-adjusted operator",
              f"{algo.upper()} with proposed static operator", algo.upper()]
    font = "Times New Roman" if any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist) else "DejaVu Serif"
    with plt.rc_context({"font.family": font, "font.size": 12, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ("royalblue", "coral", "mediumseagreen")
        data_columns = {}
        for method_index, label in enumerate(labels):
            histories = [r["fitness_history"] for r in records if r["method_index"] == method_index]
            max_length = max((len(h) for h in histories), default=0)
            values = np.full((len(histories), max_length), np.nan)
            for i, history in enumerate(histories):
                h = np.asarray(history, dtype=float)
                h = np.where(np.isfinite(h) & (h < 1e6), h, np.nan)
                values[i, :len(h)] = h
            count = np.sum(np.isfinite(values), axis=0)
            mean = np.full(max_length, np.nan)
            std = np.full(max_length, np.nan)
            valid = count > 0
            mean[valid] = np.nansum(values[:, valid], axis=0) / count[valid]
            std[valid] = np.sqrt(np.nansum((values[:, valid] - mean[valid]) ** 2, axis=0) / count[valid])
            x = np.arange(max_length)
            ax.plot(x, mean, label=label, color=colors[method_index], linewidth=1.5)
            ax.fill_between(x, mean-std, mean+std, color=colors[method_index], alpha=0.20)
            data_columns[f"{label}_mean"] = mean
            data_columns[f"{label}_std"] = std
        ax.set_title(f"Results under the {algo.upper()} algorithm")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("LCOE ($/kWh)")
        ax.grid(False)
        ax.legend(frameon=False, fontsize=10, loc="upper right")
        fig.tight_layout()
        stem = f"results_{algo}"
        fig.savefig(str(output / (stem + ".pdf")), bbox_inches="tight")
        fig.savefig(str(output / (stem + ".png")), dpi=300, bbox_inches="tight")
        # Export unequal-length histories using blank cells, never truncation/filling.
        length = max((len(v) for v in data_columns.values()), default=0)
        with (output / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["iteration"] + list(data_columns))
            for i in range(length):
                writer.writerow([i] + [float(v[i]) if i < len(v) and np.isfinite(v[i]) else "" for v in data_columns.values()])
        fig.canvas.manager.set_window_title(f"Results under the {algo.upper()} algorithm")
    return stem, fig


def print_summary(algo, records, output, stem, smoke):
    title = f"Results under the {algo.upper()} algorithm"
    print("\n" + title)
    print("=" * len(title))
    if smoke:
        print("SMOKE TEST ONLY: reduced search; not a full optimization run.")
    for index, suffix in enumerate((" + RL Operator", " + Static Operator", "")):
        candidates = [r for r in records if r["method_index"] == index]
        feasible = [r for r in candidates if r["feasible"]]
        print(f"\n[{algo.upper()}{suffix}]")
        if not feasible:
            print("  No feasible configuration found (penalty values are NOT LCOE).")
            continue
        best = min(feasible, key=lambda r: r["best_fitness"])
        print(f"  LCOE ($/kWh) = {best['best_fitness']:.6f}")
        print("  Configuration: " + ", ".join(f"{name}={best['configuration'][name]}" for name in EQUIPMENT))
        print(f"  Seed = {best['seed']}")
    print(f"\nResults plot (PDF): {output / (stem + '.pdf')}")
    print(f"Same figure (PNG): {output / (stem + '.png')}")
    print(f"Result data: {output / 'results.json'}")
    sys.stdout.flush()


def main(algo):
    parser = argparse.ArgumentParser(description=f"Results under the {algo.upper()} algorithm (default seed: 10).")
    parser.add_argument("--no-show", action="store_true", help="Save the results plot without opening a plot window.")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: outputs/<algorithm>).")
    parser.add_argument("--seed", type=int, default=10, help="Random seed (default: 10).")
    parser.add_argument("--runs", type=int, help="Independent repetitions per method (default: configured value, normally 1).")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Override the original auto-selected PyTorch device.")
    parser.add_argument("--smoke-test", action="store_true", help="Very short interface check; not a full optimization run.")
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib

    output = (args.output_dir.resolve() if args.output_dir else ROOT / "outputs" / (algo + ("_smoke" if args.smoke_test else "")))
    output.mkdir(parents=True, exist_ok=True)
    config, env_class, experiment, functions = load_experiment(algo)
    sim = config["simulation_params"]
    seed = args.seed
    sim["base_seed"] = seed
    runs = args.runs if args.runs is not None else int(sim.get("num_independent_runs", 1))
    if runs < 1:
        parser.error("--runs must be positive")
    if args.device:
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable in this Python environment.")
        if "sac_agent_params" in config:
            config["sac_agent_params"]["device"] = args.device
        if algo == "sa":
            config["experiments"]["sac_guided"]["sac_agent_params"]["device"] = args.device
    if args.smoke_test:
        if algo == "sa":
            sim["n_layers"], sim["inner_loops"] = 2, 3
        else:
            sim["total_iterations"] = 4
    iterations = sim["n_layers"] * sim["inner_loops"] if algo == "sa" else sim["total_iterations"]
    source = ROOT / "algorithms" / algo
    typical_days = []
    for i in range(5):
        path = source / "data" / f"typicalday{i}.csv"
        df = pd.read_csv(path)
        if len(df) != 24 or not {"GHI", "Tair", "Vwind", "Load"}.issubset(df.columns):
            raise ValueError(f"Expected 24 hourly rows and GHI/Tair/Vwind/Load columns: {path}")
        typical_days.append(df)
    if algo == "ts":
        experiment.set_global_seed(seed)
    records = []
    for run in range(runs):
        run_seed = seed + run
        # Keep all configured RNG seeds aligned with the seed used by the methods.
        config["random_seed"] = run_seed
        sim["base_seed"] = run_seed
        if "random_seed" in sim:
            sim["random_seed"] = run_seed
        if algo == "sa":
            for entry in config["experiments"].values():
                entry["random_seed"] = run_seed
        env = env_class(config, typical_days)
        for method_index, function in enumerate(functions):
            counted = EvalCounterEnv(env)
            start = time.perf_counter()
            # Only old banners are suppressed. Exceptions and stderr remain visible.
            with contextlib.redirect_stdout(io.StringIO()):
                result = function(config, counted, iterations, run_seed)
            wall = time.perf_counter() - start
            solution = result.get("best_solution")
            configuration = decode_solution(solution)
            fitness = float(result["best_fitness"])
            diagnostic = {"status": "no_solution"}
            verified = None
            if solution is not None:
                verified, diagnostic = env.evaluate_solution(solution)
            feasible = (verified is not None and math.isfinite(fitness) and fitness < 1e6
                        and diagnostic.get("status") in ("success", "success_mc_avg")
                        and math.isclose(float(verified), fitness, rel_tol=1e-9, abs_tol=1e-10))
            if solution is not None and math.isfinite(fitness) and fitness < 1e6 and not feasible:
                raise RuntimeError(f"{algo.upper()} method {method_index}: reported best objective did not pass independent evaluation.")
            history = [float(v) for v in result.get("fitness_history", [])]
            record = {"algorithm": algo.upper(), "method_index": method_index,
                      "method": algo.upper() + (" + RL Operator", " + Static Operator", "")[method_index],
                      "run": run + 1, "seed": run_seed, "feasible": feasible,
                      "best_fitness": fitness, "lcoe": fitness if feasible else None,
                      "raw_solution": list(solution) if solution is not None else None,
                      "configuration": configuration, "diagnostics": diagnostic,
                      "verified_objective": float(verified) if verified is not None else None,
                      "fitness_history": history, "wall_time_full_s": wall,
                      "ems_eval_count_full": counted.count, "ems_eval_time_full_s": counted.elapsed}
            record.update(target_metrics(history, wall, counted.count, counted.elapsed, float(sim.get("time_to_target_eps", 0.01))))
            records.append(record)
    # Save audit data before opening the only plot window.
    write_json(output / "results.json", records)
    write_json(output / "run_metadata.json", {"algorithm": algo, "seed": seed, "runs": runs,
               "smoke_test": args.smoke_test, "configuration": config,
               "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "torch": torch.__version__, "matplotlib": matplotlib.__version__,
               "target_metrics": "Proportional estimates based on the first history index <= 1.01 * final best; not direct target counters.",
               "equipment_decoding": "ceil(x) for positive x; zero otherwise; no forced hydrogen equipment.",
               "history_indexing": "Original indexing retained: TS baseline/static include initialization (201 points); TS RL has 200 points."})
    fields = ["method", "run", "seed", "feasible", "lcoe"] + list(EQUIPMENT) + ["wall_time_full_s", "ems_eval_count_full", "wall_time_to_target_estimate_s", "eval_to_target_estimate"]
    with (output / "best_configurations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {k: record.get(k) for k in fields}
            row.update(record["configuration"] or {})
            writer.writerow(row)
    # Build exactly one panel; print the final table before its GUI blocks.
    stem, fig = make_figure(algo, records, output, show=not args.no_show)
    print_summary(algo, records, output, stem, args.smoke_test)
    import matplotlib.pyplot as plt
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit("Run run_ga.py, run_pso.py, run_sa.py or run_ts.py instead.")
