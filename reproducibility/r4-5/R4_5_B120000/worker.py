"""One isolated, budget-limited run. No full experiment starts on import."""
from __future__ import annotations
import argparse, copy, csv, hashlib, importlib, json, os, random, sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ENTRIES = {
    'GA': ('ga_repair', 'main_ga_comparison_patched',
           ['run_classic_ga_simple', 'run_classic_ga_guided', 'run_sac_guided_ga']),
    'PSO': ('pso_repair', 'main_comparison',
            ['run_classic_pso_simple', 'run_classic_pso_guided', 'run_sac_guided_pso']),
    'SA': ('sa_repair', 'main_sa_comparison_replaced_v3',
           ['run_sa_traditional', 'run_sa_guided', 'run_sa_rl_guided']),
    'TS': ('ts_repair', 'main_tabu_comparison_REPLACEMENT',
           ['run_classic_tabu_simple', 'run_classic_tabu_guided', 'run_sac_guided_ts']),
}
VARIANTS = ['baseline', 'static', 'sac']


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if hasattr(x, 'tolist'): return clean(x.tolist())
    if isinstance(x, float) and not __import__('math').isfinite(x): return None
    return x


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


class BudgetExhausted(BaseException):
    """BaseException deliberately bypasses legacy `except Exception` handlers."""


class BudgetEnv:
    def __init__(self, env, budget, target, interval, started):
        self.env, self.budget, self.target = env, budget, target
        self.interval, self.started = interval, started
        self.calls = 0; self.ems_seconds = 0.; self.best = float('inf')
        self.best_x = None; self.best_diag = None; self.target_call = None
        self.target_seconds = None; self.curve = []; self.improvements = []
        self.status_counts = {}; self.first_inputs = []; self.first_cycle_calls = None

    def __getattr__(self, key): return getattr(self.env, key)

    def simulate_day_trace(self, *args, **kwargs):
        # Legacy display-only traces are disabled, not extra uncounted EMS calls.
        import pandas as pd
        return pd.DataFrame()

    def evaluate_solution(self, x):
        import numpy as np
        if self.calls >= self.budget: raise BudgetExhausted()
        if len(self.first_inputs) < 150: self.first_inputs.append(np.asarray(x, float).tolist())
        t = time.perf_counter()
        value, diag = self.env.evaluate_solution(x)
        self.ems_seconds += time.perf_counter() - t
        self.calls += 1
        status = (diag or {}).get('status', 'unknown')
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        valid = status == 'success' and np.isfinite(value) and float(value) < 1e6
        elapsed = time.perf_counter() - self.started
        if valid and value < self.best:
            self.best = float(value); self.best_x = np.asarray(x, float).copy(); self.best_diag = diag
            self.improvements.append({'ems_calls': self.calls, 'best_feasible_lcoe': self.best,
                                      'elapsed_search_s': elapsed})
        if valid and self.target is not None and value <= self.target and self.target_call is None:
            self.target_call, self.target_seconds = self.calls, elapsed
        if self.calls == 1 or self.calls % self.interval == 0 or self.calls == self.budget:
            self.curve.append({'ems_calls': self.calls, 'best_feasible_lcoe': self.best,
                               'elapsed_search_s': elapsed})
        return value, diag


class NullLogger:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name): return lambda *a, **k: None


def configuration(main, alg, variant, candidate, seed, protocol):
    c = copy.deepcopy(main.get_config())
    c['random_seed'] = seed
    c['simulation_params'].update(base_seed=seed, random_seed=seed, num_independent_runs=1)
    # All runs use exactly the same deterministic typical days, with no MC resampling.
    c['stochasticity_params']['enabled'] = False
    for exp in c.get('experiments', {}).values(): exp['random_seed'] = seed
    sac = c.get('sac_agent_params')
    if sac:
        sac.update(device=protocol['device'], batch_size=protocol['sac_batch_size'])
    if alg == 'SA':
        sac = c['experiments']['sac_guided']['sac_agent_params']
        sac.update(device=protocol['device'], batch_size=protocol['sac_batch_size'])
    if candidate == 'alternative':
        if variant == 'sac':
            sac['actor_lr'] *= 0.5; sac['critic_lr'] *= 0.5
        elif variant == 'static':
            if alg == 'GA': c['ga_params']['classic_params']['guided']['parent_treat_strength'] *= 0.8
            if alg == 'PSO': c['classic_guided_params']['gamma_feedback'] *= 0.8
            if alg == 'SA': c['experiments']['classic_guided']['gamma_fb'] *= 0.8
            if alg == 'TS': c['ts_params']['guided']['gamma_feedback'] *= 0.8
        else:
            if alg == 'GA': c['ga_params']['classic_params']['simple']['mutation_strength'] *= 0.8
            if alg == 'PSO': c['pso_core_params']['w'] *= 0.8
            if alg == 'SA': c['sa_params']['schedule']['T0'] *= 0.8
            if alg == 'TS': c['ts_params']['sigma_random'] *= 0.8
    return c


def run(spec, directory):
    import numpy as np
    import pandas as pd
    import torch
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    alg, variant, seed = spec['algorithm'], spec['variant'], spec['seed']
    folder, entry, functions = ENTRIES[alg]
    source = BASE / 'source_snapshot' / folder
    sys.path.insert(0, str(source))
    main = importlib.import_module(entry)
    p = spec['protocol']; budget = int(spec['budget'])
    torch.set_num_threads(p['torch_threads'])
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    c = configuration(main, alg, variant, spec['candidate'], seed, p)
    # Large loop guard; termination is enforced by the EMS call boundary.
    c['simulation_params']['total_iterations'] = budget
    if alg == 'SA':
        c['simulation_params'].update(n_layers=budget, inner_loops=1)
    save_json(directory/'effective_config.json', c)
    total_start = time.perf_counter()
    days = [pd.read_csv(source/'data'/f'typicalday{i}.csv') for i in range(5)]
    base_env = main.MicrogridEnv(c, days)
    search_start = time.perf_counter()
    env = BudgetEnv(base_env, budget, spec.get('target'), p['checkpoint_every'], search_start)
    timers = dict(sac_init_s=0., sac_action_s=0., sac_replay_s=0., sac_learn_s=0.)
    counts = dict(sac_actions=0, sac_transitions=0, sac_learn_calls=0, sac_updates=0)
    action_rows = []; initial_arrays = []; agent_refs = []

    def synchronize():
        if str(p['device']).startswith('cuda'): torch.cuda.synchronize()

    def measured(key, fn, *args, **kwargs):
        synchronize(); t = time.perf_counter()
        try: return fn(*args, **kwargs)
        finally:
            synchronize(); timers[key] += time.perf_counter()-t

    original_agent = main.SACAgent
    class TimedAgent(original_agent):
        def __init__(self, cfg):
            measured('sac_init_s', super().__init__, cfg)
            agent_refs.append(self)
            original_add = self.replay_buffer.add
            def add(state, action, reward, next_state, done):
                measured('sac_replay_s', original_add, state, action, reward, next_state,
                         bool(done or env.calls >= budget))
                counts['sac_transitions'] += 1
                action_rows.append({'step': counts['sac_transitions'], 'ems_calls': env.calls,
                    'action_returned_0': float(self.last_action[0]), 'action_returned_1': float(self.last_action[1]),
                    'action_applied_0': float(action[0]), 'action_applied_1': float(action[1]),
                    'reward': float(reward), 'best_feasible_lcoe': env.best,
                    'budget_fraction': env.calls / budget})
                if env.first_cycle_calls is None: env.first_cycle_calls = env.calls
            self.replay_buffer.add = add
        def choose_action(self, state):
            self.last_action = measured('sac_action_s', super().choose_action, state)
            counts['sac_actions'] += 1
            return self.last_action
        def learn(self):
            ready = len(self.replay_buffer) >= self.batch_size
            measured('sac_learn_s', super().learn)
            counts['sac_learn_calls'] += 1
            counts['sac_updates'] += int(ready)
    main.SACAgent = TimedAgent
    if hasattr(main, 'ExperimentLogger'): main.ExperimentLogger = NullLogger

    # Shared initialization is enforced independently of network initialization.
    def wrap_init(mod, name, kind):
        original = getattr(mod, name)
        def initialize(*args, **kwargs):
            np.random.seed(seed)
            result = original(*args, **kwargs)
            if kind == 'population': arrays = [np.asarray(result[0], dtype=float)]
            else: arrays = [result['particles'], result['velocities']]
            initial_arrays.extend(np.asarray(a, dtype='<f8').tobytes() for a in arrays)
            return result
        setattr(mod, name, initialize)
    if alg == 'GA': wrap_init(main.ga_core, 'initialize_population', 'population')
    if alg == 'PSO':
        for module_name in ['pso_core_sac', 'pso_core_simple']:
            wrap_init(importlib.import_module(module_name), 'initialize_swarm', 'swarm')
    if alg == 'SA':
        original_init = main.initialize_sa
        def sa_init(*args, **kwargs):
            result = original_init(*args, **kwargs)
            initial_arrays.append(np.asarray(result['x'], dtype='<f8').tobytes())
            return result
        main.initialize_sa = sa_init
        original_step = main.run_sa_iteration
        def sa_step(state, action, wrapped_env):
            sched = state['schedule']
            # Preserve the original 60-layer temperature endpoints over the EMS budget.
            layer = 59 * min(1., env.calls / budget)
            state['T'] = max(sched['Tmin'], sched['T0'] * sched['alpha'] ** layer)
            return original_step(state, action, wrapped_env)
        main.run_sa_iteration = sa_step
    if alg == 'TS':
        original_init = main.ts_core.initialize_tabu_state
        def ts_init(*args, **kwargs):
            result = original_init(*args, **kwargs)
            initial_arrays.append(np.asarray(result['x'], dtype='<f8').tobytes())
            return result
        main.ts_core.initialize_tabu_state = ts_init

    state_name = {'GA':'get_state_ga','PSO':'get_state','SA':'get_state_sa','TS':'get_state_ts'}[alg]
    original_state = getattr(main, state_name)
    def budget_state(*args, **kwargs):
        state = original_state(*args, **kwargs)
        state[0] = env.calls / budget
        return state
    setattr(main, state_name, budget_state)
    # Preserve original action mapping and reward implementation (including SA's legacy mapping).
    # R4.5 is a comparison protocol, not an undocumented algorithm correction.
    function = getattr(main, functions[VARIANTS.index(variant)])
    stopped = False
    try: function(c, env, budget, seed)
    except BudgetExhausted: stopped = True
    search_s = time.perf_counter() - search_start
    run_s = time.perf_counter() - total_start
    if env.calls != budget:
        raise RuntimeError(f'Algorithm stopped at {env.calls} calls; required {budget}.')
    if not initial_arrays: raise RuntimeError('Initialization audit failed.')
    if env.best_x is not None:
        # No extra unbudgeted final EMS evaluation: incumbent was already evaluated.
        solution = env.best_x.tolist()
        integers = np.maximum(0, np.ceil(env.best_x)).astype(int).tolist()
    else: solution = integers = None
    sac_total = sum(timers.values())
    row = {k:spec[k] for k in ['stage','algorithm','variant','candidate','seed','budget']}
    row.update(feasible=env.best_x is not None, lcoe=env.best, solution=solution, equipment_counts=integers,
        diagnostics=env.best_diag, ems_calls=env.calls, ems_s=env.ems_seconds,
        target=spec.get('target'), target_reached=env.target_call is not None,
        target_first_calls=env.target_call, target_first_search_s=env.target_seconds,
        target_censored=env.target_call is None, target_observation_calls=env.target_call or budget,
        run_wall_s=run_s, search_wall_s=search_s, setup_s=run_s-search_s,
        sac_total_s=sac_total, search_non_sac_s=max(0.,search_s-sac_total),
        initialization_sha256=hashlib.sha256(b''.join(initial_arrays)).hexdigest(),
        first_completed_sac_cycle_calls=env.first_cycle_calls,
        first_ems_inputs_sha256=hashlib.sha256(json.dumps(env.first_inputs).encode()).hexdigest(),
        status_counts=env.status_counts, budget_stop=stopped, **timers, **counts)
    save_json(directory/'worker_result.json',row)
    pd.DataFrame(env.curve).replace([np.inf,-np.inf],np.nan).to_csv(directory/'convergence.csv',index=False)
    pd.DataFrame(env.improvements,columns=['ems_calls','best_feasible_lcoe','elapsed_search_s']).to_csv(directory/'improvements.csv',index=False)
    pd.DataFrame(action_rows).replace([np.inf,-np.inf],np.nan).to_csv(directory/'sac_actions_rewards.csv',index=False)
    save_json(directory/'environment.json',{'python':sys.version,'platform':__import__('platform').platform(),
       'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,
       'device':p['device'],'threads':p['torch_threads']})
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--directory',required=True)
    args=ap.parse_args();run(json.loads(Path(args.spec).read_text(encoding='utf-8')),args.directory)
if __name__=='__main__': main()
