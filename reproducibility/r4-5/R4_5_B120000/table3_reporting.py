"""Actual integer designs and a descriptive historical Table 3 comparison."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

EQUIPMENT = ['nPV', 'nWT', 'nEC', 'nFC', 'ntank', 'nbat']
BASE = Path(__file__).resolve().parent


def expand_equipment(raw):
    raw = raw.copy()
    rows = []
    for _, r in raw.iterrows():
        values = r.equipment_counts
        if isinstance(values, str):
            values = json.loads(values)
        if not r.feasible:
            rows.append([pd.NA] * 6)
            continue
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 6:
            raise ValueError('A feasible result must have six equipment counts.')
        if any(not np.isfinite(x) or x < 0 or int(x) != x for x in values):
            raise ValueError('Equipment counts must be finite nonnegative integers.')
        rows.append([int(x) for x in values])
    for j, key in enumerate(EQUIPMENT):
        raw[key] = pd.array([row[j] for row in rows], dtype='Int64')
    return raw


def design_tables(raw):
    """No average configuration is presented as a realizable optimum."""
    columns = ['algorithm', 'variant', 'seed', 'budget', 'feasible', 'lcoe'] + EQUIPMENT + [
        'target', 'target_reached', 'target_first_calls', 'target_first_search_s',
        'target_censored', 'ems_calls', 'process_wall_s', 'sac_total_s', 'sac_updates']
    designs = raw[columns].copy()
    representatives = []
    for (alg, var), g in designs.groupby(['algorithm', 'variant'], sort=False):
        feasible = g[g.feasible].sort_values(['lcoe', 'seed'])
        if len(feasible):
            # Choose an actual lower-median feasible run, not mean counts or a best-of-22 claim.
            row = feasible.iloc[(len(feasible)-1)//2].to_dict()
            row['selection_rule'] = 'Actual lower-median feasible LCOE run; ties by seed'
        else:
            row = {key: None for key in columns}
            row.update(algorithm=alg, variant=var, feasible=False,
                       selection_rule='No feasible run; no representative configuration')
        row['feasible_run_count'] = len(feasible)
        representatives.append(row)
    reps = pd.DataFrame(representatives)
    reference = pd.read_csv(BASE/'table3_reference.csv')
    reference['reference_source'] = 'Original manuscript Table 3, PDF page 21; printed precision'
    reference['reported_seed'] = 42
    reference['provenance_note'] = 'Seed 42 reported by author; exact historical executable provenance not authenticated for all rows'
    new42 = designs[designs.seed == 42].copy()
    new42['new_seed42_present'] = True
    comparison = reference.merge(new42, on=['algorithm', 'variant'], how='left', validate='one_to_one')
    comparison['new_seed42_present'] = comparison['new_seed42_present'].eq(True)
    comparison['lcoe_difference_new_minus_historical'] = comparison['lcoe'] - comparison['historical_lcoe']
    comparison['comparison_scope'] = ('Descriptive only: new budget/tuning/batch/progress rules; '
        'new common pilot target vs historical run-specific target; timing is not directly comparable')
    return {'Seed42_vs_original': comparison, 'Designs_all_seeds': designs,
            'Representative_designs': reps, 'Historical_Table3': reference}
