"""Independent pilot diagnostics recorded before the formal seeds are evaluated."""
import json
from pathlib import Path
import pandas as pd
from worker import save_json


def review_pilot(root, protocol, selection):
    root = Path(root)
    rows = []
    for alg in protocol['algorithms']:
        for variant in protocol['variants']:
            candidate = selection[alg+'_'+variant]
            for seed in protocol['pilot_seeds']:
                job = root/'runs'/'pilot'/f'{alg}_{variant}_{candidate}_seed{seed}'
                r = json.loads((job/'result.json').read_text(encoding='utf-8'))
                if r['ems_calls'] != protocol['pilot_budget']:
                    raise RuntimeError('Pilot failed to consume its exact EMS budget.')
                if variant == 'sac' and r['sac_updates'] < 1:
                    raise RuntimeError('A pilot SAC run made no gradient updates. Revise settings in a new output directory.')
                improvements = pd.read_csv(root/r['artifact_dir']/'improvements.csv')
                cut = .8*r['budget']
                before = improvements[improvements.ems_calls <= cut]
                late = improvements[improvements.ems_calls > cut]
                at80 = float(before.best_feasible_lcoe.iloc[-1]) if len(before) else None
                final = r['lcoe'] if r['feasible'] else None
                gain = (at80-final)/abs(at80) if at80 not in (None, 0) and final is not None else None
                rows.append(dict(algorithm=alg, variant=variant, candidate=candidate, seed=seed,
                    budget=r['budget'], ems_calls=r['ems_calls'], feasible=r['feasible'], final_lcoe=final,
                    lcoe_at_80_percent_budget=at80, relative_gain_in_last_20_percent=gain,
                    improvement_events_in_last_20_percent=len(late),
                    first_feasible_in_last_20_percent=bool(final is not None and at80 is None),
                    last_improvement_call=int(improvements.ems_calls.iloc[-1]) if len(improvements) else None,
                    sac_updates=r['sac_updates'], process_wall_s=r['process_wall_s']))
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'pilot_budget_review.csv', index=False, encoding='utf-8-sig')
    save_json(root/'pilot_budget_review.json', dict(
        pilot_runs=len(rows), pilot_budget=protocol['pilot_budget'], formal_budget=protocol['budget'],
        exact_pilot_budgets_passed=True, all_sac_pilots_have_gradient_updates=True,
        minimum_pilot_sac_updates=int(frame.loc[frame.variant=='sac','sac_updates'].min()),
        historical_observed_max_full_calls=112500,
        formal_budget_covers_observed_historical_call_count=protocol['budget']>=112500,
        pilot_runs_with_late_improvements=int((frame.improvement_events_in_last_20_percent>0).sum()),
        validation_only=protocol.get('validation_only',False),
        interpretation='Execution checks only, not convergence certification or a guarantee of historical Table 3 reproduction. Late gains are descriptive; no post-hoc budget adjustment from formal seeds.'))
    return frame
