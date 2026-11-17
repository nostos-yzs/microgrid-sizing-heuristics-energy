"""Boundary tests for device reporting and independent-pilot checks."""
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from table3_reporting import expand_equipment, design_tables, EQUIPMENT
from budget_review import review_pilot


def row(seed, lcoe, feasible=True, variant='sac'):
    return dict(algorithm='GA', variant=variant, seed=seed, budget=120000,
        feasible=feasible, lcoe=lcoe, equipment_counts=[seed, 1, 2, 3, 4, 5] if feasible else None,
        target=.5, target_reached=False, target_first_calls=None, target_first_search_s=None,
        target_censored=True, ems_calls=120000, process_wall_s=2., sac_total_s=1., sac_updates=3)


class ReportingTests(unittest.TestCase):
    def test_representative_is_actual_median_run_and_no_feasible_stays_blank(self):
        raw = expand_equipment(pd.DataFrame([row(0,3.), row(2,1.), row(42,2.),
                                            row(42,None,False,'baseline')]))
        frames = design_tables(raw)
        reps = frames['Representative_designs'].set_index('variant')
        self.assertEqual(reps.loc['sac','seed'],42)
        self.assertEqual(reps.loc['sac','nPV'],42)
        self.assertTrue(pd.isna(reps.loc['baseline','nPV']))
        comparison = frames['Seed42_vs_original'].set_index(['algorithm','variant'])
        self.assertEqual(comparison.loc[('GA','sac'),'lcoe'],2.)
        self.assertAlmostEqual(comparison.loc[('GA','sac'),'historical_lcoe'],.412423)
        self.assertFalse(comparison.loc[('PSO','sac'),'new_seed42_present'])
        self.assertTrue(pd.isna(comparison.loc[('PSO','sac'),'lcoe']))

    def test_missing_seed42_is_never_filled_with_another_seed(self):
        frames = design_tables(expand_equipment(pd.DataFrame([row(0,1.)])))
        self.assertFalse(frames['Seed42_vs_original'].new_seed42_present.any())
        self.assertTrue(frames['Seed42_vs_original'].lcoe.isna().all())

    def test_invalid_device_counts_are_not_silently_rounded(self):
        r=row(42,1.);r['equipment_counts']=[1.5,1,2,3,4,5]
        with self.assertRaises(ValueError):expand_equipment(pd.DataFrame([r]))

    def test_late_first_feasibility_and_no_learning_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job=root/'runs'/'pilot'/'GA_sac_original_seed2001';job.mkdir(parents=True)
            artifact=job/'attempt_001';artifact.mkdir()
            result=dict(budget=120000,ems_calls=120000,feasible=True,lcoe=.4,sac_updates=1,
                        process_wall_s=1.,artifact_dir=str(artifact.relative_to(root)))
            (job/'result.json').write_text(json.dumps(result),encoding='utf-8')
            (artifact/'improvements.csv').write_text('ems_calls,best_feasible_lcoe,elapsed_search_s\n110000,0.4,1.0\n',encoding='utf-8')
            protocol=dict(algorithms=['GA'],variants=['sac'],pilot_seeds=[2001],pilot_budget=120000,budget=120000)
            review=review_pilot(root,protocol,{'GA_sac':'original'})
            self.assertTrue(review.first_feasible_in_last_20_percent.iloc[0])
            self.assertIsNone(review.relative_gain_in_last_20_percent.iloc[0])
            result['sac_updates']=0
            (job/'result.json').write_text(json.dumps(result),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'no gradient updates'):
                review_pilot(root,protocol,{'GA_sac':'original'})


if __name__=='__main__':unittest.main(verbosity=2)
