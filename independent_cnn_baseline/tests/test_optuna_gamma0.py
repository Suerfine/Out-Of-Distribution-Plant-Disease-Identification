"""Corrected controller tests: generated data only; no DPD training/downloads."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import torch

from independent_cnn_baseline.engine import read_json
from independent_cnn_baseline.optuna_gamma0 import (OBJECTIVE, R0, RUN_NAME, SearchStop,
    open_study, request_trial, run, run_trial, snapshot, trial_config, validate_spec, verify_terminal)
from independent_cnn_baseline.optuna_search import SPACE
from independent_cnn_baseline.tests.test_study import fixture, TinyNet

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(2)


def configured(folder):
    cfg, frames, pairs, cache = fixture(folder)
    cfg['selection_objective'] = dict(OBJECTIVE)
    return cfg, frames, pairs, cache


class GammaZeroSearchTests(unittest.TestCase):
    def test_fresh_r0_and_no_calibrated_incumbent_import_or_duplicate_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            study = open_study(root, 'gamma0-id')
            self.assertEqual([t.state.name for t in study.trials], ['WAITING'])
            self.assertEqual(study.user_attrs['objective'], OBJECTIVE)
            trial = request_trial(study, 8)
            self.assertEqual(trial.number, 0)
            self.assertEqual(trial.params, R0)
            self.assertTrue(trial.user_attrs['fresh_R0'])
            self.assertEqual(request_trial(open_study(root, 'gamma0-id'), 8).number, 0)
            study.tell(trial, .08)
            self.assertFalse(request_trial(study, 8).user_attrs.get('incumbent', False))
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                open_study(root, 'other-objective')

    def test_gamma_zero_identity_is_mandatory_and_source_controls_stay_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, _, _, _ = configured(root)
            original = copy.deepcopy(cfg)
            with patch('independent_cnn_baseline.optuna_gamma0.PROJECT', root):
                candidate = trial_config(cfg, R0, 0, root)
                spec = {'objective': OBJECTIVE, 'seed': 42, 'search_space': SPACE, 'window_hours': 10,
                        'max_trials': 8, 'history_import': 'none', 'baseline_trial': 'fresh_R0_first',
                        'output_dir': 'runs/baseline/' + RUN_NAME}
                validate_spec(spec)
                for wrong in (6.4, .1):
                    changed = copy.deepcopy(spec)
                    changed['objective']['gamma'] = wrong
                    with self.assertRaisesRegex(ValueError, 'objective'):
                        validate_spec(changed)
            self.assertEqual(candidate['selection_objective'], OBJECTIVE)
            self.assertEqual(candidate['seed'], 42)
            self.assertEqual(candidate['input'], original['input'])
            self.assertEqual(candidate['augmentation'], original['augmentation'])
            self.assertEqual(candidate['training']['max_epochs'], original['training']['max_epochs'])
            self.assertEqual(cfg, original)

    def test_partial_pair_recovery_exactly_matches_fresh_gamma_zero_training(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            cfg, frames, pairs, cache = configured(root)
            study = open_study(root, 'identity')
            trial = request_trial(study, 8)
            folder = root / 'resumed'
            with patch('independent_cnn_baseline.engine.select_gamma', side_effect=AssertionError('No gamma sweep')), \
                 patch('independent_cnn_baseline.optuna_search.validation', side_effect=AssertionError('No calibrated objective')):
                with patch('independent_cnn_baseline.optuna_gamma0.boundary', side_effect=[None, SearchStop('window_complete')]):
                    with self.assertRaises(SearchStop):
                        run_trial(trial, cfg, folder, 'identity', frames, pairs, cache, root, float('inf'), lambda **kw: None, TinyNet)
                self.assertEqual(study.trials[0].intermediate_values, {})
                self.assertEqual(study.trials[0].state.name, 'RUNNING')
                result = run_trial(trial, cfg, folder, 'identity', frames, pairs, cache, root, float('inf'), lambda **kw: None, TinyNet)
                fresh_study = optuna.create_study(direction='maximize', pruner=optuna.pruners.NopPruner())
                expected_folder = root / 'fresh'
                expected = run_trial(fresh_study.ask(), cfg, expected_folder, 'identity', frames, pairs, cache, root, float('inf'), lambda **kw: None, TinyNet)
            self.assertEqual(result['best']['validation'], expected['best']['validation'])
            for target in ('crop', 'disease'):
                a = torch.load(folder / target / 'last.pt', weights_only=True)
                b = torch.load(expected_folder / target / 'last.pt', weights_only=True)
                for key in a['model_state']:
                    torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
            history = read_json(folder / 'history.json')
            self.assertTrue(all(row['gamma'] == 0. for row in history))
            values = dict(study.trials[0].intermediate_values)
            self.assertEqual(values, {r['epoch']: max(x['harmonic_mean_top1'] for x in history[:r['epoch']]) for r in history})
            self.assertEqual(result['best']['epoch'], max(history, key=lambda r: r['harmonic_mean_top1'])['epoch'])
            self.assertEqual(run_trial(trial, cfg, folder, 'identity', frames, pairs, cache, root, 0, lambda **kw: None, TinyNet), result)

    def test_pair_transaction_recovers_without_repeating_either_target_epoch(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            cfg, frames, pairs, cache = configured(root)
            cfg['training']['max_epochs'] = 1
            study = optuna.create_study(direction='maximize')
            trial = study.ask()
            folder = root / 'trial'
            with patch('independent_cnn_baseline.optuna_gamma0.validation_uncalibrated', side_effect=RuntimeError('pair interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'pair interrupted'):
                    run_trial(trial, cfg, folder, 'id', frames, pairs, cache, root, float('inf'), lambda **kw: None, TinyNet)
            with patch('independent_cnn_baseline.engine.Actor.epoch_step', side_effect=AssertionError('No duplicate training')):
                result = run_trial(trial, cfg, folder, 'id', frames, pairs, cache, root, float('inf'), lambda **kw: None, TinyNet)
            self.assertEqual(result['completed_paired_epochs'], 1)
            (folder / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                verify_terminal(folder, result['identity'], frames, pairs)

    def test_pruning_only_at_paired_epoch_six_and_uses_gamma_zero_reports(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            cfg, frames, pairs, cache = configured(root)
            cfg['training'].update(max_epochs=7, patience=8)
            study = optuna.create_study(direction='maximize')
            trial = study.ask()
            with patch.object(trial, 'should_prune', return_value=True) as pruner:
                result = run_trial(trial, cfg, root/'trial', 'id', frames, pairs, cache, root, float('inf'), lambda **kw: None, TinyNet)
            self.assertEqual(result['outcome'], 'PRUNED')
            self.assertEqual(result['reason'], 'median_pruner_gamma0')
            self.assertEqual(result['completed_paired_epochs'], 6)
            pruner.assert_called_once()
            self.assertEqual(len(study.trials[0].intermediate_values), 6)
            self.assertEqual(result['best']['validation']['gamma'], 0.)

    def test_no_eligible_winner_before_completion_and_ties_keep_earlier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            study = open_study(root, 'id')
            trial = request_trial(study, 8)
            trial.report(.99, 1)
            self.assertIsNone(snapshot(study, root, 'window_complete')['best_eligible'])
            study.tell(trial, .08)
            second = request_trial(study, 8)
            study.tell(second, .08)
            pruned = request_trial(study, 8)
            pruned.report(.9, 6)
            study.tell(pruned, state=optuna.trial.TrialState.PRUNED)
            summary = snapshot(study, root, 'window_complete')
            self.assertEqual(summary['best_eligible']['number'], 0)
            self.assertEqual(summary['historical_observations_imported'], 0)
            self.assertFalse(summary['official_test_inference'])

    def test_controller_cap_includes_fresh_r0_and_never_runs_official_tests(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            project = Path(temp)
            cfg, frames, pairs, cache = configured(project)
            cfg['runtime']['torch_threads'] = 2
            spec = {'objective': OBJECTIVE, 'seed': 42, 'search_space': SPACE, 'window_hours': 10,
                'max_trials': 8, 'history_import': 'none', 'baseline_trial': 'fresh_R0_first',
                'output_dir': 'runs/baseline/' + RUN_NAME, 'source_root': 'source', 'base_config': cfg}
            path = project/'spec.json'
            path.write_text(json.dumps(spec))
            calls = []
            def fake_trial(trial, *args):
                calls.append((trial.number, trial.params))
                return {'outcome': 'COMPLETE', 'best': {'validation': {'harmonic_mean_top1': .08}}}
            with patch('independent_cnn_baseline.optuna_gamma0.PROJECT', project), \
                 patch('independent_cnn_baseline.optuna_gamma0.audit_inputs', return_value=('id', frames, pairs, cache)), \
                 patch('independent_cnn_baseline.optuna_gamma0.run_trial', side_effect=fake_trial), \
                 patch('independent_cnn_baseline.optuna_gamma0.retire_recovery'), \
                 patch('independent_cnn_baseline.engine.select_gamma', side_effect=AssertionError('No gamma search')), \
                 patch('independent_cnn_baseline.runner.refit', side_effect=AssertionError('No refit')), \
                 patch('independent_cnn_baseline.runner.evaluate_final', side_effect=AssertionError('No official inference')):
                run(path)
            self.assertEqual([n for n, p in calls], list(range(8)))
            self.assertEqual(calls[0][1], R0)
            saved = read_json(project/spec['output_dir']/'selected_recipe.json')
            self.assertEqual(saved['objective'], OBJECTIVE)
            self.assertEqual(saved['number'], 0)


if __name__ == '__main__':
    unittest.main()
