"""History-aware deadline-only continuation; synthetic observations/images only."""
import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import optuna
import torch

from independent_cnn_baseline import optuna_gamma0 as base
from independent_cnn_baseline import optuna_gamma0_continue as more
from independent_cnn_baseline.engine import atomic_json,read_json,sha256
from independent_cnn_baseline.optuna_search import SearchStop,request_trial
from independent_cnn_baseline.tests.test_study import fixture,TinyNet

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(2)


def parent_study(parent):
    parent.mkdir()
    study=base.open_study(parent,'scientific')
    for number in range(8):
        trial=request_trial(study,8)
        value=.08 if number==0 else (.06 if number==1 else .04)
        trial.report(value,6)
        outcome=optuna.trial.TrialState.COMPLETE if number<2 else optuna.trial.TrialState.PRUNED
        study.tell(trial,value if number<2 else None,state=outcome)
        terminal={'objective':base.OBJECTIVE,'outcome':outcome.name,'best':{'epoch':6,
            'validation':{'gamma':0.,'harmonic_mean_top1':value},
            'weights':{'crop':'crop/best_epoch_06.pt','disease':'disease/best_epoch_06.pt'},
            'logits':'validation_epoch_06.npz'},'training_epoch_seconds':1.}
        folder=parent/'trials'/f'trial_{number:03d}'
        atomic_json(folder/'terminal.json',terminal)
        atomic_json(folder/'verification.json',{'terminal_sha256':sha256(folder/'terminal.json')})
    return study


class ContinuationTests(unittest.TestCase):
    def test_parent_copy_preserves_all_history_and_adds_trials_beyond_eight(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            p=Path(temp);parent=p/'parent';root=p/'new';root.mkdir()
            old=parent_study(parent);before=sha256(parent/'study.sqlite3')
            signature=more.parent_signature(parent)
            more.bootstrap(parent,root,before,'scientific')
            self.assertEqual(more.parent_signature(root),signature)
            study=base.open_study(root,'scientific')
            self.assertEqual([t.state.name for t in study.trials],['COMPLETE','COMPLETE']+['PRUNED']*6)
            trial=request_trial(study,len(study.trials)+1)
            self.assertEqual(trial.number,8)
            # Existing completed curves are available to the median pruner now.
            trial.report(.001,6)
            self.assertTrue(trial.should_prune())
            study.tell(trial,state=optuna.trial.TrialState.PRUNED)
            for number in range(9,13):
                trial=request_trial(study,len(study.trials)+1)
                self.assertEqual(trial.number,number)
                study.tell(trial,.07)
            self.assertEqual(sha256(parent/'study.sqlite3'),before)
            self.assertEqual(more.parent_signature(parent),signature)

    def test_rejects_parent_identity_or_unverified_copy(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            p=Path(temp);parent=p/'parent';root=p/'new';root.mkdir()
            old=parent_study(parent);before=sha256(parent/'study.sqlite3')
            with self.assertRaisesRegex(ValueError,'identity'):
                more.bootstrap(parent,root,before,'wrong')
            (root/'study.sqlite3').write_bytes(b'partial copy')
            with self.assertRaisesRegex(ValueError,'partial'):
                more.bootstrap(parent,root,before,'scientific')
            self.assertEqual(sha256(parent/'study.sqlite3'),before)

    def test_pending_high_report_cannot_replace_parent_and_ties_keep_parent(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            p=Path(temp);parent=p/'parent';root=p/'new';root.mkdir()
            old=parent_study(parent);more.bootstrap(parent,root,sha256(parent/'study.sqlite3'),'scientific')
            study=base.open_study(root,'scientific');trial=request_trial(study,len(study.trials)+1)
            trial.report(.99,1)
            summary=more.snapshot(study,root,parent,'window_complete')
            self.assertEqual(summary['best_eligible']['number'],0)
            self.assertEqual(summary['additional_pending'],1)
            self.assertIsNone(summary['trial_limit'])
            source=read_json(parent/'trials/trial_000/terminal.json')
            atomic_json(root/'trials/trial_008/terminal.json',source)
            atomic_json(root/'trials/trial_008/verification.json',{'terminal_sha256':sha256(root/'trials/trial_008/terminal.json')})
            # Recovery after terminal write but before SQLite tell stays pending.
            provisional=more.snapshot(study,root,parent,'running')
            self.assertEqual(provisional['best_eligible']['number'],0)
            self.assertEqual(provisional['additional_pending'],1)
            study.tell(trial,.08)
            summary=more.snapshot(study,root,parent,'window_complete')
            self.assertEqual(summary['best_eligible']['number'],0)
            self.assertEqual(summary['next_sampler_phase'],'TPE')
            self.assertTrue((root/summary['best_eligible']['terminal']).exists())

    def test_additional_trial_mid_pair_cap_and_exact_recovery(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            p=Path(temp);parent=p/'parent';root=p/'new';root.mkdir()
            old=parent_study(parent);before=sha256(parent/'study.sqlite3')
            more.bootstrap(parent,root,before,'scientific')
            study=base.open_study(root,'scientific');trial=request_trial(study,len(study.trials)+1)
            cfg,frames,pairs,cache=fixture(p)
            cfg['selection_objective']=base.OBJECTIVE
            cfg['selection_continuation']={'identity':'controller','parent_trials':8}
            folder=root/'trials/trial_008'
            with patch.object(base,'boundary',side_effect=[None,SearchStop('window_complete')]):
                with self.assertRaises(SearchStop):
                    base.run_trial(trial,cfg,folder,'scientific',frames,pairs,cache,root,float('inf'),lambda **kw:None,TinyNet)
            self.assertEqual(torch.load(folder/'crop/last.pt',weights_only=True)['epoch'],1)
            self.assertEqual(study.trials[8].state.name,'RUNNING')
            resumed=request_trial(study,len(study.trials)+1)
            self.assertEqual(resumed.number,8)
            result=base.run_trial(resumed,cfg,folder,'scientific',frames,pairs,cache,root,float('inf'),lambda **kw:None,TinyNet)
            self.assertEqual(result['completed_paired_epochs'],2)
            for target in ['crop','disease']:
                self.assertEqual([x['epoch'] for x in read_json(folder/target/'history.json')],[1,2])
            self.assertEqual(sha256(parent/'study.sqlite3'),before)

    def test_new_user_pause_is_not_cleared_or_audited(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            p=Path(temp);root=p/'new';root.mkdir();parent=p/'parent';parent.mkdir()
            (root/'STOP.request').write_text('user pause')
            spec=p/'spec.json';atomic_json(spec,{})
            original={'source_root':'source','base_config':{'runtime':{'torch_threads':2}}}
            with patch.object(more,'PROJECT',p),patch.object(more,'validate_config',return_value=(root,parent,original)), \
                 patch.object(more,'audit',side_effect=AssertionError('Paused runs must not audit/start')):
                more.run(spec)
            self.assertTrue((root/'STOP.request').exists())
            self.assertFalse((root/'runner.pid').exists())


if __name__=='__main__':unittest.main()
