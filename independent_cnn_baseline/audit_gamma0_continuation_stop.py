"""CPU-only verification of the authorized five-hour continuation's cap stop."""
from __future__ import annotations

import contextlib
import gc
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import optuna
import pandas as pd
import torch

from dpd.data import COLUMNS
from .audit_gamma0_completion import finite, independent_metrics
from .engine import StudyLock, atomic_json, fingerprint, lr_factor, read_json, sha256
from .extend_gamma0_window import process_exited
from .optuna_gamma0 import OBJECTIVE, PROJECT, trial_config, verify_terminal
from .optuna_gamma0_continue import RUN_NAME, parent_signature, validate_config
from .optuna_search import distributions


def state_finite(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(state_finite(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(state_finite(x) for x in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def main():
    torch.set_num_threads(2)
    root = PROJECT / 'runs/baseline' / RUN_NAME
    spec = read_json(root / 'continuation_config.json')
    checked_root, parent, original = validate_config(spec)
    assert checked_root == root
    status = read_json(root / 'status.json')
    launch = read_json(root / 'launch.json')
    assert status['state'] == 'window_complete' and status['objective'] == OBJECTIVE
    assert status['window_hours'] == 5 and status['trial_limit'] is None
    assert not status['error'] and not (root / 'STOP.request').exists()
    start = datetime.fromisoformat(status['window_started_at_utc'])
    deadline = datetime.fromisoformat(status['window_deadline_at_utc'])
    end = datetime.fromisoformat(status['stopped_at_utc'])
    assert (deadline-start).total_seconds() == 18000 and end >= deadline
    pids = [status['pid'], launch['launcher_pid']]
    assert all(process_exited(pid) for pid in pids), 'Current controller/launcher has not positively exited'
    locks = [root, parent, PROJECT/original['source_root'],
             PROJECT/'runs/baseline/independent_cnn_baseline_r0_finalization_seed42',
             PROJECT/'runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42',
             PROJECT/'runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42']
    with contextlib.ExitStack() as stack:
        for folder in locks:
            stack.enter_context(StudyLock(folder/'runner.lock'))
        identity = read_json(root/'identity.json')
        scientific = fingerprint(identity)
        continuation = read_json(root/'continuation_identity.json')
        assert scientific == spec['parent_scientific_identity'] == status['scientific_identity']
        assert continuation['scientific_identity'] == scientific and continuation['config'] == spec
        assert fingerprint(continuation) == status['continuation_identity']
        assert spec == read_json(PROJECT/'independent_cnn_baseline/configs/optuna_convnext_base_gamma0_continue_20261102.json')
        source = PROJECT / original['source_root']
        frames = {n: pd.read_csv(source/f'{n}.csv') for n in
                  ('fit','validation_seen','validation_unseen','selection_excluded','all_train')}
        assert {n:len(f) for n,f in frames.items()} == dict(fit=5798,validation_seen=862,validation_unseen=533,selection_excluded=807,all_train=8000)
        assert set(frames['fit'].pair_id).isdisjoint(frames['validation_unseen'].pair_id)
        pairs = pd.DataFrame(read_json(source/'pairs.json'))
        assert pairs.pair_id.tolist() == list(range(333)) and int(pairs.is_unseen.sum()) == 10
        preservation = read_json(parent/'historical_preservation.json')
        pins = {str((parent/p).resolve()):h for p,h in spec['parent_sha256'].items()}
        pins.update({str((PROJECT/p).resolve()):h for p,h in preservation['historical_sha256'].items()})
        pins.update({str((source/p).resolve()):h for p,h in original['source_sha256'].items()})
        pins.update({str((PROJECT/p).resolve()):h for p,h in identity['code_sha256'].items()})
        pins.update({str((PROJECT/p).resolve()):h for p,h in continuation['continuation_code_sha256'].items()})
        model = original['base_config']['model']
        pins[str((PROJECT/model['pretrained_file']).resolve())] = model['pretrained_sha256']
        pins[str((PROJECT/original['cache_identity_file']).resolve())] = original['cache_identity_sha256']
        cache = read_json(PROJECT/original['cache_identity_file'])
        assert cache['cache_sha256'] == identity['cache_sha256']
        provenance = read_json(source/'data_provenance.json')
        data = (PROJECT/original['data_root']).resolve()
        metadata = data/'DPD_Small_train.csv'
        pins[str(metadata)] = provenance['metadata']['train']['sha256']
        assert pd.read_csv(metadata,header=None,names=COLUMNS)[COLUMNS].to_dict('records') == frames['all_train'][COLUMNS].to_dict('records')
        for name in frames['all_train'].file:
            pins[str(data/'DPD_Small_dataset'/name)] = provenance['image_hashes'][name]
            if cache['cache_sha256'][name] is not None:
                pins[str(PROJECT/'runs/cache/resized_256'/(str(name).rsplit('.',1)[0]+'.jpg'))] = cache['cache_sha256'][name]
        def check(item):
            path,digest=item
            assert sha256(path) == digest, f'Changed pinned input: {path}'
        print(f'Verifying {len(pins)} pinned paths; six study locks acquired.',flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check,pins.items()))
        inherited, current = parent_signature(parent), parent_signature(root)
        for table, rows in inherited.items():
            ids = {r[0] for r in rows}
            assert [r for r in current[table] if r[0] in ids] == rows, f'Inherited table changed: {table}'
        print('Parent observations and all input/preservation hashes passed.',flush=True)
        db_hash = sha256(root/'study.sqlite3')
        results=[]; complete_curves=[]; artifacts={}; partial=[]
        with contextlib.closing(sqlite3.connect((root/'study.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)) as db:
            db.row_factory=sqlite3.Row
            rows=[dict(x) for x in db.execute('SELECT * FROM trials ORDER BY number')]
            assert [r['number'] for r in rows] == list(range(14))
            attrs={r['key']:json.loads(r['value_json']) for r in db.execute('SELECT * FROM study_user_attributes')}
            assert attrs['objective'] == OBJECTIVE and attrs['study_identity'] == scientific
            assert attrs['continuation_identity'] == status['continuation_identity']
            for row in rows:
                number,tid=row['number'],row['trial_id']
                folder=(parent if number<8 else root)/'trials'/f'trial_{number:03d}'
                ident=read_json(folder/'identity.json'); params=ident['params']
                config=read_json(folder/'config.json')
                expected=trial_config(original['base_config'],params,number,parent if number<8 else root)
                if number>=8:expected['selection_continuation']={'identity':status['continuation_identity'],'parent_trials':8}
                assert config==expected and config['selection_objective']==OBJECTIVE
                trial_identity=fingerprint({'config':config,'study':scientific})
                assert ident['identity']==trial_identity and ident['study_identity']==scientific and ident['objective']==OBJECTIVE
                sqlparams={}
                for p in db.execute('SELECT * FROM trial_params WHERE trial_id=?',(tid,)):
                    dist=json.loads(p['distribution_json'])
                    sqlparams[p['param_name']]=dist['attributes']['choices'][int(p['param_value'])] if dist['name']=='CategoricalDistribution' else p['param_value']
                assert sqlparams==params
                if number:
                    assert len(complete_curves)<3
                    for index,(name,dist) in enumerate(distributions().items()):
                        seed=int(np.random.SeedSequence([42,number,index]).generate_state(1)[0])
                        assert optuna.samplers.RandomSampler(seed=seed).sample_independent(None,None,name,dist)==params[name]
                pair=read_json(folder/'pair_state.json'); history=read_json(folder/'history.json')
                assert pair['history']==history and finite(history)
                assert [x['epoch'] for x in history]==list(range(1,len(history)+1))
                assert all(x['gamma']==0 and x['objective_name']==OBJECTIVE['name'] for x in history)
                reports={r['step']:r['intermediate_value'] for r in db.execute('SELECT * FROM trial_intermediate_values WHERE trial_id=?',(tid,))}
                assert reports=={x['epoch']:max(y['harmonic_mean_top1'] for y in history[:x['epoch']]) for x in history}
                best=max(history,key=lambda x:x['harmonic_mean_top1'])
                assert pair['best']['epoch']==best['epoch'] and pair['best']['validation']==best
                seconds=0.; heads={}; target_epochs={}
                for target in ('crop','disease'):
                    h=read_json(folder/target/'history.json')
                    assert finite(h) and [x['epoch'] for x in h]==list(range(1,len(h)+1))
                    assert len(history)<=len(h)<=len(history)+1
                    for x in h:
                        assert math.isclose(x['lr_backbone'],params['lr_backbone']*lr_factor(x['epoch'],config['training']),rel_tol=1e-14)
                        assert math.isclose(x['lr_head'],10*x['lr_backbone'],rel_tol=1e-14)
                    seconds+=sum(x['seconds'] for x in h)
                    heads[target]={'selected_epoch':h[best['epoch']-1],'last_epoch':h[-1]}
                    target_epochs[target]=len(h)
                best_paths=pair['best']
                with np.load(folder/best_paths['logits']) as z:
                    assert np.isfinite(z['crop']).all() and np.isfinite(z['disease']).all()
                    metrics=independent_metrics(z['crop'],z['disease'],frames,pairs)
                for k in (1,5):
                    assert metrics[f'seen_top{k}']==best[f'seen_top{k}']
                    assert metrics[f'pseudo_unseen_top{k}']==best[f'unseen_top{k}']
                    assert metrics[f'harmonic_mean_top{k}']==best[f'harmonic_mean_top{k}']
                checks=[]; stop=None
                for epoch in range(1,len(history)+1):
                    epoch_best=max(history[:epoch],key=lambda x:x['harmonic_mean_top1'])
                    if epoch==30:stop=(epoch,'COMPLETE','max_epochs')
                    elif epoch-epoch_best['epoch']>=8:stop=(epoch,'COMPLETE','early_stopping_gamma0')
                    elif epoch>=6 and (epoch-6)%3==0 and len(complete_curves)>=2:
                        values=[c[epoch] for c in complete_curves if epoch in c]
                        if values:
                            median=float(np.median(values))
                            checks.append({'epoch':epoch,'best_H0':reports[epoch],'completed_median':median})
                            if reports[epoch]<median:stop=(epoch,'PRUNED','median_pruner_gamma0')
                    if stop:break
                if row['state'] in ('COMPLETE','PRUNED'):
                    terminal=verify_terminal(folder,trial_identity,frames,pairs)
                    assert terminal['outcome']==row['state'] and stop==(len(history),terminal['outcome'],terminal['reason'])
                    verification=read_json(folder/'verification.json')
                    assert verification['terminal_sha256']==sha256(folder/'terminal.json') and verification['objective']==OBJECTIVE
                    assert db.execute('SELECT value FROM trial_values WHERE trial_id=?',(tid,)).fetchone()[0]==best['harmonic_mean_top1']
                    assert all(n==len(history) for n in target_epochs.values())
                    assert math.isclose(seconds,terminal['training_epoch_seconds'],abs_tol=1e-8)
                    if number>=8:
                        artifacts.update({str((folder/p).relative_to(root)):v for p,v in terminal['artifact_sha256'].items()})
                        for p in ('terminal.json','verification.json'):artifacts[str((folder/p).relative_to(root))]=sha256(folder/p)
                        assert not list(folder.glob('*/last.pt'))
                    reason=terminal['reason']
                else:
                    assert row['state']=='RUNNING' and stop is None and number==13
                    assert not (folder/'terminal.json').exists()
                    assert db.execute('SELECT value FROM trial_values WHERE trial_id=?',(tid,)).fetchone() is None
                    reason='window_complete_active_target_checkpointed'
                    for target in ('crop','disease'):
                        last_path=folder/target/'last.pt'
                        saved=torch.load(last_path,map_location='cpu',weights_only=True)
                        assert saved['fingerprint']==trial_identity and saved['target']==target
                        assert saved['epoch']==target_epochs[target] and saved['history']==read_json(folder/target/'history.json')
                        assert saved['scheduler_state']==dict(epoch=saved['epoch'],horizon=30,factor=lr_factor(saved['epoch'],config['training']))
                        assert saved['optimizer_state']['state'] and saved['scaler_state']
                        assert saved['generator_state'].numel()>0 and saved['rng_state']['cuda']
                        assert state_finite(saved)
                        assert list(saved['val_logits'].shape)==[1395,55 if target=='crop' else 175]
                        partial.append({'target':target,'epoch':saved['epoch'],'sha256':sha256(last_path),
                                        'scheduler':saved['scheduler_state'],'optimizer_scaler_rng_loader_preserved':True})
                        del saved;gc.collect()
                        selected=torch.load(folder/best_paths['weights'][target],map_location='cpu',weights_only=True)
                        assert selected['fingerprint']==trial_identity and selected['epoch']==best['epoch'] and state_finite(selected)
                        del selected;gc.collect()
                    for path in folder.rglob('*'):
                        if path.is_file():artifacts[str(path.relative_to(root))]=sha256(path)
                results.append({'number':number,'state':row['state'],'reason':reason,'params':params,
                    'selected_epoch':best['epoch'],'paired_epochs':len(history),'target_epochs':target_epochs,
                    'target_epoch_seconds':seconds,'validation':metrics,'head_history':heads,'pruning_checks':checks})
                if row['state']=='COMPLETE':complete_curves.append(reports)
                print(f'Trial {number}: {row["state"]}, paired {len(history)}, best H0 {best["harmonic_mean_top1"]:.8f}; verified.',flush=True)
        assert sha256(root/'study.sqlite3')==db_hash
        summary=read_json(root/'study_summary.json'); selected=read_json(root/'selected_recipe_for_window.json')
        winner=max([r for r in results if r['state']=='COMPLETE'],key=lambda r:r['validation']['harmonic_mean_top1'])
        assert winner['number']==selected['number']==summary['best_eligible']['number']==0
        assert selected=={'objective':OBJECTIVE,**summary['best_eligible']}
        assert (summary['additional_complete'],summary['additional_pruned'],summary['additional_pending'])==(0,5,1)
        assert summary['state']=='window_complete' and summary['next_sampler_phase']=='startup_random'
        assert len(summary['trials'])==14 and summary['calibrated_observations_imported']==0
        assert not list(root.rglob('predictions_*.csv')) and not list(root.rglob('scores.npz'))
        assert not status['official_test_inference'] and not status['automatic_final_refit']
        new=[r for r in results if r['number']>=8]
        runtime={'target_epoch_seconds':sum(r['target_epoch_seconds'] for r in new),
                 'paired_epochs':sum(r['paired_epochs'] for r in new),
                 'target_epochs':sum(sum(r['target_epochs'].values()) for r in new),
                 'post_audit_wall_seconds':(end-start).total_seconds(),
                 'deadline_checkpoint_overrun_seconds':(end-deadline).total_seconds(),
                 'started_at_utc':start.isoformat(),'stopped_at_utc':end.isoformat(),
                 'parent_target_epoch_seconds':spec['parent_target_epoch_seconds'],
                 'parent_wall_seconds':spec['parent_wall_seconds'],
                 'previous_calibrated_window_and_all_refits_excluded':True}
        audit={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'objective':OBJECTIVE,
               'scientific_identity':scientific,'continuation_identity':status['continuation_identity'],
               'positive_process_exit':pids,'all_six_OS_locks_verified_free':True,
               'inherited_complete':2,'inherited_pruned':6,'new_complete':0,'new_pruned':5,'new_pending':1,
               'selected_trial':0,'TPE_activated':False,'TPE_complete_threshold':3,
               'parent_files_verified':len(spec['parent_sha256']),
               'historical_files_verified':len(preservation['historical_sha256']),
               'historical_final_artifacts_verified':preservation['existing_final_artifacts_verified'],
               'source_files':len(original['source_sha256']),'original_images':8000,'cache_images':7193,
               'pinned_inputs_sha256':pins,'new_artifact_sha256':artifacts,'sqlite_sha256':db_hash,
               'all_terminal_outcomes_params_streams_reports_H0_top5_and_ties_verified':True,
               'all_selected_weights_and_logits_finite':True,'schedule_horizon':30,
               'pending_trial':13,'pending_pair_epochs':2,'pending_target_checkpoints':partial,
               'next_target':'disease','next_epoch':3,'pending_trial_eligible':False,
               'calibrated_observations_imported':0,'official_test_inference':False,'full_data_refit':False,
               'runtime':runtime,'trials':results,'audit_script_sha256':sha256(__file__)}
        audit['summary_artifact_sha256']={p:sha256(root/p) for p in
             ('status.json','study_summary.json','results.md','selected_recipe_for_window.json','identity.json',
              'continuation_identity.json','continuation_config.json','history_import_verified.json','training.log',
              'launch.json')}
        atomic_json(root/'window_stop_verified_20261102.json',audit)
        atomic_json(root/'validation_metrics_verified_20261102.json',{'objective':OBJECTIVE,'selected_trial':0,
                    'trials':[{k:r[k] for k in ('number','state','params','selected_epoch','validation')} for r in results]})
    print(json.dumps({'audit_passed':True,'runtime':runtime,'new_complete':0,'new_pruned':5,'new_pending':1,
                      'selected_trial':0,'next_target':'disease','next_epoch':3}),flush=True)


if __name__=='__main__':
    main()
