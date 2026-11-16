"""Read-only CPU verification of the completed gamma-zero selection artifacts."""
from __future__ import annotations

import contextlib
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from dpd.data import COLUMNS
from .engine import StudyLock, atomic_json, fingerprint, lr_factor, read_json, sha256
from .extend_gamma0_window import process_exited
from .optuna_gamma0 import (OBJECTIVE, PROJECT, R0, trial_config, validate_spec,
                            verify_terminal)
from .optuna_search import distributions

ROOT = PROJECT / "runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42"


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def harmonic(a, b):
    return 2 * a * b / (a + b) if a + b else 0.


def independent_metrics(crop, disease, frames, pairs):
    """Direct stable ranking of all 323 training pairs; no calibration routine."""
    frame = pd.concat([frames['validation_seen'], frames['validation_unseen']], ignore_index=True)
    nc = len(frames['validation_seen'])
    cp = torch.log_softmax(torch.from_numpy(crop), 1).numpy()
    dp = torch.log_softmax(torch.from_numpy(disease), 1).numpy()
    assert np.isfinite(cp).all() and np.isfinite(dp).all()
    scores = cp[:, pairs.crop_id.to_numpy()] + dp[:, pairs.disease_id.to_numpy()]
    candidates = pairs.loc[~pairs.is_unseen, 'pair_id'].to_numpy()
    assert len(candidates) == 323
    ranked = candidates[np.argsort(-scores[:, candidates], axis=1, kind='stable')]
    labels = frame.pair_id.to_numpy()
    results = {'gamma': 0., 'candidate_count': 323, 'seen_count': nc, 'pseudo_unseen_count': len(frame) - nc}
    for k in (1, 5):
        hit = (ranked[:, :k] == labels[:, None]).any(1)
        a, b = float(hit[:nc].mean()), float(hit[nc:].mean())
        results.update({f'seen_top{k}': a, f'pseudo_unseen_top{k}': b, f'harmonic_mean_top{k}': harmonic(a, b)})
    raw, composed = {}, {}
    for part, sl in [('seen_validation', slice(None, nc)), ('pseudo_unseen_validation', slice(nc, None))]:
        raw[part] = {}; composed[part] = {}
        for target, logits in [('crop', crop), ('disease', disease)]:
            truth = frame[f'{target}_id'].to_numpy()
            raw[part][target + '_top1'] = float((logits.argmax(1)[sl] == truth[sl]).mean())
            predicted = pairs[f'{target}_id'].to_numpy()[ranked[:, 0]]
            composed[part][target + '_top1'] = float((predicted[sl] == truth[sl]).mean())
    pu_candidates = np.sort(frames['validation_unseen'].pair_id.unique())
    assert len(pu_candidates) == 20
    pu_rank = pu_candidates[np.argsort(-scores[nc:, pu_candidates], axis=1, kind='stable')]
    results['raw_independent_factors'] = raw
    results['factors_of_winning_composition'] = composed
    results['pseudo_unseen_only_20_candidates_diagnostic'] = {
        f'top{k}': float((pu_rank[:, :k] == labels[nc:, None]).any(1).mean()) for k in (1, 5)}
    return results


def main():
    torch.set_num_threads(2)
    state = read_json(ROOT / 'status.json')
    assert state['state'] == 'selection_complete' and state['objective'] == OBJECTIVE
    assert state['max_trials'] == 8 and state['historical_observations_imported'] == 0
    assert not state['official_test_inference'] and not state['automatic_final_refit']
    assert not (ROOT / 'STOP.request').exists()
    pids = [37564, 32572, 41184, 32752, 39524, 35888]
    assert all(process_exited(pid) for pid in pids)
    spec = read_json(ROOT / 'study_config.json')
    assert spec == read_json(PROJECT / 'independent_cnn_baseline/configs/optuna_convnext_base_gamma0_seed42.json')
    assert validate_spec(spec) == ROOT
    auth = read_json(ROOT / 'extension_authorization_20261101.json')
    assert state['window_hours'] == 20 and state['window_deadline_at_utc'] == auth['new_deadline_at_utc']
    assert len(state['budget_extensions']) == 1 and state['window_started_at_utc'] == auth['window_started_at_utc']
    immutable = read_json(ROOT / 'identity.json')
    assert fingerprint(immutable) == state['audit_identity']
    assert sha256(ROOT / 'identity.json') == auth['identity_file_sha256']
    assert sha256(ROOT / 'study_config.json') == auth['study_config_sha256']
    original_status = read_json(ROOT / 'extension_previous_status_20261101.json')
    applied = read_json(ROOT / 'extension_applied_20261101.json')
    assert sha256(ROOT / 'extension_previous_status_20261101.json') == applied['before_status_sha256']
    source = PROJECT / spec['source_root']
    frames = {name: pd.read_csv(source / f'{name}.csv') for name in
              ['fit', 'validation_seen', 'validation_unseen', 'selection_excluded', 'all_train']}
    assert {k: len(v) for k,v in frames.items()} == {'fit':5798, 'validation_seen':862,
        'validation_unseen':533, 'selection_excluded':807, 'all_train':8000}
    pairs = pd.DataFrame(read_json(source / 'pairs.json'))
    assert pairs.pair_id.tolist() == list(range(333)) and int(pairs.is_unseen.sum()) == 10
    assert set(frames['fit'].pair_id).isdisjoint(frames['validation_unseen'].pair_id)
    with contextlib.ExitStack() as locks:
        for folder in auth['lock_roots']:
            locks.enter_context(StudyLock(PROJECT / folder / 'runner.lock'))
        locks.enter_context(StudyLock(ROOT / 'extension_supervisor.lock'))
        source_hashes = {str((source / p).resolve()): h for p,h in spec['source_sha256'].items()}
        source_hashes.update({str((PROJECT / p).resolve()): h for p,h in immutable['code_sha256'].items()})
        model = spec['base_config']['model']
        assert model['pretrained_revision'] == '67758fcbcf9f007ec7630bf8636b74a8f24e52a0'
        source_hashes[str((PROJECT/model['pretrained_file']).resolve())] = model['pretrained_sha256']
        preservation = read_json(ROOT / 'historical_preservation.json')
        historical = {str((PROJECT / p).resolve()): h for p,h in preservation['historical_sha256'].items()}
        source_hashes.update(historical)
        assert sha256(PROJECT / spec['cache_identity_file']) == spec['cache_identity_sha256']
        pin = read_json(PROJECT / spec['cache_identity_file'])
        assert immutable['cache_sha256'] == pin['cache_sha256']
        provenance = read_json(source / 'data_provenance.json')
        data_root = (PROJECT / spec['data_root']).resolve()
        metadata = data_root / 'DPD_Small_train.csv'
        source_hashes[str(metadata)] = provenance['metadata']['train']['sha256']
        assert pd.read_csv(metadata, header=None, names=COLUMNS)[COLUMNS].to_dict('records') == frames['all_train'][COLUMNS].to_dict('records')
        for f in frames['all_train'].file:
            source_hashes[str(data_root/'DPD_Small_dataset'/f)] = provenance['image_hashes'][f]
            if pin['cache_sha256'][f] is not None:
                source_hashes[str(PROJECT/'runs/cache/resized_256'/(Path(f).stem+'.jpg'))] = pin['cache_sha256'][f]
        print(f'Checking {len(source_hashes)} pinned source/data/cache/historical paths...', flush=True)
        def check(item):
            p,h=item
            assert sha256(p) == h, f'Hash mismatch: {p}'
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check, source_hashes.items()))
        print('Pinned inputs and historical preservation passed.', flush=True)
        db_hash = sha256(ROOT / 'study.sqlite3')
        with contextlib.closing(sqlite3.connect((ROOT/'study.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            sqlite_trials = [dict(x) for x in db.execute('SELECT * FROM trials ORDER BY number')]
            assert [x['number'] for x in sqlite_trials] == list(range(8))
            attrs = {x['key']: json.loads(x['value_json']) for x in db.execute('SELECT * FROM study_user_attributes')}
            assert attrs['study_identity'] == state['audit_identity'] and attrs['objective'] == OBJECTIVE
            results = []; completed_curves = []; manifest = {}
            for trial in sqlite_trials:
                number, tid = trial['number'], trial['trial_id']
                folder = ROOT/'trials'/f'trial_{number:03d}'
                config = read_json(folder/'config.json')
                ident = read_json(folder/'identity.json')
                params = ident['params']
                assert config == trial_config(spec['base_config'], params, number, ROOT)
                assert ident['objective'] == OBJECTIVE and ident['study_identity'] == state['audit_identity']
                identity = fingerprint({'config':config, 'study':state['audit_identity']})
                assert ident['identity'] == identity
                terminal = verify_terminal(folder, identity, frames, pairs)
                verified = read_json(folder/'verification.json')
                assert verified['terminal_sha256'] == sha256(folder/'terminal.json') and verified['objective'] == OBJECTIVE
                assert terminal['outcome'] == trial['state']
                sqlparams = {}
                for p in db.execute('SELECT * FROM trial_params WHERE trial_id=?', (tid,)):
                    dist = json.loads(p['distribution_json'])
                    sqlparams[p['param_name']] = dist['attributes']['choices'][int(p['param_value'])] if dist['name']=='CategoricalDistribution' else p['param_value']
                assert sqlparams == params
                ta = {x['key']:json.loads(x['value_json']) for x in db.execute('SELECT * FROM trial_user_attributes WHERE trial_id=?', (tid,))}
                assert not ta.get('incumbent') and bool(ta.get('fresh_R0')) == (number==0)
                if number == 0:
                    assert params == R0
                else:
                    assert len(completed_curves) < 3
                    for index, (name, dist) in enumerate(distributions().items()):
                        seed = int(np.random.SeedSequence([42, number, index]).generate_state(1)[0])
                        assert optuna.samplers.RandomSampler(seed=seed).sample_independent(None,None,name,dist) == params[name]
                history = read_json(folder/'history.json'); pair = read_json(folder/'pair_state.json')
                assert history == pair['history'] and finite(history)
                assert [x['epoch'] for x in history] == list(range(1,len(history)+1))
                reports = {x['step']:x['intermediate_value'] for x in db.execute('SELECT * FROM trial_intermediate_values WHERE trial_id=?',(tid,))}
                assert reports == {x['epoch']:max(y['harmonic_mean_top1'] for y in history[:x['epoch']]) for x in history}
                assert db.execute('SELECT value FROM trial_values WHERE trial_id=?',(tid,)).fetchone()[0] == terminal['best']['validation']['harmonic_mean_top1']
                stop = None; prune_checks = []
                for epoch in range(1,len(history)+1):
                    best = max(history[:epoch], key=lambda x:x['harmonic_mean_top1'])
                    if epoch == 30:
                        stop=(epoch,'COMPLETE','max_epochs')
                    elif epoch-best['epoch'] >= 8:
                        stop=(epoch,'COMPLETE','early_stopping_gamma0')
                    elif epoch >= 6 and (epoch-6)%3==0 and len(completed_curves)>=2:
                        comparable = [h[epoch] for h in completed_curves if epoch in h]
                        if comparable:
                            median = float(np.median(comparable))
                            prune_checks.append({'epoch':epoch,'best_H0':reports[epoch],'completed_median':median})
                            if reports[epoch] < median:
                                stop=(epoch,'PRUNED','median_pruner_gamma0')
                    if stop:break
                assert stop == (len(history),terminal['outcome'],terminal['reason'])
                heads = {}; seconds = 0.
                for target in ('crop','disease'):
                    h = read_json(folder/target/'history.json')
                    assert finite(h) and [x['epoch'] for x in h] == list(range(1,len(history)+1))
                    for row in h:
                        assert math.isclose(row['lr_backbone'], params['lr_backbone']*lr_factor(row['epoch'],config['training']),rel_tol=1e-14)
                        assert math.isclose(row['lr_head'],10*row['lr_backbone'],rel_tol=1e-14)
                    seconds += sum(x['seconds'] for x in h)
                    heads[target]={'selected_epoch':h[terminal['best']['epoch']-1], 'last_epoch':h[-1]}
                assert math.isclose(seconds,terminal['training_epoch_seconds'],abs_tol=1e-8)
                with np.load(folder/terminal['best']['logits']) as z:
                    metric = independent_metrics(z['crop'], z['disease'],frames,pairs)
                saved = terminal['best']['validation']
                for k in (1,5):
                    assert metric[f'seen_top{k}'] == saved[f'seen_top{k}']
                    assert metric[f'pseudo_unseen_top{k}'] == saved[f'unseen_top{k}']
                    assert metric[f'harmonic_mean_top{k}'] == saved[f'harmonic_mean_top{k}']
                manifest.update({str((folder/p).relative_to(ROOT)):h for p,h in terminal['artifact_sha256'].items()})
                for name in ['terminal.json','verification.json']:
                    manifest[str((folder/name).relative_to(ROOT))] = sha256(folder/name)
                results.append({'number':number,'outcome':terminal['outcome'],'reason':terminal['reason'],
                    'params':params,'selected_epoch':terminal['best']['epoch'],'paired_epochs':len(history),
                    'target_epoch_seconds':seconds,'validation':metric,'head_history':heads,'pruning_checks':prune_checks})
                if terminal['outcome']=='COMPLETE':completed_curves.append(reports)
                print(f'Trial {number}: {terminal["outcome"]}, epoch {terminal["best"]["epoch"]}, H0 {saved["harmonic_mean_top1"]:.8f}; verified.',flush=True)
        assert sha256(ROOT/'study.sqlite3') == db_hash
        summary=read_json(ROOT/'study_summary.json'); selected=read_json(ROOT/'selected_recipe.json')
        winner=max([x for x in results if x['outcome']=='COMPLETE'],key=lambda x:x['validation']['harmonic_mean_top1'])
        assert winner['number']==selected['number']==summary['best_eligible']['number']==0
        assert selected=={'objective':OBJECTIVE,**summary['best_eligible']}
        assert summary['search_complete'] and len(summary['trials'])==8
        assert not list(ROOT.rglob('predictions_*.csv')) and not list(ROOT.rglob('scores.npz'))
        assert not list(ROOT.rglob('last.pt'))
        end=datetime.fromisoformat(state['finished_at_utc']);start=datetime.fromisoformat(state['window_started_at_utc'])
        first_stop=datetime.fromisoformat(original_status['stopped_at_utc'])
        resume=datetime.fromisoformat(state['resumed_at_utc'])
        prior_runtime=read_json(PROJECT/'runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42/window_stop_verified_20261101.json')
        runtime={'target_epoch_seconds':sum(x['target_epoch_seconds'] for x in results),
            'paired_epochs':sum(x['paired_epochs'] for x in results),'target_epochs':2*sum(x['paired_epochs'] for x in results),
            'wall_seconds_from_postaudit_start':(end-start).total_seconds(),
            'first_segment_wall_seconds':(first_stop-start).total_seconds(),
            'continuation_wall_seconds':(end-resume).total_seconds(),
            'handover_and_reaudit_seconds':(resume-first_stop).total_seconds(),
            'finished_at_utc':end.isoformat(),'unused_authorized_seconds':(datetime.fromisoformat(state['window_deadline_at_utc'])-end).total_seconds(),
            'prior_calibrated_window_target_epoch_seconds':prior_runtime['training_epoch_seconds_new'],
            'prior_calibrated_window_wall_seconds':prior_runtime['post_audit_window_wall_seconds'],
            'prior_selection_refits_excluded_from_new_runtime':True}
        audit={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'study_identity':state['audit_identity'],
            'objective':OBJECTIVE,'fresh_trials':8,'complete':2,'pruned':6,'pending':0,'selected_trial':0,
            'positive_process_exit':pids,'all_five_study_locks_and_supervisor_lock_verified_free':True,
            'pinned_source_files':14,'pinned_code_files':len(immutable['code_sha256']),
            'pretrained_sha256':model['pretrained_sha256'],'original_images':8000,
            'selection_cache_images':sum(x is not None for x in pin['cache_sha256'].values()),
            'preserved_historical_files':len(historical),'preserved_final_artifacts':preservation['existing_final_artifacts_verified'],
            'terminal_and_verification_and_artifact_sha256':manifest,'pinned_inputs_sha256':source_hashes,
            'all_retained_selected_weights_logits_metrics_finite':True,'all_trials_H0_recomputed_from_logits':True,
            'all_sqlite_terminal_params_reports_outcomes_match':True,'all_earlystop_prune_reasons_verified':True,
            'all_LR_schedules_horizon30':True,'historical_observations_imported':0,
            'TPE_observation_threshold':3,'TPE_threshold_reached':False,'sampled_trials_startup_random_streams_verified':True,
            'official_test_inference':False,'automatic_full_data_refit':False,'sqlite_sha256':db_hash,
            'script_sha256':sha256(__file__),'runtime':runtime,'trials':results}
        for name in ['status.json','study_summary.json','results.md','selected_recipe.json','identity.json','study_config.json','historical_preservation.json',
                     'extension_authorization_20261101.json','extension_previous_status_20261101.json','extension_applied_20261101.json']:
            audit.setdefault('summary_artifact_sha256',{})[name]=sha256(ROOT/name)
        atomic_json(ROOT/'selection_completion_audit_20261101.json',audit)
        atomic_json(ROOT/'validation_metrics_verified_20261101.json',{'objective':OBJECTIVE,'selected_trial':0,
            'trials':[{k:x[k] for k in ['number','outcome','params','selected_epoch','validation']} for x in results]})
    print(json.dumps({'audit_passed':True,'runtime':runtime,'selected':results[0]['validation'],'candidate':results[1]['validation']}),flush=True)


if __name__=='__main__':
    main()
