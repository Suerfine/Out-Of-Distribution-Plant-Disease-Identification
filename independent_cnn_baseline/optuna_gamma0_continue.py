"""Five-hour gamma-zero continuation using the completed study's observations."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sqlite3
import sys
import traceback
from pathlib import Path

import optuna
import torch

from . import optuna_gamma0 as base
from .engine import StudyLock, Tee, atomic_json, fingerprint, keep_awake, now, read_json, sha256
from .optuna_search import SearchStop, boundary, immutable_json, request_trial, retire_recovery
from .runner import window_timing

PROJECT = base.PROJECT
RUN_NAME = 'independent_cnn_baseline_convnext_base_optuna_gamma0_continue_20261102'


def validate_config(spec):
    if spec['objective'] != base.OBJECTIVE or spec['window_hours'] != 5 or spec['trial_limit'] is not None:
        raise ValueError('Expected gamma-zero, five hours and deadline-only trial allocation')
    if spec['history_policy'] != 'clone_verified_gamma0_parent' or spec['parent_trials'] != 8:
        raise ValueError('Only the eight verified gamma-zero parent observations may be carried forward')
    if spec['official_test_inference'] or spec['automatic_final_refit']:
        raise ValueError('Continuation is validation selection only')
    root = (PROJECT / spec['output_dir']).resolve()
    parent = (PROJECT / spec['parent_root']).resolve()
    if root != PROJECT / 'runs/baseline' / RUN_NAME or parent != PROJECT / 'runs/baseline' / base.RUN_NAME:
        raise ValueError('Continuation and immutable parent must use separate approved folders')
    parent_spec_path = PROJECT / spec['parent_spec']
    if sha256(parent_spec_path) != spec['parent_spec_sha256']:
        raise ValueError('Pinned parent scientific manifest changed')
    original = read_json(parent_spec_path)
    if base.validate_spec(original) != parent:
        raise ValueError('Wrong parent study')
    return root, parent, original


def parent_signature(parent):
    with contextlib.closing(sqlite3.connect((Path(parent)/'study.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        return {table: db.execute('SELECT * FROM '+table+' ORDER BY 1').fetchall() for table in
                ['studies','study_directions','study_user_attributes','trials','trial_params',
                 'trial_user_attributes','trial_values','trial_intermediate_values']}


def bootstrap(parent, root, expected_sha256, scientific_identity):
    """SQLite backup reads the completed parent; all subsequent writes use the copy."""
    parent, root = Path(parent), Path(root)
    if sha256(parent/'study.sqlite3') != expected_sha256:
        raise ValueError('Parent SQLite changed after authorization')
    destination = root/'study.sqlite3'
    marker = root/'history_import_verified.json'
    if destination.exists():
        if not marker.exists() or read_json(marker)['parent_sqlite_sha256'] != expected_sha256:
            raise ValueError('Unverified/partial history copy; inspect before recovery')
        return
    signature = parent_signature(parent)
    rows = signature['trials']
    if len(rows) != 8 or [x[1] for x in rows] != list(range(8)) or any(x[3] not in ('COMPLETE','PRUNED') for x in rows):
        raise ValueError('Parent must contain exactly eight terminal gamma-zero trials')
    attrs = {row[2]: json.loads(row[3]) for row in signature['study_user_attributes']}
    if attrs.get('objective') != base.OBJECTIVE or attrs.get('study_identity') != scientific_identity:
        raise ValueError('Cannot import another objective or scientific identity')
    with contextlib.closing(sqlite3.connect((parent/'study.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)) as source:
        with contextlib.closing(sqlite3.connect(destination)) as target:
            source.backup(target)
    if parent_signature(root) != signature or sha256(parent/'study.sqlite3') != expected_sha256:
        raise ValueError('History copy differs or mutated its parent')
    atomic_json(marker, {'verified_at_utc':now(), 'parent_sqlite_sha256':expected_sha256,
        'scientific_identity':scientific_identity, 'gamma_zero_observations':8,
        'complete':2, 'pruned':6, 'calibrated_observations':0,
        'trial_numbers_params_values_reports_and_states_preserved':True})


def snapshot(study, root, parent, state):
    root, parent = Path(root), Path(parent)
    rows=[]
    for trial in study.trials:
        folder=(parent if trial.number < 8 else root)/'trials'/f'trial_{trial.number:03d}'
        row={'number':trial.number,'state':trial.state.name,'params':trial.params,'value':trial.value,
             'provenance':'verified_parent_gamma0' if trial.number < 8 else 'additional_trial',
             'paired_epochs':len(trial.intermediate_values)}
        if (folder/'terminal.json').exists():
            terminal=read_json(folder/'terminal.json')
            if terminal['objective'] != base.OBJECTIVE or terminal['outcome'] not in ('COMPLETE','PRUNED'):
                raise ValueError('Terminal/SQLite objective or outcome mismatch')
            if trial.state.name != 'RUNNING' and terminal['outcome'] != trial.state.name:
                raise ValueError('Terminal/SQLite outcome mismatch')
            if read_json(folder/'verification.json')['terminal_sha256'] != sha256(folder/'terminal.json'):
                raise ValueError('Terminal verification marker mismatch')
            best=terminal['best']
            row.update(selected_epoch=best['epoch'],gamma=0.,validation=best['validation'],
                weights={t:os.path.relpath(folder/p,root).replace('\\','/') for t,p in best['weights'].items()},
                logits=os.path.relpath(folder/best['logits'],root).replace('\\','/'),
                terminal=os.path.relpath(folder/'terminal.json',root).replace('\\','/'),
                training_epoch_seconds=terminal['training_epoch_seconds'])
        rows.append(row)
    eligible=[r for r in rows if r['state']=='COMPLETE' and 'terminal' in r]
    best=max(eligible,key=lambda r:r['value']) if eligible else None
    new=[r for r in rows if r['number'] >= 8]
    complete_observations=sum(r['state']=='COMPLETE' for r in rows)
    summary={'state':state,'updated_at_utc':now(),'objective':base.OBJECTIVE,'parent_gamma_zero_trials':8,
        'additional_trials_started':len(new),'additional_complete':sum(r['state']=='COMPLETE' for r in new),
        'additional_pruned':sum(r['state']=='PRUNED' for r in new),
        'additional_pending':sum(r['state'] not in ('COMPLETE','PRUNED','FAIL') for r in new),
        'trials':rows,'best_eligible':best,'trial_limit':None,'stop_policy':'five_hour_deadline',
        'complete_observations':complete_observations,'TPE_warmup_complete_required':3,
        'next_sampler_phase':'TPE' if complete_observations>=3 else 'startup_random',
        'calibrated_observations_imported':0,'official_test_inference':False,'automatic_final_refit':False}
    atomic_json(root/'study_summary.json',summary)
    if state in ('window_complete','paused') and best is not None:
        atomic_json(root/'selected_recipe_for_window.json',{'objective':base.OBJECTIVE,**best})
    lines=['# Additional gamma-zero validation trials','',f'State: {state}.',
        'Five-hour deadline; no eight-trial cap. Prior gamma-zero observations are retained; calibrated observations are excluded.',
        'Only verified COMPLETE recipes can win; interrupted/pruned trials are ineligible. No refit or official inference.', '',
        '| Trial | Source | State | Best H0 | Selected epoch | Paired epochs |',
        '|---|---|---|---:|---:|---:|']
    for r in rows:
        value=f"{r['value']:.4%}" if r['value'] is not None else 'pending'
        lines.append(f"| {r['number']} | {r['provenance']} | {r['state']} | {value} | {r.get('selected_epoch','—')} | {r['paired_epochs']} |")
    if best:
        lines += ['',f"Best eligible: trial {best['number']} / H0 {best['value']:.4%}, epoch {best['selected_epoch']}, gamma0."]
    (root/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def audit(spec, original, root, parent, resume):
    if read_json(parent/'status.json')['state'] != 'selection_complete' or (parent/'STOP.request').exists():
        raise ValueError('Completed parent changed or has a user pause')
    for name,digest in spec['parent_sha256'].items():
        if sha256(parent/name) != digest:
            raise ValueError(f'Parent artifact changed: {name}')
    immutable_json(root/'parent_study_config.json',original)
    scientific,frames,pairs,cache=base.audit_inputs(original,root,smoke=not resume)
    if scientific != spec['parent_scientific_identity']:
        raise ValueError('Pinned scientific protocol changed')
    input_audit=read_json(root/'audit.json')
    input_audit.update(parent_gamma_zero_observations_carried_forward=8,
                       historical_trial_observations_imported=8,calibrated_observations_imported=0)
    atomic_json(root/'audit.json',input_audit)
    for number in range(8):
        folder=parent/'trials'/f'trial_{number:03d}'
        ident=read_json(folder/'identity.json')
        base.verify_terminal(folder,ident['identity'],frames,pairs)
    historical=read_json(parent/'historical_preservation.json')
    for name,digest in historical['historical_sha256'].items():
        if sha256(PROJECT/name) != digest:
            raise ValueError(f'Historical/final artifact changed: {name}')
    code={name:sha256(PROJECT/name) for name in
          ['independent_cnn_baseline/optuna_gamma0_continue.py','independent_cnn_baseline/run_optuna_gamma0_continue.ps1']}
    record={'config':spec,'scientific_identity':scientific,'continuation_code_sha256':code}
    immutable_json(root/'continuation_identity.json',record)
    continuation=fingerprint(record)
    atomic_json(root/'continuation_audit.json',{'audited_at_utc':now(),'continuation_identity':continuation,
        'scientific_identity':scientific,'parent_trials_verified':8,'parent_complete':2,'parent_pruned':6,
        'preserved_historical_hashes':len(historical['historical_sha256']),
        'calibrated_observations_imported':0,'original_images':8000,'selection_cache_images':7193,
        'objective':base.OBJECTIVE,'trial_limit':None,'window_hours':5,'official_test_inference':False})
    bootstrap(parent,root,spec['parent_sqlite_sha256'],scientific)
    return scientific,continuation,frames,pairs,cache


def run(spec_path, resume=False, deadline_utc=None, clear_stop=False):
    spec=read_json(spec_path)
    root,parent,original=validate_config(spec)
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(original['base_config']['runtime']['torch_threads'])
    lock_roots=[root,parent,PROJECT/original['source_root'],
        PROJECT/'runs/baseline/independent_cnn_baseline_r0_finalization_seed42',
        PROJECT/'runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42',
        PROJECT/'runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42']
    with contextlib.ExitStack() as stack:
        for folder in lock_roots:stack.enter_context(StudyLock(folder/'runner.lock'))
        state_path=root/'status.json';state=read_json(state_path) if state_path.exists() else {}
        if state.get('window_started_at_utc') and not resume:
            raise ValueError('Existing continuation requires --resume and its saved deadline')
        if clear_stop:
            if not resume:raise ValueError('--clear-stop requires explicit user resume')
            (root/'STOP.request').unlink(missing_ok=True)
        if (root/'STOP.request').exists() or (state.get('state')=='paused' and not clear_stop):
            print('User pause preserved; no training launched.',flush=True);return
        if deadline_utc and not resume:raise ValueError('Deadline is recovery-only')
        immutable_json(root/'continuation_config.json',spec)
        (root/'runner.pid').write_text(str(os.getpid()),encoding='ascii')
        log=stack.enter_context((root/'training.log').open('a',encoding='utf-8',buffering=1))
        stack.enter_context(contextlib.redirect_stdout(Tee(sys.stdout,log)))
        stack.enter_context(contextlib.redirect_stderr(Tee(sys.stderr,log)))
        stack.enter_context(keep_awake())
        def update(**fields):
            state.update(fields,pid=os.getpid(),updated_at_utc=now(),objective=base.OBJECTIVE)
            atomic_json(state_path,state)
        study=None
        try:
            update(state='auditing',phase='audit')
            scientific,continuation,frames,pairs,cache=audit(spec,original,root,parent,resume)
            saved=deadline_utc or (state.get('window_deadline_at_utc') if resume else None)
            if resume and not saved:raise ValueError('Recovery needs the exact saved deadline')
            deadline,fields=window_timing(state,5,saved)
            update(state='running',phase='selection',**fields,scientific_identity=scientific,
                continuation_identity=continuation,error=None,parent_trials=8,trial_limit=None,
                stop_policy='five_hour_deadline',official_test_inference=False,automatic_final_refit=False,
                parent_target_epoch_seconds=spec['parent_target_epoch_seconds'],parent_wall_seconds=spec['parent_wall_seconds'])
            study=base.open_study(root,scientific)
            old=study.user_attrs.get('continuation_identity')
            if old is not None and old != continuation:raise ValueError('Continuation code/config identity changed')
            study.set_user_attr('continuation_identity',continuation)
            snapshot(study,root,parent,'running')
            while True:
                boundary(root,deadline)
                # Allocate another slot after every outcome. The five-hour clock,
                # rather than a fixed trial count, is the authorized stopping rule.
                trial=request_trial(study,len(study.trials)+1)
                if trial is None or trial.number < 8:raise ValueError('Expected an additional trial, numbered eight or later')
                cfg=base.trial_config(original['base_config'],trial.params,trial.number,root)
                cfg['selection_continuation']={'identity':continuation,'parent_trials':8}
                folder=PROJECT/cfg['output_dir']
                if shutil.disk_usage(root).free < 8*2**30:raise RuntimeError('Less than 8 GiB free; preserve checkpoints')
                print(f'Starting/resuming additional gamma-zero trial {trial.number}: {trial.params}',flush=True)
                result=base.run_trial(trial,cfg,folder,scientific,frames,pairs,cache,root,deadline,update)
                outcome=optuna.trial.TrialState[result['outcome']]
                study.tell(trial,result['best']['validation']['harmonic_mean_top1'] if outcome==optuna.trial.TrialState.COMPLETE else None,state=outcome)
                retire_recovery(folder,root)
                update(active=None)
                snapshot(study,root,parent,'running')
        except SearchStop as stopped:
            if study is not None:snapshot(study,root,parent,stopped.state)
            update(state=stopped.state,phase='checkpointed',stopped_at_utc=now())
            print(f'{stopped.state}: five-hour continuation checkpointed; no automatic new budget/refit.',flush=True)
        except BaseException:
            update(state='failed',error=traceback.format_exc())
            traceback.print_exc();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',default='independent_cnn_baseline/configs/optuna_convnext_base_gamma0_continue_20261102.json')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--deadline-utc')
    parser.add_argument('--clear-stop',action='store_true')
    args=parser.parse_args()
    run(args.spec,args.resume,args.deadline_utc,args.clear_stop)
