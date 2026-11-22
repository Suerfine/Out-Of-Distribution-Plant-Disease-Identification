"""One trainable Base disease pipeline with fitting-only feature recomposition."""
import argparse
import gc
import json
import math
import os
import time
import traceback

import numpy as np
import torch
from torch import nn

from protocol import HERE, StudyLock, load_config, prepare, rng_state, restore_rng, status, tensor_hash, sha256, stamp, write_json
from storage import CheckpointStore, checked_save, LowDiskSpace, IntegrityError
from data import epoch_schedule, training_loader, batch_metadata, exposure_arrays
from model import FeatureComposition, initialize, optimizer, cached_backward
from evaluation import validate_epoch, is_improvement, should_stop, finalize
from independent_cnn_baseline.engine import lr_factor


class ScientificFailure(RuntimeError): pass


def make_scaler(cfg):
    return torch.amp.GradScaler('cuda', enabled=cfg['amp'], init_scale=cfg['amp_initial_scale'])


def apply_step(model, opt, scaler):
    before = scaler.get_scale(); scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    return int(scaler.get_scale() < before)


def preflight(p, cfg, run):
    status(run, 'gpu_preflight')
    model = initialize(p, cfg)
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.backbone.modules()):
        raise IntegrityError('Encoder BatchNorm would be updated twice during gradient replay')
    schedule = epoch_schedule(p['roles']['fit'], p['pairs'], cfg, 0)
    loader = training_loader(p['roles']['fit'], schedule, {**cfg, 'workers': 0}, 0, 0)
    images, ids = next(iter(loader)); del loader
    if not np.array_equal(ids.numpy(), schedule[0]['unique_indices']): raise IntegrityError('Preflight input order differs')
    meta = batch_metadata(schedule[0], cfg['device'])
    names = ('backbone', 'head', 'composer', 'crop_factor', 'disease_factor')
    before = {name: tensor_hash(getattr(model, name).state_dict()) for name in names}
    opt = optimizer(model, p['baseline_config']); scaler = make_scaler(cfg)
    applied = 0; torch.cuda.reset_peak_memory_stats()
    for attempt in range(12):
        model.train(); opt.zero_grad(set_to_none=True)
        loss, detail = cached_backward(model, images, meta, cfg, 1, cfg['micro_batch'], scaler, check_replay=True)
        if not apply_step(model, opt, scaler): applied += 1
        if applied == 2: break
    else: raise ScientificFailure('Could not apply two finite composition updates')
    changed = {name: before[name] != tensor_hash(getattr(model, name).state_dict()) for name in names}
    if not all(changed.values()): raise ScientificFailure('A required model component did not update: ' + str(changed))
    store = CheckpointStore(run / 'preflight_checkpoints', p['fingerprint'], cfg['min_free_bytes'])
    store.save({'fingerprint': p['fingerprint'], 'stage': 'probe', 'epoch': 0, 'cursor': 0,
        'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state()})
    def next_update():
        model.train(); opt.zero_grad(set_to_none=True)
        cached_backward(model, images, meta, cfg, 1, cfg['micro_batch'], scaler, check_replay=True)
        apply_step(model, opt, scaler)
        return tensor_hash(model.state_dict())
    expected = next_update(); saved = store.load()
    model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
    restore_rng(saved['rng']); del saved
    if next_update() != expected: raise ScientificFailure('Composition checkpoint next-update replay differs')
    value = {'passed': True, 'utc': stamp(), 'model': cfg['model'],
        'backbone_parameters': sum(x.numel() for x in model.backbone.parameters()),
        'component_weights_changed': changed, 'stochastic_encoder_replay_exact': True,
        'checkpoint_next_update_exact': True, 'real_images': meta['primary_count'],
        'unique_encoded_images': len(images), 'synthetic_examples': len(meta['target_disease']),
        'novel_synthetic_examples': int(schedule[0]['novel'].sum()),
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'updates_discarded': True,
        'loss_components': detail, 'ssl_encoder_reused': p['ssl_encoder_sha256']}
    write_json(run / 'preflight.json', value); status(run, 'preflight_passed', **{k:v for k,v in value.items() if k != 'utc'})
    del model, opt, scaler, images; gc.collect(); torch.cuda.empty_cache()


def empty_totals():
    return {'samples': 0, 'encoded_samples': 0, 'synthetic_samples': 0, 'novel_synthetic_samples': 0,
        'loss_sum': 0., 'correct': 0, 'skips': 0, 'steps': 0, 'seconds': 0., 'components': {}}


def sync_histories(run, state):
    directory = run / 'finetune'; directory.mkdir(exist_ok=True)
    history = state['histories']['finetune']
    write_json(directory / 'history.json', history)
    write_json(directory / 'summary.json', {'completed_epochs': len(history), 'last': history[-1] if history else None})


def run_training(p, cfg, run, micro):
    store = CheckpointStore(run, p['fingerprint'], cfg['min_free_bytes']); saved = store.load()
    if saved:
        sync_histories(run, saved)
        if saved['stage'] == 'report': return
        if saved['stage'] != 'finetune': raise IntegrityError('Unexpected composition checkpoint stage')
        state = {k:v for k,v in saved.items() if k not in ('model', 'optimizer', 'scaler', 'rng')}
        model = FeatureComposition(p['baseline_config'], cfg, pretrained=False).to(cfg['device'])
        opt = optimizer(model, p['baseline_config']); scaler = make_scaler(cfg)
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng']); del saved
    else:
        model = initialize(p, cfg); opt = optimizer(model, p['baseline_config']); scaler = make_scaler(cfg)
        state = {'fingerprint': p['fingerprint'], 'stage': 'finetune', 'epoch': 0, 'cursor': 0,
            'histories': {'finetune': []}, 'best': None, 'stale_epochs': 0, 'totals': empty_totals()}
        write_json(run / 'initialization.json', {'ssl_encoder_sha256': p['ssl_encoder_sha256'],
            'backbone_hash': tensor_hash(model.backbone.state_dict()), 'head_hash': tensor_hash(model.head.state_dict()),
            'fresh_optimizer': True, 'seed': cfg['seed'], 'factor_seed': cfg['seed'] + 1703})
    last_save = time.monotonic()
    def commit():
        nonlocal last_save
        status(run, 'checkpointing', stage='finetune', epoch=state['epoch'], step=state['cursor'])
        store.save({**state, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                    'scaler': scaler.state_dict(), 'rng': rng_state()})
        last_save = time.monotonic()
    if not store.pointer.exists(): commit()
    while not should_stop(state['epoch'], state['stale_epochs'], cfg):
        epoch = state['epoch']; model.train()
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lr_factor(epoch + 1, p['baseline_config']['training'])
        schedule = epoch_schedule(p['roles']['fit'], p['pairs'], cfg, epoch)
        loader = training_loader(p['roles']['fit'], schedule, cfg, epoch, state['cursor'])
        for images, ids in loader:
            started = time.monotonic(); step = state['cursor']; item = schedule[step]
            if not np.array_equal(ids.numpy(), item['unique_indices']): raise IntegrityError('Real/donor input order changed')
            meta = batch_metadata(item, cfg['device']); opt.zero_grad(set_to_none=True)
            loss, details = cached_backward(model, images, meta, cfg, epoch + 1, micro, scaler)
            skipped = apply_step(model, opt, scaler); t = state['totals']; count = meta['primary_count']
            t['samples'] += count; t['encoded_samples'] += len(images)
            t['synthetic_samples'] += cfg['synthetic_batch']; t['novel_synthetic_samples'] += int(item['novel'].sum())
            t['loss_sum'] += loss * count; t['correct'] += details.pop('real_correct')
            for key,value in details.items(): t['components'][key] = t['components'].get(key, 0.) + value * count
            t['skips'] += skipped; t['steps'] += 1; t['seconds'] += time.monotonic() - started
            state['cursor'] += 1
            if step == 0 or state['cursor'] % 10 == 0 or state['cursor'] == len(schedule):
                status(run, 'training', stage='finetune', epoch=epoch + 1, epochs=cfg['finetune_epochs'],
                    step=state['cursor'], total_steps=len(schedule), loss=t['loss_sum'] / t['samples'],
                    micro_batch=micro, synthetic_examples=t['synthetic_samples'], novel_synthetic_examples=t['novel_synthetic_samples'])
            if time.monotonic() - last_save >= cfg['checkpoint_seconds']: commit()
            if (run / 'STOP.request').exists():
                commit(); status(run, 'paused', stage='finetune', epoch=epoch + 1, step=state['cursor']); return 'paused'
        del loader
        t = state['totals']; arrays = exposure_arrays(schedule, len(p['roles']['fit']), len(p['pairs']))
        if t['samples'] != 5798 or t['steps'] != len(schedule) or t['skips'] == t['steps']:
            raise ScientificFailure('Incomplete primary image exposure or no applied updates')
        if t['encoded_samples'] != int(arrays['encoded_counts'].sum()) or not np.all(arrays['primary_counts'] == 1):
            raise IntegrityError('Actual primary/donor image exposure differs from the schedule')
        before_validation = rng_state()
        try: validation, bank, bank_hash = validate_epoch(model, p, cfg, run, epoch + 1)
        finally: restore_rng(before_validation)
        score = validation['objective']['harmonic_mean_top1']
        if not math.isfinite(score): raise ScientificFailure('Nonfinite validation objective')
        if is_improvement(score, state['best']):
            directory = run / 'best_models'; directory.mkdir(exist_ok=True); path = directory / f'epoch_{epoch+1:02}.pt'
            digest = checked_save(path, {'model': model.state_dict(), 'fingerprint': p['fingerprint'],
                'epoch': epoch + 1, 'objective': validation['objective']}, cfg['min_free_bytes'], run / 'checkpoint_events.jsonl')
            state['best'] = {'epoch': epoch + 1, 'harmonic_mean_top1': score, 'path': str(path), 'sha256': digest,
                'validation_bank': bank, 'validation_bank_sha256': bank_hash}
            state['stale_epochs'] = 0
        else: state['stale_epochs'] += 1
        directory = run / 'finetune'; directory.mkdir(exist_ok=True); exposure = directory / f'exposures_epoch{epoch+1:02}.npz'
        np.savez_compressed(exposure, **arrays)
        value = {'epoch': epoch + 1, 'training': t, 'loss': t['loss_sum'] / t['samples'], 'seconds': t['seconds'],
            'loss_components': {k:v / t['samples'] for k,v in t['components'].items()},
            'micro_batch': micro, 'validation': validation, 'exposure_sha256': sha256(exposure)}
        state['histories']['finetune'].append(value); state['epoch'] += 1; state['cursor'] = 0; state['totals'] = empty_totals()
        commit(); sync_histories(run, state)
        status(run, 'epoch_complete', stage='finetune', epoch=state['epoch'], epochs=cfg['finetune_epochs'],
               loss=value['loss'], validation_H=score, best_epoch=state['best']['epoch'], stale_epochs=state['stale_epochs'])
    if state['best'] is None: raise ScientificFailure('No checkpoint selected')
    write_json(run / 'selection_frozen.json', {'frozen_utc': stamp(), 'fingerprint': p['fingerprint'],
        'best': state['best'], 'official_scores_used': False, 'selection': 'maximum gamma0 validation H; earliest exact tie',
        'completed_supervised_epochs': state['epoch'], 'stop_reason': 'patience8' if state['stale_epochs'] >= cfg['patience'] else 'max30'})
    state['stage'] = 'report'; store.save(state)
    del model, opt, scaler; gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default=str(HERE / 'config.json'))
    parser.add_argument('--micro-batch', type=int); parser.add_argument('--prepare-only', action='store_true'); parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(); cfg = load_config(args.config); micro = args.micro_batch or cfg['micro_batch']
    if micro not in cfg['allowed_micro_batches']: raise ValueError('Unapproved encoder microbatch')
    run = HERE / 'runs' / cfg['run_name']; run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(2); os.environ['HF_HUB_OFFLINE'] = '1'
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    with StudyLock(run / 'worker.lock'):
        try:
            status(run, 'preparing', micro_batch=micro); p = prepare(cfg, run)
            if args.prepare_only: return
            if not (run / 'preflight.json').exists(): preflight(p, {**cfg, 'micro_batch': micro}, run)
            if args.preflight_only: return
            if run_training(p, cfg, run, micro) == 'paused': return
            check = finalize(p, cfg, run)
            status(run, 'complete', verification_checks=check['checks'], results=str(run / 'results.md'))
        except torch.OutOfMemoryError as error:
            status(run, 'retryable_cuda_oom', error=repr(error), micro_batch=micro); raise SystemExit(42)
        except LowDiskSpace as error:
            status(run, 'waiting_for_disk_space', error=repr(error)); raise SystemExit(43)
        except (ScientificFailure, FloatingPointError) as error:
            status(run, 'scientific_failure', error=repr(error), traceback=traceback.format_exc()); raise SystemExit(65)
        except (IntegrityError, ValueError, AssertionError) as error:
            status(run, 'integrity_failure', error=repr(error), traceback=traceback.format_exc()); raise SystemExit(78)
        except BaseException as error:
            status(run, 'failed', error=repr(error), traceback=traceback.format_exc()); raise


if __name__ == '__main__': main()
