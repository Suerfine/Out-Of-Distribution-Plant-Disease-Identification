"""Development-only gamma-zero validation and independent saved-result checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from protocol import status, write_json, sha256, source_hashes
from independent_cnn_baseline.uncalibrated_objective import validation_uncalibrated
from dpd.cnn import compose_pair_scores
from storage import IntegrityError


def is_improvement(value, best):
    return best is None or value > best['harmonic_mean_top1']


def should_stop(epoch, stale, cfg):
    return epoch >= cfg['finetune_epochs'] or stale >= cfg['patience']


def factor_metrics(logits, labels):
    order = np.argsort(-logits, axis=1, kind='stable')
    top1 = order[:, 0] == labels
    top5 = (order[:, :5] == labels[:, None]).any(1)
    return {'top1': float(top1.mean()), 'top5': float(top5.mean())}


def composed(crop, disease, pairs):
    scores = compose_pair_scores(torch.log_softmax(torch.as_tensor(crop), 1).numpy(),
                                 torch.log_softmax(torch.as_tensor(disease), 1).numpy(), pairs)
    scores[:, pairs.is_unseen.to_numpy()] = -np.inf
    return scores


def validation_metrics(disease, p, crop=None):
    rows = p['roles']['validation_seen'] + p['roles']['validation_unseen']
    pairs = pd.DataFrame(p['pairs'])
    frames = {role: pd.DataFrame({'pair_id': [r['composition_id'] for r in p['roles'][role]]})
              for role in ('validation_seen', 'validation_unseen')}
    with np.load(p['baseline_validation_logits']) as bank:
        reference_crop = bank['crop'].copy()
        reference_disease = bank['disease'].copy()
    if crop is None: crop = reference_crop
    if not np.array_equal(crop, reference_crop): raise IntegrityError('Development crop logits changed')
    if disease.shape != (1395, 175) or not np.isfinite(disease).all():
        raise IntegrityError('Invalid development disease logits')
    objective = validation_uncalibrated(crop, disease, frames, pairs)
    scores = composed(crop, disease, pairs)
    baseline_scores = composed(reference_crop, reference_disease, pairs)
    order = np.argsort(-scores, axis=1, kind='stable')
    baseline_pred = baseline_scores.argmax(1)
    targets = np.array([r['composition_id'] for r in rows])
    dy = np.array([r['disease_id'] for r in rows])
    cy = np.array([r['crop_id'] for r in rows])
    correct, baseline_correct = order[:, 0] == targets, baseline_pred == targets
    report = {'objective': objective, 'splits': {}}
    for role, selection in [('validation_seen', slice(0, 862)), ('validation_unseen', slice(862, None))]:
        labels = targets[selection]; hits = correct[selection]; bc = baseline_correct[selection]
        comps = {str(cid): {'images': int((labels == cid).sum()), 'correct': int(hits[labels == cid].sum())}
                 for cid in np.unique(labels)}
        report['splits'][role] = {'images': len(labels), 'pair_top1': float(hits.mean()),
            'pair_top5': float((order[selection, :5] == labels[:, None]).any(1).mean()),
            'pair_macro_top1': float(np.mean([v['correct'] / v['images'] for v in comps.values()])),
            'crop': factor_metrics(crop[selection], cy[selection]),
            'disease': factor_metrics(disease[selection], dy[selection]), 'per_composition': comps,
            'fixes_vs_baseline': int((~bc & hits).sum()), 'regressions_vs_baseline': int((bc & ~hits).sum())}
    return report


@torch.inference_mode()
def validate_epoch(model, p, cfg, run, epoch):
    from data import EvaluationImages
    rows = p['roles']['validation_seen'] + p['roles']['validation_unseen']
    data = DataLoader(EvaluationImages(rows), batch_size=cfg['eval_batch_size'], num_workers=cfg['workers'],
                      pin_memory=True, generator=torch.Generator().manual_seed(cfg['seed'] + 70000))
    model.eval(); values = []; count = 0
    status(run, 'validation', stage='finetune', epoch=epoch, images=0, total_images=len(rows))
    for images in data:
        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
            logits = model(images.to(cfg['device'], non_blocking=True))
        values.append(logits.float().cpu().numpy()); count += len(images)
        if count % 100 == 0:
            status(run, 'validation', stage='finetune', epoch=epoch, images=count, total_images=len(rows))
    disease = np.concatenate(values)
    metrics = validation_metrics(disease, p)
    directory = run / 'validation'; directory.mkdir(exist_ok=True)
    path = directory / f'epoch_{epoch:02}_logits.npz'
    temp = path.with_suffix('.npz.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, disease=disease, filenames=np.array([r['filename'] for r in rows]),
                            composition_targets=np.array([r['composition_id'] for r in rows]))
    temp.replace(path)
    write_json(directory / f'epoch_{epoch:02}_metrics.json', metrics)
    return metrics, str(path), sha256(path)


def finalize(p, cfg, run):
    from data import epoch_schedule, exposure_arrays
    history = json.loads((run / 'finetune/history.json').read_text())
    if not history: raise IntegrityError('No completed composition epochs')
    selection = json.loads((run / 'selection_frozen.json').read_text())
    best = selection['best']
    if best['epoch'] != 1 + int(np.argmax([h['validation']['objective']['harmonic_mean_top1'] for h in history])):
        raise IntegrityError('Selected epoch differs from earliest maximum')
    if p['source_hashes'] != source_hashes(): raise IntegrityError('Source changed during training')
    checks = 2
    for path, digest in p['reference_hashes'].items():
        if sha256(path) != digest: raise IntegrityError('Reference changed: ' + path)
        checks += 1
    for path,digest in p['cache_hashes'].items():
        if sha256(path)!=digest: raise IntegrityError('A cached image changed')
        checks+=1
    for value in history:
        path = run / 'finetune' / f"exposures_epoch{value['epoch']:02}.npz"
        if sha256(path) != value['exposure_sha256']: raise IntegrityError('Exposure bank changed')
        schedule=epoch_schedule(p['roles']['fit'],p['pairs'],cfg,value['epoch']-1)
        expected=exposure_arrays(schedule,5798,len(p['pairs']))
        with np.load(path) as bank:
            for key,array in expected.items():
                if not np.array_equal(bank[key],array): raise IntegrityError('Primary/donor/synthesis schedule differs: '+key)
                checks+=1
            if not np.all(bank['primary_counts']==1): raise IntegrityError('Primary image was missed or duplicated')
        if value['training']['samples'] != 5798 or value['training']['encoded_samples']!=int(expected['encoded_counts'].sum()):
            raise IntegrityError('Actual fitting/donor exposure differs')
        if value['training']['synthetic_samples']!=len(schedule)*cfg['synthetic_batch']:
            raise IntegrityError('Synthetic examples were not used for every step')
        checks+=3
    if sha256(best['path']) != best['sha256']: raise IntegrityError('Selected model changed')
    if sha256(best['validation_bank']) != best['validation_bank_sha256']: raise IntegrityError('Selected validation logits changed')
    with np.load(best['validation_bank']) as bank:
        rows = p['roles']['validation_seen'] + p['roles']['validation_unseen']
        if bank['filenames'].tolist() != [r['filename'] for r in rows]: raise IntegrityError('Validation image order changed')
        metrics = validation_metrics(bank['disease'], p)
    if metrics != history[best['epoch'] - 1]['validation']:
        raise IntegrityError('Saved validation metrics do not reproduce')
    checks += 4
    reference = json.loads((run / 'baseline_validation_reference.json').read_text())
    ssl_reference=json.loads((Path(cfg['ssl_development_run'])/'results.json').read_text())['experiment_validation']
    probe=json.loads((run/'preflight.json').read_text())
    if not probe['passed'] or not probe['stochastic_encoder_replay_exact'] or not probe['checkpoint_next_update_exact']:
        raise IntegrityError('GPU composition/recovery preflight failed')
    payload=torch.load(best['path'],map_location='cpu',weights_only=True)
    if payload['fingerprint']!=p['fingerprint']: raise IntegrityError('Selected model fingerprint differs')
    for tensor in payload['model'].values():
        if not bool(torch.isfinite(tensor).all()): raise IntegrityError('Nonfinite selected model')
        checks+=1
    del payload
    report = {'experiment': cfg['experiment'], 'selected_epoch': best['epoch'],
              'ssl_epochs_reused':20,'new_ssl_updates':0,'completed_supervised_epochs':len(history),
              'gamma': 0.0, 'baseline': reference, 'experiment_validation': metrics,
              'ssl_reference':ssl_reference,'synthesis_plan':p['synthesis_plan'],
              'selection': selection, 'official_test_evaluation': False, 'full_data_refit_started': False}
    write_json(run / 'results.json', report)
    b, e = reference['objective'], metrics['objective']
    text = ['# Feature-composition Base development results', '',
            'Original Base development split; uncalibrated scores; no official-test evaluation or automatic full-data refit.', '',
            '| Model | Seen Top-1 | Pseudo-unseen Top-1 | H |', '|---|---:|---:|---:|']
    for name, m in [('Saved Base reference', b),('Saved SSL development reference',ssl_reference['objective']),('Feature composition',e)]:
        text.append(f"| {name} | {m['seen_top1']:.2%} | {m['unseen_top1']:.2%} | {m['harmonic_mean_top1']:.2%} |")
    text += ['', f"Selected supervised epoch: {best['epoch']}; reused20 fitting-only SSL epochs and completed {len(history)} new composition-training epochs.",
             f"Selection: {selection['stop_reason']}; maximum raw validation H, earliest exact tie.", '',
             'Known candidate class identities can be synthesized; actual validation/excluded/test images never supply training donors. Inference uses real images and the unchanged independent crop/disease scorer.',
             'This is a single-seed complete-recipe comparison, including auxiliary heads, donor exposures and target balancing. The saved Base and SSL references were reused, with no control retraining. See results.json for disease, per-combination and paired decision metrics.']
    (run / 'results.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    result = {'passed': True, 'checks': checks, 'official_test_images_loaded': False,
              'sources_and_references_unchanged': True, 'selected_metrics_reproduced': True,
              'primary_fitting_images_per_epoch':5798,'additional_donor_exposures_verified':True,
              'synthetic_examples_verified':True,'validation_gamma':0.0}
    write_json(run / 'verification.json', result)
    return result
