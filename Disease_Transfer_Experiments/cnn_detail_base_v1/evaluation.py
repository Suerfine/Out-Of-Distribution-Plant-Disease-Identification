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
def validate_epoch(model,p,cfg,run,epoch):
    from data import DetailImages
    from storage import replace_with_retry
    rows=p['roles']['validation_seen']+p['roles']['validation_unseen']
    kw={'num_workers':cfg['workers'],'pin_memory':True,'generator':torch.Generator().manual_seed(cfg['seed']+70000)}
    if cfg['workers']:kw['prefetch_factor']=cfg['prefetch_factor']
    data=DataLoader(DetailImages(rows,p['baseline_config'],cfg,model.arm),batch_size=cfg['eval_batch_size'],**kw)
    model.eval();model.backbone.set_grad_checkpointing(False);values=[];count=0
    status(run,'validation',arm=model.arm,epoch=epoch,images=0,total_images=len(rows))
    try:
        for batch in data:
            local=batch['local'].to(cfg['device'],non_blocking=True) if 'local' in batch else None
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']):
                logits=model(batch['global'].to(cfg['device'],non_blocking=True),local)
            values.append(logits.float().cpu().numpy());count+=len(logits)
            if count%100==0:status(run,'validation',arm=model.arm,epoch=epoch,images=count,total_images=len(rows))
    finally:model.backbone.set_grad_checkpointing(cfg['grad_checkpointing'])
    disease=np.concatenate(values);metrics=validation_metrics(disease,p)
    directory=run/'validation';directory.mkdir(exist_ok=True);path=directory/f'epoch_{epoch:02}_logits.npz'
    temp=path.with_suffix('.npz.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle,disease=disease,filenames=[r['filename'] for r in rows],composition_targets=[r['composition_id'] for r in rows])
    replace_with_retry(temp,path);write_json(directory/f'epoch_{epoch:02}_metrics.json',metrics)
    return metrics,str(path),sha256(path)


def finalize(p,cfg,run,arm):
    from protocol import arm_fingerprint
    from data import epoch_indices
    h=json.loads((run/'history.json').read_text());selection=json.loads((run/'selection_frozen.json').read_text());best=selection['best']
    if not h or best['epoch']!=1+int(np.argmax([x['validation']['objective']['harmonic_mean_top1'] for x in h])):
        raise IntegrityError('Selected epoch is not earliest maximum')
    if p['source_hashes']!=source_hashes():raise IntegrityError('Source changed during training')
    checks=2
    for field in ('reference_hashes','image_hashes','cache_hashes'):
        for path,digest in p[field].items():
            if sha256(path)!=digest:raise IntegrityError('Changed '+field+': '+path)
            checks+=1
    for row in h:
        path=run/f"exposures_epoch{row['epoch']:02}.npz"
        if sha256(path)!=row['exposure_sha256']:raise IntegrityError('Exposure bank changed')
        with np.load(path) as bank:
            if not np.array_equal(bank['indices'],epoch_indices(5798,cfg,row['epoch']-1)) or not np.all(bank['image_counts']==1):
                raise IntegrityError('Missing/duplicate primary image exposure')
        if row['training']['samples']!=5798:raise IntegrityError('Incorrect fitting image count')
        checks+=3
    if sha256(best['path'])!=best['sha256'] or sha256(best['validation_bank'])!=best['validation_bank_sha256']:
        raise IntegrityError('Selected model/logit checksum differs')
    with np.load(best['validation_bank']) as bank:
        rows=p['roles']['validation_seen']+p['roles']['validation_unseen']
        if bank['filenames'].tolist()!=[r['filename'] for r in rows]:raise IntegrityError('Validation filename order differs')
        if bank['composition_targets'].tolist()!=[r['composition_id'] for r in rows]:raise IntegrityError('Validation labels differ')
        metrics=validation_metrics(bank['disease'],p)
    if metrics!=h[best['epoch']-1]['validation']:raise IntegrityError('Selected metrics do not reproduce')
    payload=torch.load(best['path'],map_location='cpu',weights_only=True)
    if payload['fingerprint']!=arm_fingerprint(p,arm) or payload['arm']!=arm:raise IntegrityError('Selected arm identity differs')
    for value in payload['model'].values():
        if not torch.isfinite(value).all():raise IntegrityError('Nonfinite selected weight')
        checks+=1
    del payload
    probe=json.loads((run/'preflight.json').read_text())
    if not probe['passed'] or not probe['checkpoint_next_update_exact']:raise IntegrityError('GPU/recovery preflight failed')
    reference=json.loads((run.parent/'baseline_validation_reference.json').read_text())
    report={'arm':arm,'experiment':cfg['experiment'],'selected_epoch':best['epoch'],'completed_epochs':len(h),
        'gamma':0.,'baseline':reference,'experiment_validation':metrics,'selection':selection,
        'official_test_evaluation':False,'full_data_refit_started':False,'backbone':cfg['model']}
    write_json(run/'results.json',report)
    text=['# ConvNeXt-Base detail development results','',f'Arm: {arm}; original development split; gamma zero.',
        '', '| Model | Seen Top-1 | Pseudo-unseen Top-1 | H |','|---|---:|---:|---:|']
    for name,m in [('Saved Base',reference['objective']),(arm,metrics['objective'])]:
        text.append(f"| {name} | {m['seen_top1']:.2%} | {m['unseen_top1']:.2%} | {m['harmonic_mean_top1']:.2%} |")
    text+=['',f"Selected epoch {best['epoch']}; completed {len(h)} epochs; stop reason {selection['stop_reason']}.",
        'Inputs use original-image-derived 512 caches, with the same 384 global view in both arms. The local arm adds four 224 real patches and attention pooling. The crop network/reference are reused.',
        'This is one seed and a complete-recipe comparison; additional local views increase computation. No official-test inference or automatic full-data refit was performed.']
    (run/'results.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    check={'passed':True,'checks':checks+6,'arm':arm,'selected_metrics_reproduced':True,
        'sources_inputs_references_unchanged':True,'fitting_images_per_epoch':5798,'official_test_images_loaded':False}
    write_json(run/'verification.json',check);return check


def finalize_study(p,cfg,root):
    reference=json.loads((root/'baseline_validation_reference.json').read_text());results={}
    lines=['# Two ConvNeXt-Base detail experiments','', 'Uncalibrated original development validation; no official-test evaluation.',
        '', '| Model | Seen Top-1 | Pseudo-unseen Top-1 | H |','|---|---:|---:|---:|']
    def add(name,m):lines.append(f"| {name} | {m['seen_top1']:.2%} | {m['unseen_top1']:.2%} | {m['harmonic_mean_top1']:.2%} |")
    add('Saved Base224',reference['objective'])
    for arm in cfg['arms']:
        check=json.loads((root/arm/'verification.json').read_text())
        if not check['passed']:raise IntegrityError('Unverified arm: '+arm)
        results[arm]=json.loads((root/arm/'results.json').read_text());add(arm,results[arm]['experiment_validation']['objective'])
    write_json(root/'results.json',{'baseline':reference,'arms':results,'best_experimental_arm':max(results,key=lambda k:results[k]['experiment_validation']['objective']['harmonic_mean_top1']),
        'official_test_evaluation':False,'full_data_refit_started':False})
    (root/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return sum(json.loads((root/arm/'verification.json').read_text())['checks'] for arm in cfg['arms'])


