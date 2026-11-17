"""Blend original and oversampled disease heads with development-selected weights."""
from collections import Counter
from copy import deepcopy
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from study import COLUMNS, PROJECT, WORKSPACE, clip_transforms, ensure_resized, train_lora
from dpd.feature_gen import CosineClassifier
from within_crop_disease import BASELINE, DETAIL, HERE, IndependentNet, official_rows, transforms, write_json


def score_candidates(crop, cnn, clip, labels, seen, seen_cols, ids, settings, weights,
                     *, hybrid_cutoff=None, seen_floor=None):
    crop_score, cnn_disease = crop, cnn[0]
    original = crop_score + cnn_disease
    updated = crop_score + cnn[1]
    clip_scores = crop_score + clip
    gap = clip_scores[:,seen_cols].max(1) - clip_scores[:,~seen_cols].max(1)
    unseen_pred = ids[~seen_cols][clip_scores[:,~seen_cols].argmax(1)]
    results = {}
    for w in weights:
        combined = original + w*(updated-original)
        seen_pred = ids[seen_cols][combined[:,seen_cols].argmax(1)]
        metrics = {}
        def summarize(pred, route):
            correct = pred == labels
            s, u = int(correct[seen].sum()), int(correct[~seen].sum())
            a, b = s / seen.sum(), u / (~seen).sum()
            return dict(seen_correct=s, unseen_correct=u, seen=float(a), unseen=float(b),
                        seen_routed=int((seen & route).sum()), unseen_routed=int((~seen & ~route).sum()),
                        h=float(2*a*b/(a+b)) if a+b else 0.)

        for name, (mix, offset) in settings.items():
            if mix is None:
                route = gap >= offset
                metrics[name] = summarize(np.where(route, seen_pred, unseen_pred), route)
                continue
            ensemble = mix*combined + (1-mix)*clip_scores
            ensemble_gap = ensemble[:,seen_cols].max(1) - ensemble[:,~seen_cols].max(1)
            route = ensemble_gap >= offset
            pred = ids[(ensemble - offset*seen_cols).argmax(1)]
            assert np.array_equal(np.isin(pred, ids[seen_cols]), route)
            metrics[name] = summarize(pred, route)
            metrics[f'routed_{name}'] = summarize(np.where(route, seen_pred, unseen_pred), route)
            if name == 'fixed_tuned' and w == .5:
                if seen_floor is not None:
                    correct_scores = np.sort(ensemble_gap[seen & (seen_pred == labels)])
                    assert len(correct_scores) >= seen_floor > 0
                    hybrid_cutoff = float(correct_scores[-seen_floor])
                if hybrid_cutoff is not None:
                    route = ensemble_gap >= hybrid_cutoff
                    metrics['routed_matched_seen'] = summarize(np.where(route, seen_pred, unseen_pred), route)
        results[w] = metrics
    return results, hybrid_cutoff


def main():
    sys.path.insert(0, str(HERE.parent/'leafgan_v1'))
    from classifier_multi import Images

    torch.set_num_threads(4)
    out = HERE/'seen_oversampling'
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    pairs = pd.DataFrame(p['pairs']).sort_values('pair_id').reset_index(drop=True)
    ci, di = pairs.crop_id.to_numpy(), pairs.disease_id.to_numpy()
    cfg = json.loads((BASELINE/'config.json').read_text())
    _, tf = transforms(cfg)
    prior = HERE/'clip_disease_only'
    lora = json.loads((PROJECT/'config/vlm/v10.3_text_anchor_three_splits_config.json').read_text())
    lora['model']['pretrained'] = str(HERE/'ViT-B-16-state.pt')
    lora['runtime']['num_workers'] = 0
    scale = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())['classifier']['logit_scale']
    fixed = json.loads((prior/'official/soft_gate_comparison.json').read_text())
    settings = {'hard_zero':(None,0.), 'hard_tuned':(None,3.)}
    for mode, setting in fixed['selection']['fixed'].items():
        settings[f'fixed_{mode}'] = (setting['params']['weight'], setting['offset'])
    weights = (0., .25, .5, .75, 1.)

    def encode_cnn(model, rows):
        loader = torch.utils.data.DataLoader(Images(rows, tf), batch_size=16)
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
            features = torch.cat([model.backbone(x.cuda()).float().cpu() for x,_ in loader])
        return features.cuda()

    fit, vs, vu = [p['roles'][k] for k in ('fit','validation_seen','validation_unseen')]
    val = vs + vu
    fit_pairs = np.array([r['composition_id'] for r in fit])
    ids = np.union1d(fit_pairs, [r['composition_id'] for r in vu])
    seen_cols = np.isin(ids, fit_pairs)
    source = next(k for k in p['reference_hashes']
                  if k.replace(chr(92), '/').endswith('/disease/best_epoch_15.pt'))
    model = IndependentNet(p['baseline_config'], 'disease')
    model.load_state_dict(torch.load(source, map_location='cpu',weights_only=True)['model_state'])
    model.cuda().eval()
    counts = Counter(fit_pairs)
    healthy = {r['crop_id'] for r in fit if r['disease_name']=='healthy'}
    rng = np.random.default_rng(42)
    extra = np.concatenate([rng.choice(np.flatnonzero(fit_pairs==key), min(16,n), replace=False)
                            for key,n in sorted(counts.items()) if 8<=n<=25 and ci[key] in healthy])
    print('Building the capped oversampling head on development training images', flush=True)
    x, xv = encode_cnn(model, fit), encode_cnn(model, val)
    labels = torch.as_tensor(di[fit_pairs], device='cuda')
    train_x, train_y = torch.cat((x,x[extra])), torch.cat((labels,labels[extra]))
    new = deepcopy(model.head)
    optimizer = torch.optim.AdamW(new.parameters(), lr=1e-4)
    rnd = torch.Generator(device='cuda').manual_seed(43)
    for _ in range(3):
        for batch in torch.randperm(len(train_y), device='cuda', generator=rnd).split(64):
            loss = F.cross_entropy(new(train_x[batch]), train_y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        dev_new = new(xv).cpu().numpy()
    del model, new, optimizer, x, xv, train_x, train_y
    torch.cuda.empty_cache()
    with np.load(HERE/'runs/seed_42/features.npz') as bank:
        idx = {n:i for i,n in enumerate(bank['filenames'])}
        xv = torch.from_numpy(bank['embeddings'][[idx[r['filename']] for r in val]])
    chk = torch.load(prior/'disease_plus_gan_head.pt', map_location='cpu', weights_only=True)
    with torch.inference_mode():
        dev_clip = CosineClassifier(chk['weight'],scale)(xv).numpy()
    with np.load(p['baseline_validation_logits']) as bank:
        dcols = di[ids]
        dev_args = (bank['crop'][:,ci[ids]], (bank['disease'][:,dcols],dev_new[:,dcols]),
                                       dev_clip[:,dcols], np.array([r['composition_id'] for r in val]),
                                       np.arange(len(val))<len(vs), seen_cols, ids, settings, weights)
    development, _ = score_candidates(*dev_args)
    seen_floor = development[0.]['fixed_tuned']['seen_correct']
    development, hybrid_cutoff = score_candidates(*dev_args, seen_floor=seen_floor)
    assert development[.5]['routed_matched_seen']['seen_correct'] == seen_floor
    assert development[0.]['hard_tuned']['seen_correct'] == 593
    selected = {}
    for name in settings:
        unseen_floor = development[0.][name]['unseen_correct']
        selected[name] = max((w for w in weights if development[w][name]['unseen_correct']>=unseen_floor),
                             key=lambda w:(development[w][name]['seen_correct'], -w))
    write_json(out/'blend_selection.json',dict(weights=selected, development=development))
    print('Development-selected oversampled-head weights', selected, flush=True)

    clip_dir = prior/'official'
    clip_model, _ = train_lora.build(lora,pairs,'cuda')
    clip_model.clip.visual.load_state_dict(torch.load(clip_dir/'visual_encoder.pt',map_location='cpu',weights_only=True))
    ckpt = torch.load(clip_dir/'disease_plus_gan_head.pt',map_location='cpu',weights_only=True)
    head = CosineClassifier(ckpt['weight'],scale).cuda().eval()
    _, clip_tf = clip_transforms(lora['input'],lora['augmentation'])
    inputs = []
    for split in ('test_seen','test_unseen'):
        frame = pd.read_csv(WORKSPACE/'Plant_Disease'/f'DPD_Small_{split}.csv',header=None,names=COLUMNS)
        cache = ensure_resized(frame.file,lora['input']['resize_cache'])
        loader = train_lora.loader(frame,cache,clip_tf,lora,shuffle=False)
        features, _ = train_lora.encode(clip_model,loader,'cuda',lora['runtime']['amp'],
                                       tuple(lora['evaluation'].get('tta',['identity'])))
        with torch.inference_mode():
            scores = head(torch.as_tensor(features,device='cuda')).cpu().numpy()
        with np.load(clip_dir/f'{split}.npz') as bank:
            assert np.array_equal(bank['filenames'],frame.file.to_numpy())
            assert np.array_equal(bank['labels'],frame.pair_id.to_numpy())
        inputs.append((split,frame,scores))
    del clip_model, head
    torch.cuda.empty_cache()
    model = IndependentNet(cfg,'disease')
    model.load_state_dict(torch.load(BASELINE/'disease_net.pt',map_location='cpu',weights_only=True))
    model.head.load_state_dict(torch.load(out/'oversampling_head.pt',map_location='cpu',weights_only=True))
    model.cuda().eval()
    buckets = []
    with np.load(BASELINE/'factor_logits.npz') as bank:
        for split, frame, scores in inputs:
            rows = official_rows(split)
            assert np.array_equal(frame.file.to_numpy(), [r['filename'] for r in rows])
            encoded = encode_cnn(model,rows)
            with torch.inference_mode():
                disease = model.head(encoded).cpu().numpy()
            buckets.append((bank[f'{split}_crop'][:,ci],bank[f'{split}_disease'][:,di],disease[:,di],
                            scores[:,di],frame.pair_id.to_numpy(),np.full(len(frame),split=='test_seen')))
    crop, old, new, clip_scores, labels, is_seen = [np.concatenate(z) for z in zip(*buckets)]
    ids = pairs.pair_id.to_numpy()
    official, _ = score_candidates(crop,(old,new),clip_scores,labels,is_seen,~pairs.is_unseen.to_numpy(),
                                  ids,settings,sorted(set((0.,.5,1.,*selected.values()))),
                                  hybrid_cutoff=hybrid_cutoff)
    for mode in ('zero','tuned'):
        expected = fixed['official'][mode]['fixed']
        assert official[0.][f'fixed_{mode}']['seen_correct']==expected['seen_correct']
        assert official[0.][f'fixed_{mode}']['unseen_correct']==expected['unseen_correct']
    write_json(out/'ensemble_routed_experts_results.json',
               dict(settings=settings, selected=selected, matched_cutoff=hybrid_cutoff,
                    development=development, official=official))
    print(json.dumps(official),flush=True)


if __name__ == '__main__':
    main()
