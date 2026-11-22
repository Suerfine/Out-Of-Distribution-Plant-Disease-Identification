"""Assess crop context in the learned hard gate with frozen experts."""
import json

import numpy as np
import pandas as pd
import torch

from expert_correctness_gate import BASE, DATA, DETAIL, HERE, fit_gate, gate_score, write_json
from dpd.data import COLUMNS
from dpd.feature_gen import CosineClassifier
from soft_moe import inputs


def build(crop, cnn, clip, ci, di, si, ui, labels, *, top_three=False):
    old, a, b, seen_hit, unseen_hit = inputs(crop, cnn, clip, ci, di, si, ui, labels)
    rows = np.arange(len(crop))
    ac, bc = ci[si[a.argmax(1)]], ci[ui[b.argmax(1)]]
    best_crop = crop.argmax(1)
    same_crop_gap = np.zeros(len(crop))
    availability = np.zeros((len(crop), 2), dtype=bool)
    for c in np.unique(best_crop):
        mask = best_crop == c
        sd, ud = di[si[ci[si] == c]], di[ui[ci[ui] == c]]
        availability[mask] = (len(sd) > 0, len(ud) > 0)
        if len(sd) and len(ud):
            same_crop_gap[mask] = clip[mask][:,sd].max(1) - clip[mask][:,ud].max(1)
    expanded = np.column_stack((old, ac != bc, crop[rows,ac] - crop[rows,bc],
                                same_crop_gap, availability))
    assert np.isfinite(expanded).all()
    gates = {'original':old, 'crop_context':expanded}
    if top_three:
        clip_seen = crop[:,ci[si]] + clip[:,di[si]]
        def top_mean(scores):
            return np.partition(scores, -3, axis=1)[:, -3:].mean(1)
        gap = top_mean(clip_seen) - top_mean(b)
        gates['top_three'] = np.column_stack((expanded, gap))
    return gates, seen_hit, unseen_hit


def run(dev, official, labels, seen, sources, test_seen, expected, *, reference='original'):
    x, s_hit, u_hit = dev
    tx, ts_hit, tu_hit = official

    def hits(scores, cutoff, mask, s, u):
        route = scores >= cutoff
        return [int((mask & route & s).sum()), int((mask & ~route & u).sum())]

    def cutoff(scores, mask, unseen_hits):
        if unseen_hits == 0:
            return float('-inf')
        sorted_hits = np.sort(scores[mask & u_hit])
        return float(np.nextafter(sorted_hits[unseen_hits-1], np.inf))

    def train(fit):
        models = {name:fit_gate(z, seen, fit) for name,z in x.items()}
        return models, {name:gate_score(x[name], params) for name,params in models.items()}

    rng = np.random.default_rng(42)
    groups = {k:i%5 for pool in (labels[seen], labels[~seen])
              for i,k in enumerate(rng.permutation(np.unique(pool)))}
    fold = np.array([groups[k] for k in labels])
    diagnostics = {}
    for kind, partitions in [('pair_folds',fold), ('source_folds',sources)]:
        results = []
        for key in np.unique(partitions):
            test = partitions == key
            fit = ~test
            _, scores = train(fit)
            fit_target = hits(scores[reference], 0., fit, s_hit, u_hit)[1]
            test_target = hits(scores[reference], 0., test, s_hit, u_hit)[1]
            results.append({name:{
                'locked':hits(z, cutoff(z, fit, fit_target), test, s_hit, u_hit),
                'matched_unseen_diagnostic':hits(z, cutoff(z, test, test_target), test, s_hit, u_hit),
            } for name,z in scores.items()})
        diagnostics[kind] = {key:{name:np.sum([r[name][key] for r in results],axis=0).tolist()
                                 for name in x} for key in ('locked','matched_unseen_diagnostic')}
    all_dev = np.ones(len(seen),dtype=bool)
    models, scores = train(all_dev)
    target = hits(scores[reference], 0., all_dev, s_hit, u_hit)[1]
    cutoffs = {name:cutoff(z, all_dev, target) for name,z in scores.items()}
    out = {}
    for name,model in models.items():
        z = gate_score(tx[name],model)
        out[name] = {}
        for kind, value in [('default',0.), ('matched_on_development',cutoffs[name])]:
            a,b = hits(z,value,np.ones(len(test_seen),dtype=bool),ts_hit,tu_hit)
            s,u = a/int(test_seen.sum()), b/int((~test_seen).sum())
            out[name][kind] = dict(seen_correct=a, unseen_correct=b, seen=s, unseen=u, h=2*s*u/(s+u))
    assert [out['original']['default'][k] for k in ('seen_correct','unseen_correct')] == expected
    return dict(selection=dict(cutoffs=cutoffs, models=models, development_unseen_target=target),
                development=diagnostics, official=out)


def main(frozen=False, top_three=False):
    torch.set_num_threads(4)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    pairs = sorted(p['pairs'],key=lambda r:r['pair_id'])
    ci,di = [np.array([r[k] for r in pairs]) for k in ('crop_id','disease_id')]
    fit,vs,vu = (p['roles'][k] for k in ('fit','validation_seen','validation_unseen'))
    val = vs+vu
    labels = np.array([r['composition_id'] for r in val])
    seen = np.arange(len(val)) < len(vs)
    sources = np.array([r['filename'].split('_',1)[0] for r in val])
    si,ui = (np.unique([r['composition_id'] for r in rows]) for rows in (fit,vu))
    out = HERE/'clip_disease_only'
    if frozen:
        bank_dir = HERE/'siglip_gate_pilot'
        clip = np.load(bank_dir/'clip_development_scores.npy')
        official_dir = bank_dir/'official_frozen_clip'
    else:
        scale = json.loads((BASE/'config/vlm/v11.3_lora_features_config.json').read_text())['classifier']['logit_scale']
        weight = torch.load(out/'disease_plus_gan_head.pt',map_location='cpu',weights_only=True)['weight']
        with np.load(HERE/'runs/seed_42/features.npz') as bank:
            order = {n:i for i,n in enumerate(bank['filenames'])}
            x = torch.from_numpy(bank['embeddings'][[order[r['filename']] for r in val]])
        with torch.inference_mode():
            clip = CosineClassifier(weight,scale)(x).numpy()
        official_dir = out/'official'
    with np.load(p['baseline_validation_logits']) as bank:
        dev = build(bank['crop'],bank['disease'],clip,ci,di,si,ui,labels,top_three=top_three)
    splits = ('test_seen','test_unseen')
    tests = [pd.read_csv(DATA/f'DPD_Small_{split}.csv',header=None,names=COLUMNS) for split in splits]
    for split,frame in zip(splits,tests):
        with np.load(out/'official'/f'{split}.npz') as bank:
            assert np.array_equal(frame.file.to_numpy(),bank['filenames'])
            assert np.array_equal(frame.pair_id.to_numpy(),bank['labels'])
    path = BASE/'runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz'
    with np.load(path) as bank:
        crop,cnn = [np.concatenate([bank[f'{s}_{k}'] for s in splits]) for k in ('crop','disease')]
    y = np.concatenate([frame.pair_id.to_numpy() for frame in tests])
    test_seen = np.arange(len(y)) < len(tests[0])
    unseen_pairs = np.array([r['is_unseen'] for r in pairs])
    official = build(crop,cnn,np.load(official_dir/'clip_scores.npy'),ci,di,
                     np.flatnonzero(~unseen_pairs),np.flatnonzero(unseen_pairs),y,top_three=top_three)
    result = run(dev,official,labels,seen,sources,test_seen,[1390,951] if frozen else [1374,1090],
                 reference='crop_context' if top_three else 'original')
    if top_three and not frozen:
        assert [result['official']['crop_context']['default'][k]
                for k in ('seen_correct','unseen_correct')] == [1407,1080]
    write_json(official_dir/('top_three_gate.json' if top_three else 'crop_context_gate.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='selection'}),flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--top-three', action='store_true')
    args = parser.parse_args()
    main(args.frozen, args.top_three)
