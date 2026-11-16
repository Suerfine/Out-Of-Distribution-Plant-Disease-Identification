"""Evaluate an image-aware gate on crop–disease pairs withheld from gate fitting."""
import json

import numpy as np
import torch

from learned_gate import HERE, decisions, fit_gate, metrics, score_gate


def threshold_for(logits, seen_pred, unseen_pred, labels, seen, unseen_floor):
    best = None
    for threshold in np.r_[np.unique(logits), np.inf]:
        m = metrics(np.where(logits >= threshold, seen_pred, unseen_pred), labels, seen)
        if m['unseen_correct'] >= unseen_floor and (best is None or m['seen_correct'] > best[0]):
            best = m['seen_correct'], float(threshold)
    return best[1]


def main():
    torch.set_num_threads(1)
    hybrid = HERE/'hybrid'
    rule = json.loads((hybrid/'selection.json').read_text())
    prep = json.loads((HERE.parent/'cnn_detail_base_v1/runs/development_seed42/prepared.json').read_text())
    pairs = sorted(prep['pairs'], key=lambda r:r['pair_id'])
    crops,diseases = [np.array([r[k] for r in pairs]) for k in ('crop_id','disease_id')]
    rows = prep['roles']['validation_seen']+prep['roles']['validation_unseen']
    lookup = {r['filename']:i for i,r in enumerate(rows)}
    with np.load(hybrid/'development_scores.npz') as clip, np.load(prep['baseline_validation_logits']) as cnn:
        names = clip['filenames']
        order = np.array([lookup[n] for n in names])
        assert len(order)==len(rows) and len(set(order))==len(rows)
        y = np.array([rows[i]['composition_id'] for i in order])
        s = order<len(prep['roles']['validation_seen'])
        a,b,x = decisions(clip['clip'],(cnn['crop'][:,crops]+cnn['disease'][:,diseases])[order],
                          np.array(sorted({r["composition_id"] for r in prep["roles"]["fit"]})), np.unique(y[~s]), crops)
    with np.load(HERE/'runs/seed_42/features.npz') as bank:
        idx = {n:i for i,n in enumerate(bank['filenames'])}
        image = bank['embeddings'][[idx[n] for n in names]]
    assert np.isfinite(image).all()
    fixed = np.where(x[:,0]+rule['cnn_gate_weight']*x[:,1]/rule['cnn_gap_scale']>=rule['threshold'],a,b)
    rng = np.random.default_rng(42)
    by_pair = {}
    for group in (False,True):
        for i,k in enumerate(rng.permutation(np.unique(y[s==group]))):
            by_pair[k] = i%5
    fold_ids = np.array([by_pair[k] for k in y])
    candidates = {'score_only':x,'image_and_scores':np.column_stack((x,image))}
    results = {'current_gate':metrics(fixed,y,s)}
    preds = {}
    for name,features in candidates.items():
        predictions = np.full(len(y),-1)
        fold_results = []
        for fold in range(5):
            test = fold_ids==fold
            cal = fold_ids==(fold+1)%5
            train = ~(test|cal)
            assert not set(y[train]) & set(y[test]) and not set(y[cal]) & set(y[test])
            params = fit_gate(features[train],s[train])
            floor = int(((fixed[cal]==y[cal])&~s[cal]).sum())
            threshold = threshold_for(score_gate(features[cal],params),a[cal],b[cal],y[cal],s[cal],floor)
            predictions[test] = np.where(score_gate(features[test],params)>=threshold,a[test],b[test])
            fold_results.append({'fold':fold,**metrics(predictions[test],y[test],s[test])})
        assert (predictions>=0).all()
        results[name] = dict(pooled=metrics(predictions,y,s),folds=fold_results)
        preds[name] = predictions
    out = HERE/'image_gate'
    out.mkdir(exist_ok=True)
    (out/'development_results.json').write_text(json.dumps(results,indent=2))
    np.savez_compressed(out/'development_predictions.npz',filenames=names,labels=y,fold=fold_ids,current_gate=fixed,**preds)
    print(results['current_gate'])
    for name in candidates:
        print(name,results[name]['pooled'])


if __name__=='__main__':
    main()

