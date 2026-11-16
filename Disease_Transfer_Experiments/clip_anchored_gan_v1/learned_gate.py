"""Fit a cross-validated confidence gate for two fixed experts."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/'Out-Of-Distribution-Plant-Disease-Identification'


def decisions(clip, cnn, seen, unseen, crops):
    cs,cu = clip[:,seen],clip[:,unseen]
    ns,nu = cnn[:,seen],cnn[:,unseen]
    cnn_seen = seen[ns.argmax(1)]
    clip_unseen = unseen[cu.argmax(1)]
    clip_seen = seen[cs.argmax(1)]
    best_two = np.partition(ns,-2,axis=1)[:,-2:]
    gap = cs.max(1)-cu.max(1)
    cnn_gap = ns.max(1)-nu.max(1)
    x = np.column_stack((gap,cnn_gap,best_two[:,1]-best_two[:,0],
                          crops[cnn_seen]==crops[clip_seen],crops[cnn_seen]==crops[clip_unseen]))
    return cnn_seen,clip_unseen,x


def fit_gate(x, y):
    center = x.mean(0)
    scale = np.maximum(x.std(0),1e-6)
    z = torch.from_numpy(np.column_stack(((x-center)/scale,np.ones(len(x))))).double()
    y = torch.from_numpy(y.astype(float))
    w = torch.zeros(z.shape[1],dtype=torch.float64,requires_grad=True)
    opt = torch.optim.LBFGS([w],max_iter=60,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z@w,y)+.02*w[:-1].square().mean()
        loss.backward()
        return loss
    opt.step(closure)
    return center,scale,w.detach().numpy()


def score_gate(x, params):
    center,scale,w = params
    return ((x-center)/scale)@w[:-1]+w[-1]


def metrics(pred,y,is_seen):
    good = pred==y
    s,u = [float(good[part].mean()) for part in (is_seen,~is_seen)]
    return dict(seen=s,unseen=u,h=2*s*u/(s+u) if s+u else 0,
                seen_correct=int(good[is_seen].sum()),unseen_correct=int(good[~is_seen].sum()))


def main():
    torch.set_num_threads(1)
    out = HERE/'learned_gate'
    out.mkdir(exist_ok=True)
    h = HERE/'hybrid'
    rule = json.loads((h/'selection.json').read_text())
    prep = json.loads((HERE.parent/'cnn_detail_base_v1/runs/development_seed42/prepared.json').read_text())
    pairs = sorted(prep['pairs'],key=lambda r:r['pair_id'])
    crops,diseases = [np.array([r[key] for r in pairs]) for key in ('crop_id','disease_id')]
    official_unseen = np.flatnonzero([r['is_unseen'] for r in pairs])
    rows = prep['roles']['validation_seen']+prep['roles']['validation_unseen']
    index = {r['filename']:i for i,r in enumerate(rows)}
    with np.load(h/'development_scores.npz') as c,np.load(prep['baseline_validation_logits']) as n:
        order = np.array([index[name] for name in c['filenames']])
        assert len(set(order))==len(rows)
        y = np.array([rows[i]['composition_id'] for i in order])
        is_seen = order<len(prep['roles']['validation_seen'])
        u = np.unique(y[~is_seen])
        s = np.setdiff1d(np.arange(333),np.r_[u,official_unseen])
        a,b,x = decisions(c['clip'],(n['crop'][:,crops]+n['disease'][:,diseases])[order],s,u,crops)
    def previous(x):
        return x[:,0]+rule['cnn_gate_weight']*x[:,1]/rule['cnn_gap_scale']>=rule['threshold']
    old_dev = metrics(np.where(previous(x),a,b),y,is_seen)
    # Keep entire crop–disease pairs together when cross-fitting.
    rng = np.random.default_rng(42)
    folds = {}
    for group in (False,True):
        for i,pair in enumerate(rng.permutation(np.unique(y[is_seen==group]))):
            folds[pair] = i%5
    assignment = np.array([folds[pair] for pair in y])
    oof = np.empty(len(y))
    for fold in range(5):
        keep = assignment!=fold
        oof[~keep] = score_gate(x[~keep],fit_gate(x[keep],is_seen[keep]))
    candidates = []
    for threshold in np.r_[np.unique(oof),np.inf]:
        dev = metrics(np.where(oof>=threshold,a,b),y,is_seen)
        if dev['unseen_correct']>=old_dev['unseen_correct']:
            candidates.append((dev['h'],float(threshold),dev))
    _,threshold,dev = max(candidates,key=lambda item:item[0])
    assert np.isfinite(threshold)
    params = fit_gate(x,is_seen)
    choice = dict(threshold=threshold,features=['clip_gap','cnn_gap','cnn_top_two_margin','seen_crop_agreement','unseen_crop_agreement'],
                  gate_parameters=[p.tolist() for p in params],oof_development=dev,previous_development=old_dev)
    (out/'selection.json').write_text(json.dumps(choice,indent=2))
    preds = {'previous_gate':[],'learned_gate':[]}
    all_y, all_seen = [],[]
    base = BASE/'runs/baseline/independent_cnn_baseline_convnext_base_seed42/scores.npz'
    with np.load(base) as n:
        for split in ('test_seen','test_unseen'):
            with np.load(HERE/'official'/f'anchored_{split}.npz') as c:
                y = c['labels']
                assert np.array_equal(y,n[f'{split}_labels'])
                with (ROOT/f'Plant_Disease/DPD_Small_{split}.csv').open(encoding='utf-8-sig',newline='') as f:
                    assert np.array_equal(c['filenames'],[r[0] for r in csv.reader(f)])
                a,b,x = decisions(c['scores'],n[f'{split}_scores'],np.setdiff1d(np.arange(333),official_unseen),official_unseen,crops)
                preds['previous_gate'].append(np.where(previous(x),a,b))
                preds['learned_gate'].append(np.where(score_gate(x,params)>=threshold,a,b))
                all_y.append(y)
                all_seen.append(np.full(len(y),split=='test_seen'))
    results = {key:metrics(np.concatenate(pred),np.concatenate(all_y),np.concatenate(all_seen)) for key,pred in preds.items()}
    assert results['previous_gate']['seen_correct']==1338 and results['previous_gate']['unseen_correct']==1134
    (out/'results.json').write_text(json.dumps({'selection':choice,'official':results},indent=2))
    print(json.dumps({'previous_development':old_dev,'learned_development':dev,'official':results}))


if __name__=='__main__':
    main()
