"""Blend the hard-routed pipeline with standalone CLIP and ConvNeXt."""
import json

import numpy as np
import pandas as pd
import torch

from crop_context_gate import build
from expert_correctness_gate import BASE, DATA, DETAIL, HERE, fit_gate, gate_score, write_json
from dpd.data import COLUMNS
from dpd.feature_gen import CosineClassifier


def metrics(pred, y, seen, mask):
    hit = (pred == y) & mask
    a,b = int((hit&seen).sum()),int((hit&~seen).sum())
    s,u = a/int((seen&mask).sum()),b/int((~seen&mask).sum())
    return dict(correct=[a,b],seen=s,unseen=u,h=2*s*u/(s+u) if s+u else 0.,
                overall=(a+b)/int(mask.sum()))


def heads(crop, cnn, clip, pair_clip, ci, di, si, ui, y):
    ids = np.union1d(si,ui)
    seen_cols = np.isin(ids,si)
    features,_,_ = build(crop,cnn,clip,ci,di,si,ui,y)
    conv = crop[:,ci[ids]] + cnn[:,di[ids]]
    unseen_specialist = crop[:,ci[ui]] + clip[:,di[ui]]
    return dict(ids=ids,cols=seen_cols,context=features['crop_context'],y=y,
                conv=conv.astype(np.float64),clip=pair_clip[:,ids].astype(np.float64)*100,
                unseen=unseen_specialist.astype(np.float64))


def route_predictions(data, scores):
    cols, ids = data['cols'], data['ids']
    return ids[cols][scores[:,cols].argmax(1)],ids[~cols][scores[:,~cols].argmax(1)]


def best_cutoff(gap, predictions, y, seen, fit):
    a,b = predictions
    candidates = np.unique(np.r_[0.,gap[fit]])
    return float(max(candidates,key=lambda t:(metrics(np.where(gap>=t,a,b),y,seen,fit)['h'],-abs(t))))


def fit_settings(data, seen, fit):
    y,cols = data['y'],data['cols']
    gate = fit_gate(data['context'],seen,fit)
    gap = gate_score(data['context'],gate)
    a,_ = route_predictions(data,data['conv'])
    b = data['ids'][~cols][data['unseen'].argmax(1)]
    settings = dict(gate=gate)
    settings['cutoff'] = best_cutoff(gap,(a,b),y,seen,fit)
    for key in ('clip','conv'):
        z = data[key]
        gap = z[:,cols].max(1)-z[:,~cols].max(1)
        settings[f'{key}_offset'] = best_cutoff(gap,route_predictions(data,z),y,seen,fit)
    return settings


def score_models(data, settings):
    cols = data['cols']
    route = gate_score(data['context'],settings['gate']) >= settings['cutoff']
    gated = np.full_like(data['conv'],-np.inf)
    gated[:,cols] = np.where(route[:,None],data['conv'][:,cols],-np.inf)
    gated[:,~cols] = np.where(route[:,None],-np.inf,data['unseen'])
    scores = {'gate':gated, **{k:data[k] - settings[f'{k}_offset']*cols for k in ('clip','conv')}}
    return scores


def mixture(scores, temperature, weights):
    def probability(z):
        z = z/temperature
        return np.exp(z-np.logaddexp.reduce(z,axis=1,keepdims=True))
    return mix_probabilities([probability(scores[k]) for k in ('gate','clip','conv')],weights)


def mix_probabilities(probs, weights):
    a,b,c = probs
    x,y,z = weights
    return x*a+y*b+z*c


def choose(data, seen, fit):
    settings = fit_settings(data,seen,fit)
    scores = score_models(data,settings)
    ids,y = data['ids'],data['y']
    floor = metrics(ids[scores['gate'].argmax(1)],y,seen,fit)['correct'][0]
    candidates = []
    for temperature in (1.,2.,4.):
        probs = []
        for key in ('gate','clip','conv'):
            z = scores[key]/temperature
            probs.append(np.exp(z-np.logaddexp.reduce(z,axis=1,keepdims=True)))
        for i in range(1,9):
            for j in range(1,10-i):
                weights = [i/10,j/10,(10-i-j)/10]
                pred = ids[mix_probabilities(probs,weights).argmax(1)]
                result = metrics(pred,y,seen,fit)
                candidates.append((result,temperature,weights))
    best,temperature,weights = max(candidates,key=lambda row:(row[0]['correct'][0]>=floor,
        row[0]['h'] if row[0]['correct'][0]>=floor else row[0]['correct'][0],row[0]['seen']))
    settings.update(temperature=temperature,weights=weights,seen_floor=floor,development=best)
    return settings


def evaluate(data, seen, mask, settings):
    scores = score_models(data,settings)
    blended = mixture(scores,settings['temperature'],settings['weights'])
    scores['ensemble'] = blended
    pred = {k:data['ids'][z.argmax(1)] for k,z in scores.items()}
    return {k:metrics(v,data['y'],seen,mask) for k,v in pred.items()},pred


def main():
    torch.set_num_threads(4)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    pairs = sorted(p['pairs'],key=lambda r:r['pair_id'])
    ci,di = [np.array([r[k] for r in pairs]) for k in ('crop_id','disease_id')]
    fit,vs,vu = (p['roles'][k] for k in ('fit','validation_seen','validation_unseen'))
    val = vs+vu
    ids = lambda rows: np.array([r['composition_id'] for r in rows])
    y = ids(val)
    seen = np.arange(len(y)) < len(vs)
    out = HERE/'clip_disease_only'
    cfg = json.loads((BASE/'config/vlm/v11.3_lora_features_config.json').read_text())
    weight = torch.load(out/'disease_plus_gan_head.pt',map_location='cpu',weights_only=True)['weight']
    names = [r['filename'] for r in val]
    with np.load(HERE/'runs/seed_42/features.npz') as bank:
        order = {n:i for i,n in enumerate(bank['filenames'])}
        x = torch.from_numpy(bank['embeddings'][[order[n] for n in names]])
    with torch.inference_mode():
        logits = CosineClassifier(weight,cfg['classifier']['logit_scale'])(x).numpy()
    with np.load(p['baseline_validation_logits']) as cnn,np.load(HERE/'hybrid/development_scores.npz') as clip:
        order = {n:i for i,n in enumerate(clip['filenames'])}
        data = heads(cnn['crop'],cnn['disease'],logits,clip['clip'][[order[n] for n in names]],
                     ci,di,np.unique(ids(fit)),np.unique(ids(vu)),y)
    rng = np.random.default_rng(42)
    groups = {pair:i%5 for g in (y[seen],y[~seen]) for i,pair in enumerate(rng.permutation(np.unique(g)))}
    folds = np.array([groups[pair] for pair in y])
    held = {k:np.empty_like(y) for k in ('gate','clip','conv','ensemble')}
    for fold in range(5):
        test = folds==fold
        _,pred = evaluate(data,seen,test,choose(data,seen,~test))
        for name,v in pred.items():
            held[name][test] = v[test]
    full = np.ones(len(y),dtype=bool)
    dev = {k:metrics(v,y,seen,full) for k,v in held.items()}
    settings = choose(data,seen,full)
    print('Locked settings',json.dumps({k:v for k,v in settings.items() if k!='gate'}),flush=True)
    splits = ('test_seen','test_unseen')
    frames = [pd.read_csv(DATA/f'DPD_Small_{s}.csv',header=None,names=COLUMNS) for s in splits]
    clip_scores = []
    for s,frame in zip(splits,frames):
        with np.load(HERE/'official'/f'anchored_{s}.npz') as bank:
            assert np.array_equal(bank['filenames'],frame.file.to_numpy())
            assert np.array_equal(bank['labels'],frame.pair_id.to_numpy())
            clip_scores.append(bank['scores'])
    path = BASE/'runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz'
    with np.load(path) as bank:
        crop,cnn = [np.concatenate([bank[f'{s}_{k}'] for s in splits]) for k in ('crop','disease')]
    unseen_pairs = np.array([r['is_unseen'] for r in pairs])
    y = np.concatenate([f.pair_id.to_numpy() for f in frames])
    seen = np.arange(len(y))<len(frames[0])
    data = heads(crop,cnn,np.load(out/'official/clip_scores.npy'),np.concatenate(clip_scores),
                 ci,di,np.flatnonzero(~unseen_pairs),np.flatnonzero(unseen_pairs),y)
    official,pred = evaluate(data,seen,np.ones(len(y),dtype=bool),settings)
    assert official['gate']['correct']==[1347,1218]
    disagreement = pred['ensemble']!=pred['gate']
    gains = {name:dict(fixes=int((m&disagreement&(pred['ensemble']==y)).sum()),
                       regressions=int((m&disagreement&(pred['gate']==y)).sum()))
             for name,m in [('seen',seen),('unseen',~seen)]}
    report = dict(selection=settings,held_out_development=dev,official=official,changes_from_gate=gains)
    write_json(out/'official/three_way_ensemble.json',report)
    print(json.dumps(dict(development=dev,official=official,changes=gains)),flush=True)


if __name__ == '__main__':
    main()
