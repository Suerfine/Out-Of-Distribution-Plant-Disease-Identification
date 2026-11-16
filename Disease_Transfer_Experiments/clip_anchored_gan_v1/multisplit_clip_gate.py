"""Train and evaluate CLIP-only gates across three pair-disjoint development splits."""
import json

import numpy as np
import pandas as pd
import torch

from multisplit_gate import (HERE, WORKSPACE, PROJECT, COLUMNS, SEEDS, DETAIL,
                             rebuild_clip, make_validation_split, neural)
from learned_gate import fit_gate, score_gate, metrics
from neural_gate import threshold_for

BASE = HERE/'multisplit_clip_gate'


def summarize_scores(scores, seen, unseen, crops):
    ss,us = scores[:,seen],scores[:,unseen]
    a,b = seen[ss.argmax(1)],unseen[us.argmax(1)]
    st,ut = np.partition(ss,-2,axis=1)[:,-2:],np.partition(us,-2,axis=1)[:,-2:]
    x = np.column_stack((st[:,1]-ut[:,1],st[:,1]-st[:,0],ut[:,1]-ut[:,0],
                         st[:,1],ut[:,1],crops[a]==crops[b]))
    return a,b,x


def score_banks():
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    cfg = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())
    cfg['model']['pretrained'] = str(HERE/'ViT-B-16-state.pt')
    pairs = pd.DataFrame(p['pairs']).sort_values('pair_id').reset_index(drop=True)
    crops = pairs.crop_id.to_numpy()
    train = pd.read_csv(WORKSPACE/'Plant_Disease/DPD_Small_train.csv',header=None,names=COLUMNS)
    train['row'] = np.arange(len(train))
    v = cfg['validation']
    for seed in SEEDS:
        folder = BASE/f'split_{seed}'
        folder.mkdir(exist_ok=True,parents=True)
        bank = folder/'gate_inputs.npz'
        if bank.exists():
            continue
        fit,seen,unseen,held = make_validation_split(train,v['n_holdout_pairs'],v['min_holdout_images'],
            v['val_fraction'],seed,tuple(v['exclude_holdout_diseases']),v['max_images_per_holdout_pair'])
        val = pd.concat((seen,unseen),ignore_index=True)
        fit_pairs = np.array(sorted(set(fit.pair_id)))
        assert set(fit.file).isdisjoint(val.file) and set(held).isdisjoint(fit_pairs)
        names = val.file.to_numpy(dtype=str)
        if seed==42:
            assert set(fit.file)=={r['filename'] for r in p['roles']['fit']}
            with np.load(HERE/'hybrid/development_scores.npz') as saved:
                index = {n:i for i,n in enumerate(saved['filenames'])}
                scores = saved['clip'][[index[n] for n in names]]
        else:
            print(f'Split {seed}: rebuilding CLIP GAN head from cached fitting features',flush=True)
            scores = rebuild_clip(seed,train,fit,val,held,pairs,cfg)
        assert np.isfinite(scores).all()
        a,b,x = summarize_scores(scores,fit_pairs,np.sort(held),crops)
        temp = bank.with_suffix('.tmp.npz')
        np.savez_compressed(temp,filenames=names,x=x,a=a,b=b,y=val.pair_id.to_numpy(),
                            seen=np.arange(len(val))<len(seen))
        temp.replace(bank)
        print(f'Split {seed}: saved {len(val)} aligned gate inputs',flush=True)

def compare():
    loaded = []
    for seed in SEEDS:
        with np.load(BASE/f'split_{seed}/gate_inputs.npz') as z:
            loaded.append({k:z[k] for k in z.files})
    all_rows = {k:np.concatenate([d[k] for d in loaded]) for k in loaded[0]}
    x,a,b,y,s,names = [all_rows[k] for k in ('x','a','b','y','seen','filenames')]
    split = np.repeat(SEEDS,[len(d['y']) for d in loaded])
    # Assign the same pair to the same fold across all splits and all photos.
    unseen_pairs = np.unique(y[~s])
    assert len(unseen_pairs)>20
    rng = np.random.default_rng(42)
    mapping = {}
    for keys in (unseen_pairs,np.setdiff1d(np.unique(y),unseen_pairs)):
        mapping.update({k:i%5 for i,k in enumerate(rng.permutation(keys))})
    fold_id = np.array([mapping[z] for z in y])
    details = []
    for test_split in SEEDS:
        pred = {k:np.full(len(y),-1) for k in ('raw_clip','tuned_gap','linear','neural_8','neural_16_8')}
        tested = split==test_split
        for fold in range(5):
            test = tested & (fold_id==fold)
            cal = ~tested & (fold_id==(fold+1)%5)
            fit = ~tested & (fold_id!=fold) & (fold_id!=(fold+1)%5)
            assert set(y[test]).isdisjoint(y[fit]) and set(y[cal]).isdisjoint(y[fit])
            assert set(names[test]).isdisjoint(names[fit]) and set(names[test]).isdisjoint(names[cal])
            def gap(z):
                return z[:,0]
            pred['raw_clip'][test] = np.where(gap(x[test])>=0,a[test],b[test])
            floor = int(((np.where(gap(x[cal])>=0,a[cal],b[cal])==y[cal])&~s[cal]).sum())
            params = fit_gate(x[fit],s[fit])
            selectors = {'tuned_gap':gap,'linear':lambda z:score_gate(z,params),
                         'neural_8':neural(x[fit],s[fit],(8,),42+fold),
                         'neural_16_8':neural(x[fit],s[fit],(16,8),42+fold)}
            for name,score in selectors.items():
                tc = score(x[cal])
                threshold = threshold_for(tc,a[cal],b[cal],y[cal],s[cal],floor)
                pred[name][test] = np.where(score(x[test])>=threshold,a[test],b[test])
        for name,p in pred.items():
            assert (p[tested]>=0).all()
            details.append(dict(split=test_split,gate=name,**metrics(p[tested],y[tested],s[tested])))
    report = {'unique_pseudo_unseen_pairs':len(unseen_pairs),'splits':details,
              'mean':{arm:{key:float(np.mean([r[key] for r in details if r['gate']==arm]))
                           for key in ('seen','unseen','h')}
                      for arm in ('raw_clip','tuned_gap','linear','neural_8','neural_16_8')}}
    (BASE/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['mean']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    score_banks()
    compare()
