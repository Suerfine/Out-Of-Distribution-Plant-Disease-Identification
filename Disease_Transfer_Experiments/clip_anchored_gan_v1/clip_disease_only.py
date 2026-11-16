"""Compare disease-only CLIP heads on development validation."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import torch

import synthetic_gate as synthesis
from multisplit_gate import (HERE, WORKSPACE, PROJECT, COLUMNS, DETAIL, make_validation_split,
    load_clip, text_conditions, seed_everything, fit_classifier)

OUT = HERE/'clip_disease_only'


def score(pred,y,s):
    hit = pred==y
    a,b = hit[s].mean(),hit[~s].mean()
    return dict(seen=float(a),unseen=float(b),h=float(2*a*b/(a+b) if a+b else 0),
                seen_correct=int(hit[s].sum()),unseen_correct=int(hit[~s].sum()))


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    cfg = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())
    cfg['model']['pretrained'] = str(HERE/'ViT-B-16-state.pt')
    pairs = pd.DataFrame(p['pairs']).sort_values('pair_id').reset_index(drop=True)
    ci,di = pairs.crop_id.to_numpy(),pairs.disease_id.to_numpy()
    train = pd.read_csv(WORKSPACE/'Plant_Disease/DPD_Small_train.csv',header=None,names=COLUMNS)
    train['row'] = np.arange(len(train))
    v = cfg['validation']
    fit,vs,vu,held = make_validation_split(train,v['n_holdout_pairs'],v['min_holdout_images'],
        v['val_fraction'],42,tuple(v['exclude_holdout_diseases']),v['max_images_per_holdout_pair'])
    held = np.sort(held)
    assert len(held)==20 and set(fit.file)=={r['filename'] for r in p['roles']['fit']}
    assert set(held).isdisjoint(fit.pair_id)
    val = pd.concat((vs,vu),ignore_index=True)
    y = val.pair_id.to_numpy()
    seen = np.arange(len(y))<len(vs)
    with np.load(HERE/'runs/seed_42/features.npz') as bank:
        assert np.array_equal(train.file.to_numpy(),bank['filenames'])
        x_all = bank['embeddings']
    cfg_d = deepcopy(cfg)
    cfg_d['condition']['parts'] = ['disease']
    net,_,tokenizer = load_clip(cfg['model'],'cuda')
    _,disease_by_pair = text_conditions(cfg_d,net,tokenizer,pairs,'cuda')
    del net
    _,idx = np.unique(di,return_index=True)
    assert len(idx)==175
    init = torch.as_tensor(disease_by_pair[idx],device='cuda')
    x = torch.as_tensor(x_all[fit.row.to_numpy()],device='cuda')
    labels = torch.as_tensor(fit.disease_id.to_numpy(copy=True),device='cuda')
    allowed = torch.zeros(175,dtype=torch.bool,device='cuda')
    allowed[labels.unique()] = True
    assert allowed.all()
    synthesis.OUT = OUT
    synthesis.EXTRA_PER_PAIR = cfg['synthesis']['n_per_unseen']
    vectors,sy = synthesis.synthetic_bank(train,fit,pairs,held,cfg,x_all)
    assert len(sy)==6000 and np.array_equal(np.unique(sy),held)
    syn_y = di[sy]
    rng = np.random.default_rng(42)
    pools = {d:np.flatnonzero(fit.disease_id.to_numpy()==d) for d in np.unique(syn_y)}
    repeats = np.array([rng.choice(pools[d]) for d in syn_y])
    labels_extra = torch.as_tensor(syn_y,device='cuda')
    xv = torch.as_tensor(x_all[val.row.to_numpy()],device='cuda')
    conventional = p['roles']['validation_seen']+p['roles']['validation_unseen']
    index = {r['filename']:i for i,r in enumerate(conventional)}
    idx = np.array([index[name] for name in val.file])
    with np.load(p['baseline_validation_logits']) as cnn,np.load(HERE/'hybrid/development_scores.npz') as clip:
        crop,cnn_disease = cnn['crop'][idx],cnn['disease'][idx]
        cidx = {name:i for i,name in enumerate(clip['filenames'])}
        original = clip['clip'][[cidx[n] for n in val.file]]
    si = np.array(sorted(set(fit.pair_id)))
    ids = np.sort(np.r_[si,held])
    a = si[(crop[:,ci[si]]+cnn_disease[:,di[si]]).argmax(1)]
    original = 100*original[:,ids].astype(np.float64)
    for c in np.unique(ci[ids]):
        cols = ci[ids]==c
        original[:,cols] -= np.logaddexp.reduce(original[:,cols],axis=1)[:,None]
    original += crop[:,ci[ids]]
    bank = {'pair_based':original}
    extra = torch.as_tensor(vectors,device='cuda')
    for name,tx,ty in [('disease_real_only',x,labels),
                       ('disease_repeated_real',torch.cat((x,x[repeats])),torch.cat((labels,labels_extra))),
                       ('disease_plus_gan',torch.cat((x,extra)),torch.cat((labels,labels_extra)))]:
        seed_everything(42)
        head = fit_classifier(tx,ty,allowed,init,cfg['classifier'],1,tag=name)
        with torch.inference_mode():
            disease = head(xv).cpu().numpy()
        assert np.isfinite(disease).all()
        bank[name] = crop[:,ci[ids]]+disease[:,di[ids]]
        torch.save(head.state_dict(),OUT/f'{name}_head.pt')
    results = {}
    s = np.isin(ids,si)
    for arm,m in bank.items():
        u = held[m[:,~s].argmax(1)]
        gap = m[:,s].max(1)-m[:,~s].max(1)
        grid = [(g,score(np.where(gap>=g,a,u),y,seen)) for g in np.round(np.arange(0,15.01,.1),2)]
        gamma,cal = max(grid,key=lambda item:item[1]['h'])
        results[arm] = dict(raw_handoff=grid[0][1],calibrated_development=cal,gamma=float(gamma),
                            unseen_expert_only=int(((u==y)&~seen).sum()),
                            standalone_raw=score(ids[m.argmax(1)],y,seen))
    (OUT/'development_results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    main()
