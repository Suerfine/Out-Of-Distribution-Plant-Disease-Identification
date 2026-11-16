"""Test GAN augmentation for a gate on real development photos."""
import json

import numpy as np
import pandas as pd
import torch

from multisplit_gate import (HERE, WORKSPACE, PROJECT, COLUMNS, DETAIL, make_validation_split,
    load_clip, text_conditions, seed_everything, fit_classifier, build_anchor_pool, train_generator)
from learned_gate import decisions, fit_gate, score_gate, metrics

OUT = HERE/'synthetic_gate'
EXTRA_PER_PAIR = 24


def synthetic_bank(train, fit, pairs, held, cfg, x_all):
    path = OUT/'synthetic.npz'
    if path.exists():
        with np.load(path) as bank:
            return bank['vectors'],bank['pair_ids']
    seed_everything(42)
    net,_,tokenizer = load_clip(cfg['model'],'cuda')
    prompt,cond = [torch.as_tensor(a,device='cuda') for a in text_conditions(cfg,net,tokenizer,pairs,'cuda')]
    del net
    x = torch.as_tensor(x_all[fit.row.to_numpy()],device='cuda')
    y = torch.tensor(fit.pair_id.to_numpy(),device='cuda')
    trained = torch.zeros(len(pairs),dtype=torch.bool,device='cuda')
    trained[np.unique(fit.pair_id)] = True
    assert not trained[held].any()
    seed_everything(42)
    aux_cfg = cfg['aux_classifier']
    aux = fit_classifier(x,y,trained,prompt,aux_cfg,aux_cfg['epochs'],tag='aux')
    pool,_ = build_anchor_pool(x,fit,pairs)
    generate = train_generator(x,y,cond,aux,cfg['gan'],pool)
    ids = np.repeat(held,EXTRA_PER_PAIR)
    ids_gpu = torch.tensor(ids,device='cuda')
    with torch.no_grad():
        vectors = generate(cond[ids_gpu],ids_gpu).cpu().numpy()
    assert np.isfinite(vectors).all() and len(vectors)==len(ids)
    np.savez_compressed(path,vectors=vectors,pair_ids=ids)
    return vectors,ids


def gate_metrics(use_seen, y, seen, a, b):
    return dict(seen_routing=float(use_seen[seen].mean()),
                unseen_routing=float((~use_seen[~seen]).mean()),
                seen_routed_correct=int(use_seen[seen].sum()),
                unseen_routed_correct=int((~use_seen[~seen]).sum()),
                **metrics(np.where(use_seen,a,b),y,seen))


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    cfg = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())
    cfg['model']['pretrained'] = str(HERE/'ViT-B-16-state.pt')
    pairs = pd.DataFrame(p['pairs']).sort_values('pair_id').reset_index(drop=True)
    crops,diseases = pairs.crop_id.to_numpy(),pairs.disease_id.to_numpy()
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
    assert (np.isin(y,held)==~seen).all()
    with np.load(HERE/'runs/seed_42/features.npz') as bank:
        assert np.array_equal(train.file.to_numpy(),bank['filenames'])
        x_all = bank['embeddings']
    vectors,ids = synthetic_bank(train,fit,pairs,held,cfg,x_all)
    x = x_all[val.row.to_numpy()]
    old_rule = json.loads((HERE/'hybrid/selection.json').read_text())
    rows = p['roles']['validation_seen']+p['roles']['validation_unseen']
    index = {r['filename']:i for i,r in enumerate(rows)}
    with np.load(HERE/'hybrid/development_scores.npz') as c,np.load(p['baseline_validation_logits']) as n:
        c_index = {name:i for i,name in enumerate(c['filenames'])}
        aligned = [index[name] for name in val.file]
        cnn = n['crop'][aligned][:,crops]+n['disease'][aligned][:,diseases]
        clip = c['clip'][[c_index[name] for name in val.file]]
    a,b,xscores = decisions(clip,cnn,np.array(sorted(set(fit.pair_id))),held,crops)
    reference = xscores[:,0]+old_rule['cnn_gate_weight']*xscores[:,1]/old_rule['cnn_gap_scale']>=old_rule['threshold']
    assert gate_metrics(reference,y,seen,a,b)['seen_correct']==641
    rng = np.random.default_rng(42)
    folds = {}
    for group in (False,True):
        folds.update({pair:i%5 for i,pair in enumerate(rng.permutation(np.unique(y[seen==group])))})
    fold = np.array([folds[pair] for pair in y])
    predictions = {arm:np.zeros(len(y),dtype=bool) for arm in ('real_only','real_repeated','real_plus_gan')}
    details = []
    for k in range(5):
        test,cal = fold==k,fold==(k+1)%5
        fit_gate_rows = ~(test|cal)
        allowed = np.unique(y[fit_gate_rows & ~seen])
        extra_mask = np.isin(ids,allowed)
        extra = vectors[extra_mask]
        repeats = np.array([rng.choice(np.flatnonzero(fit_gate_rows&(y==pair))) for pair in ids[extra_mask]])
        assert len(extra)==len(repeats) and not set(ids[extra_mask]).intersection(y[test|cal])
        floor = int((reference[cal]&seen[cal]).sum())
        for arm,addon in [('real_only',x[:0]),('real_repeated',x[repeats]),('real_plus_gan',extra)]:
            tx = np.concatenate((x[fit_gate_rows],addon))
            ty = np.concatenate((seen[fit_gate_rows],np.zeros(len(addon),dtype=bool)))
            params = fit_gate(tx,ty)
            cal_s = score_gate(x[cal],params)
            test_s = score_gate(x[test],params)
            # Preserve the reference's seen routing rate on the calibration fold.
            possible = [t for t in np.unique(cal_s) if (cal_s[seen[cal]]>=t).sum()>=floor]
            threshold = float(max(possible))
            predictions[arm][test] = test_s>=threshold
            details.append(dict(fold=k,arm=arm,threshold=threshold,added=len(addon),
                                **gate_metrics(predictions[arm][test],y[test],seen[test],a[test],b[test])))
    result = {'current_gate':gate_metrics(reference,y,seen,a,b),
              **{k:gate_metrics(v,y,seen,a,b) for k,v in predictions.items()}}
    (OUT/'results.json').write_text(json.dumps({'pooled':result,'folds':details},indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
