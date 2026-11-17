"""Build split-specific expert predictions, then compare gate capacities."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from study import (HERE, WORKSPACE, PROJECT, COLUMNS, make_validation_split, load_clip,
                   text_conditions, seed_everything, build_anchor_pool, fit_classifier, train_generator)
from independent_cnn_baseline.engine import Actor, predict
from independent_cnn_baseline.models import IndependentNet
from learned_gate import decisions, fit_gate, score_gate, metrics
from neural_gate import threshold_for

SEEDS = (42,43,44)
BASE = HERE/'multisplit_gate'
DETAIL = HERE.parent/'cnn_detail_base_v1'
ENCODER_EPOCHS = 15


def checkpointed_convnext(target, fit, val, cfg, folder, cache):
    identity = hashlib.sha256((json.dumps(cfg,sort_keys=True)+
                               json.dumps(fit.file.tolist())+target).encode()).hexdigest()
    actor = Actor(target,cfg,folder,identity)
    actor.model.backbone.set_grad_checkpointing(True)
    while actor.epoch < ENCODER_EPOCHS:
        actor.epoch_step(fit,None,cache)
    return predict(actor.model.cuda(),val,cache,cfg,batch_size=16).numpy()


def rebuild_clip(seed, train, fit, val, held, pairs, config):
    seed_everything(seed)
    model,_,tokenizer = load_clip(config['model'],'cuda')
    prompt,cond = [torch.as_tensor(item,device='cuda') for item in text_conditions(config,model,tokenizer,pairs,'cuda')]
    del model
    with np.load(HERE/'runs'/f'seed_{seed}'/'features.npz') as z:
        assert np.array_equal(z['filenames'],train.file.to_numpy())
        features = torch.as_tensor(z['embeddings'],device='cuda')
    x = features[fit.row.to_numpy(copy=True)]
    y = torch.as_tensor(fit.pair_id.to_numpy(copy=True),device='cuda')
    trained = torch.zeros(333,dtype=torch.bool,device='cuda')
    trained[sorted(set(fit.pair_id))] = True
    pool,_ = build_anchor_pool(x,fit,pairs)
    seed_everything(seed)
    opts = config['aux_classifier']
    aux = fit_classifier(x,y,trained,prompt,opts,opts['epochs'],tag='aux')
    gen = train_generator(x,y,cond,aux,config['gan'],pool)
    sy = torch.as_tensor(np.sort(held),device='cuda').repeat_interleave(config['synthesis']['n_per_unseen'])
    with torch.no_grad():
        sx = gen(cond[sy],sy)
    seed_everything(seed)
    allowed = torch.tensor(~pairs.is_unseen.to_numpy(),device='cuda')
    head = fit_classifier(torch.cat((x,sx)),torch.cat((y,sy)),allowed,prompt,config['classifier'],1)
    with torch.inference_mode():
        return head(features[val.row.to_numpy(copy=True)]).cpu().numpy()/100


def score_banks():
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    cnn = deepcopy(p['baseline_config'])
    clip = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())
    clip['model']['pretrained'] = str(HERE/'ViT-B-16-state.pt')
    pairs = pd.DataFrame(p['pairs']).sort_values('pair_id').reset_index(drop=True)
    train = pd.read_csv(WORKSPACE/'Plant_Disease/DPD_Small_train.csv',header=None,names=COLUMNS)
    train['row'] = np.arange(len(train))
    crops,diseases = pairs.crop_id.to_numpy(),pairs.disease_id.to_numpy()
    cache = PROJECT/'runs/cache/resized_256'
    v = clip['validation']
    for seed in SEEDS:
        folder = BASE/f'split_{seed}'
        folder.mkdir(exist_ok=True)
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
            ref = p['roles']['validation_seen']+p['roles']['validation_unseen']
            idx = {r['filename']:i for i,r in enumerate(ref)}
            assert set(fit.file)=={r['filename'] for r in p['roles']['fit']}
            with np.load(p['baseline_validation_logits']) as n,np.load(HERE/'hybrid/development_scores.npz') as c:
                order = [idx[name] for name in names]
                cidx = {n:i for i,n in enumerate(c['filenames'])}
                cnn_scores = n['crop'][order][:,crops]+n['disease'][order][:,diseases]
                clip_scores = c['clip'][[cidx[n] for n in names]]
        else:
            for target in ('crop','disease'):
                path = folder/f'{target}_logits.npy'
                if not path.is_file():
                    print(f'Split {seed}: refitting {target} ConvNeXt',flush=True)
                    logits = checkpointed_convnext(target,fit,val,cnn,folder/target,cache)
                    temp = path.with_suffix('.tmp.npy')
                    np.save(temp,logits)
                    temp.replace(path)
            cnn_scores = (np.load(folder/'crop_logits.npy')[:,crops]
                          +np.load(folder/'disease_logits.npy')[:,diseases])
            print(f'Split {seed}: rebuilding CLIP GAN head',flush=True)
            clip_scores = rebuild_clip(seed,train,fit,val,held,pairs,clip)
        assert np.isfinite(cnn_scores).all() and np.isfinite(clip_scores).all()
        a,b,x = decisions(clip_scores,cnn_scores,fit_pairs,np.sort(held),crops)
        temp = bank.with_suffix('.tmp.npz')
        np.savez_compressed(temp,filenames=names,x=x,a=a,b=b,y=val.pair_id.to_numpy(),
                            seen=np.arange(len(val))<len(seen))
        temp.replace(bank)
        print(f'Split {seed}: saved {len(val)} aligned gate inputs',flush=True)


def neural(x,y,widths,seed):
    torch.manual_seed(seed)
    center,scale = x.mean(0),np.maximum(x.std(0),1e-6)
    layers = []
    size = x.shape[1]
    for width in widths:
        layers.extend((torch.nn.Linear(size,width),torch.nn.Tanh()))
        size = width
    layers.append(torch.nn.Linear(size,1))
    model = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.05)
    tx = torch.tensor((x-center)/scale,dtype=torch.float32)
    ty = torch.tensor(y,dtype=torch.float32)
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx).squeeze(1),ty)
        loss = loss + .02*sum(p.square().mean() for p in model.parameters())
        loss.backward()
        opt.step()
    @torch.no_grad()
    def score(z):
        return model(torch.tensor((z-center)/scale,dtype=torch.float32)).squeeze(1).numpy()
    return score


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
        pred = {k:np.full(len(y),-1) for k in ('gap_rule','linear','neural_8','neural_16_8')}
        tested = split==test_split
        for fold in range(5):
            test = tested & (fold_id==fold)
            cal = ~tested & (fold_id==(fold+1)%5)
            fit = ~tested & (fold_id!=fold) & (fold_id!=(fold+1)%5)
            assert set(y[test]).isdisjoint(y[fit]) and set(y[cal]).isdisjoint(y[fit])
            assert set(names[test]).isdisjoint(names[fit]) and set(names[test]).isdisjoint(names[cal])
            scale = max(x[fit,1].std(),1e-6)
            def gap(z):
                return z[:,0]+.02*z[:,1]/scale
            floor = int(((np.where(gap(x[cal])>=.07,a[cal],b[cal])==y[cal])&~s[cal]).sum())
            params = fit_gate(x[fit],s[fit])
            selectors = {'gap_rule':gap,'linear':lambda z:score_gate(z,params),
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
                      for arm in ('gap_rule','linear','neural_8','neural_16_8')}}
    (BASE/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['mean']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    score_banks()
    compare()
