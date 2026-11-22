"""Fit crop-context gates on complementary development splits."""
import argparse
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import torch

from crop_context_gate import build
from expert_correctness_gate import DATA, DETAIL, HERE, fit_gate, gate_score, write_json
from multisplit_gate import checkpointed_convnext
from study import COLUMNS, PROJECT, load_clip, make_validation_split, seed_everything, text_conditions
from dpd.feature_gen import CosineClassifier, fit_classifier
from dpd.resize_cache import ensure_resized
from synthetic_gate import synthetic_bank
import synthetic_gate


OUT = HERE/'multisplit_crop_context'
SEEDS = (42, 43, 44)


def clip_scores(seed, root, train, fit, validation, held, pairs, cfg):
    with np.load(HERE/'runs'/f'seed_{seed}'/'features.npz') as bank:
        assert np.array_equal(train.file.to_numpy(), bank['filenames'])
        embeddings = bank['embeddings']
    if seed == 42:
        state = torch.load(HERE/'clip_disease_only/disease_plus_gan_head.pt',
                           map_location='cpu',weights_only=True)
        head = CosineClassifier(state['weight'], cfg['classifier']['logit_scale'])
        with torch.inference_mode():
            return head(torch.from_numpy(embeddings[validation.row.to_numpy()])).numpy()
    synthetic_gate.OUT = root
    synthetic_gate.EXTRA_PER_PAIR = cfg['synthesis']['n_per_unseen']
    generated, sy = synthetic_bank(train,fit,pairs,held,cfg,embeddings)
    assert np.array_equal(sy,np.repeat(held,synthetic_gate.EXTRA_PER_PAIR))
    text_cfg = deepcopy(cfg)
    text_cfg['condition']['parts'] = ['disease']
    model, _, tokenizer = load_clip(cfg['model'],'cuda')
    _, conditions = text_conditions(text_cfg,model,tokenizer,pairs,'cuda')
    del model
    di = pairs.disease_id.to_numpy()
    classes, order = np.unique(di,return_index=True)
    assert np.array_equal(classes,np.arange(175))
    features = np.concatenate((embeddings[fit.row.to_numpy()], generated))
    labels = np.concatenate((fit.disease_id.to_numpy(),di[sy]))
    seed_everything(42)
    head = fit_classifier(torch.as_tensor(features,device='cuda'),
                          torch.as_tensor(labels,device='cuda'),
                          torch.ones(175,device='cuda',dtype=torch.bool),
                          torch.as_tensor(conditions[order],device='cuda'),
                          cfg['classifier'],1,tag=f'split_{seed}_disease')
    torch.save(head.state_dict(),root/'clip_disease_head.pt')
    with torch.inference_mode():
        return head(torch.as_tensor(embeddings[validation.row.to_numpy()],device='cuda')).cpu().numpy()


def build_banks():
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    pairs = pd.DataFrame(p['pairs']).sort_values('pair_id').reset_index(drop=True)
    ci,di = pairs.crop_id.to_numpy(), pairs.disease_id.to_numpy()
    cfg = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())
    cfg['model']['pretrained'] = str(HERE/'ViT-B-16-state.pt')
    v = cfg['validation']
    train = pd.read_csv(DATA/'DPD_Small_train.csv',header=None,names=COLUMNS)
    train['row'] = np.arange(len(train))
    cache = ensure_resized(train.file,p['baseline_config']['input']['resize_cache'])
    for seed in SEEDS:
        root = OUT/f'split_{seed}'
        root.mkdir(exist_ok=True,parents=True)
        if (root/'gate_inputs.npz').exists():
            continue
        fit,seen,unseen,held = make_validation_split(
            train,v['n_holdout_pairs'],v['min_holdout_images'],v['val_fraction'],seed,
            tuple(v['exclude_holdout_diseases']),v['max_images_per_holdout_pair'])
        validation = pd.concat((seen,unseen),ignore_index=True)
        held, si = np.sort(held), np.sort(fit.pair_id.unique())
        assert set(fit.file).isdisjoint(validation.file) and not np.intersect1d(si,held).size
        write_json(root/'manifest.json',dict(seed=seed,fit=fit.file.tolist(),validation=validation.file.tolist(),
                                            held_pairs=held.tolist()))
        if seed == 42:
            ref = p['roles']['validation_seen']+p['roles']['validation_unseen']
            assert set(fit.file)=={r['filename'] for r in p['roles']['fit']}
            index = {r['filename']:i for i,r in enumerate(ref)}
            order = [index[name] for name in validation.file]
            with np.load(p['baseline_validation_logits']) as bank:
                crop, disease = bank['crop'][order],bank['disease'][order]
        else:
            scores = {}
            for target in ('crop','disease'):
                path = root/f'{target}_logits.npy'
                if not path.exists():
                    print(f'Split {seed}: fitting the ConvNeXt {target} model',flush=True)
                    config = deepcopy(p['baseline_config'])
                    scores[target] = checkpointed_convnext(target,fit,validation,config,root/target,cache)
                    temp = path.with_suffix('.tmp.npy')
                    np.save(temp,scores[target])
                    temp.replace(path)
                    torch.cuda.empty_cache()
                else:
                    scores[target] = np.load(path)
            crop,disease = scores['crop'],scores['disease']
        clip = clip_scores(seed,root,train,fit,validation,held,pairs,cfg)
        y = validation.pair_id.to_numpy()
        features,s,u = build(crop,disease,clip,ci,di,si,held,y)
        names = validation.file.to_numpy(dtype=str)
        group = np.arange(len(y)) < len(seen)
        if seed == 42:
            routed = features['original'][:,0] >= 3.
            assert (int((routed&s).sum()),int((~routed&u).sum())) == (593,315)
        tmp = root/'gate_inputs.tmp.npz'
        np.savez_compressed(tmp,x=features['crop_context'],y=y,names=names,seen=group,seen_hit=s,unseen_hit=u)
        tmp.replace(root/'gate_inputs.npz')
        print(f'Split {seed}: saved inputs for {len(y)} validation images',flush=True)


def compare():
    banks = []
    for seed in SEEDS:
        with np.load(OUT/f'split_{seed}'/'gate_inputs.npz') as data:
            banks.append({k:data[k] for k in data.files})
    x,y,names,seen,sh,uh = [np.concatenate([d[k] for d in banks]) for k in banks[0]]
    split = np.repeat(SEEDS,[len(d['y']) for d in banks])
    rng = np.random.default_rng(42)
    pseudo_unseen = np.unique(y[~seen])
    mapping = {pair:i%5 for keys in (pseudo_unseen,np.setdiff1d(np.unique(y),pseudo_unseen))
               for i,pair in enumerate(rng.permutation(keys))}
    folds = np.array([mapping[pair] for pair in y])

    def metrics(mask, route):
        a,b = int((mask&route&sh).sum()),int((mask&~route&uh).sum())
        ns,nu = int((mask&seen).sum()),int((mask&~seen).sum())
        s,u = a/ns,b/nu
        return dict(correct=[a,b], images=[ns,nu], seen=s, unseen=u, h=2*s*u/(s+u) if s+u else 0.)

    report = {}
    for held in SEEDS:
        single = 42 if held != 42 else 43
        tested = split == held
        decisions = {k:np.zeros(len(y),dtype=bool) for k in ('single','pooled','matched_diagnostic')}
        audit = []
        for fold in range(5):
            ev = tested & (folds == fold)
            eligible = ~tested & (folds != fold) & ~np.isin(names,names[ev])
            assert set(y[eligible]).isdisjoint(y[ev]) and set(names[eligible]).isdisjoint(names[ev])
            pooled = fit_gate(x,seen,eligible)
            comparison = fit_gate(x,seen,eligible & (split == single))
            a,b = gate_score(x[ev],comparison),gate_score(x[ev],pooled)
            decisions['single'][ev] = a >= 0
            decisions['pooled'][ev] = b >= 0
            # This diagnostic matches unseen hits using the evaluation labels.
            target = int(((a<0)&uh[ev]).sum())
            values = np.sort(b[uh[ev]])
            cutoff = np.nextafter(values[target-1],np.inf) if target else -np.inf
            decisions['matched_diagnostic'][ev] = b >= cutoff
            audit.append(dict(fold=fold,test_images=int(ev.sum()),pooled_fit=int(eligible.sum()),
                              single_fit=int((eligible&(split==single)).sum())))
        report[str(held)] = dict(single_training_split=single,results={k:metrics(tested,d) for k,d in decisions.items()},
                                 folds=audit)
    write_json(OUT/'results.json',report)
    print(json.dumps({seed:summary['results'] for seed,summary in report.items()}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not args.compare_only:
        build_banks()
    compare()
