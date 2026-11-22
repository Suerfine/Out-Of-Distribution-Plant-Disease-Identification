"""Use a frozen SigLIP crop classifier as a second opinion for routing."""
from itertools import product
import json

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.nn import functional as F

from crop_context_gate import build
from expert_correctness_gate import BASE, DATA, DETAIL, HERE, fit_gate, gate_score, write_json
from dpd.data import COLUMNS
from dpd.feature_gen import CosineClassifier
from dpd.resize_cache import ensure_resized
from siglip_gate_pilot import CachedImages


def crop_scores(features, train, train_labels, evaluation):
    x = torch.as_tensor(features[train], device='cuda')
    y = torch.as_tensor(train_labels, device='cuda')
    head = torch.nn.Linear(x.shape[1],55,device='cuda')
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.LBFGS(head.parameters(),max_iter=100,line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(x),y) + .001*head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        return head(torch.as_tensor(features[evaluation],device='cuda')).cpu().numpy()


def encoding(rows, folder):
    path = folder/'features.npz'
    names = rows.file.to_numpy(dtype=str)
    if path.exists():
        with np.load(path) as bank:
            assert np.array_equal(bank['filenames'], names)
            return bank['features']
    arch = 'ViT-B-16-SigLIP'
    config = open_clip.get_pretrained_cfg(arch,'webli')
    opts = {f'image_{key}':config[key] for key in ('mean','std','interpolation','resize_mode')}
    model, _, tf = open_clip.create_model_and_transforms(arch,pretrained=str(HERE/'ViT-B-16-SigLIP.npz'),**opts)
    visual = model.visual.cuda().eval()
    del model
    cache = ensure_resized(rows.file,256)
    paths = [cache/(name.rsplit('.',1)[0]+'.jpg') for name in names]
    loader = torch.utils.data.DataLoader(CachedImages(paths,tf),batch_size=8)
    print(f'Encoding {len(names)} images for the independent crop model',flush=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        features = torch.cat([F.normalize(visual(images.cuda()).float(),dim=-1).cpu() for images in loader]).numpy()
    assert np.isfinite(features).all()
    np.savez(path,filenames=names,features=features)
    return features


def evidence(crop,disease,clip,alt,ci,di,si,ui,y,blend_unseen):
    x,s,u = build(crop,disease,clip,ci,di,si,ui,y)
    unseen_disease = (clip+disease)*.5 if blend_unseen else clip
    a,b = crop[:,ci[si]]+disease[:,di[si]],crop[:,ci[ui]]+unseen_disease[:,di[ui]]
    ca,cb = ci[si[a.argmax(1)]],ci[ui[b.argmax(1)]]
    rows = np.arange(len(y))
    return x['crop_context'],alt[rows,ca]-alt[rows,cb],s,ui[b.argmax(1)]==y


def counts(route, mask, s, u):
    return [int((route&mask&s).sum()),int((~route&mask&u).sum())]


def choose(distance, alt, s, u, mask):
    base = distance >= 0
    original = counts(base,mask,s,u)
    best = (original[1],original[0],0)
    selection = (0.,0.)
    for width,weight in product((.1,.25,.5,.75,1.,1.5,2.),(.1,.25,.5,1.,2.,4.)):
        route = distance + weight*alt*(abs(distance)<=width) >= 0
        seen,unseen = counts(route,mask,s,u)
        score = (unseen,seen,-int((mask&(route!=base)).sum()))
        if seen>=original[0] and score>best:
            selection,best = (width,weight),score
    return selection


def main(blend_unseen=False):
    torch.set_num_threads(2)
    out = HERE/'independent_crop_gate'
    out.mkdir(exist_ok=True)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    pairs = sorted(p['pairs'],key=lambda r:r['pair_id'])
    ci,di = [np.array([r[key] for r in pairs]) for key in ('crop_id','disease_id')]
    training = pd.read_csv(DATA/'DPD_Small_train.csv',header=None,names=COLUMNS)
    splits = ('test_seen','test_unseen')
    test = [pd.read_csv(DATA/f'DPD_Small_{part}.csv',header=None,names=COLUMNS) for part in splits]
    rows = pd.concat((training,*test),ignore_index=True)
    assert rows.file.is_unique
    upstream = HERE/'clip_disease_only/official'
    for part,frame in zip(splits,test):
        with np.load(upstream/f'{part}.npz') as bank:
            assert np.array_equal(bank['filenames'],frame.file.to_numpy())
            assert np.array_equal(bank['labels'],frame.pair_id.to_numpy())
    embeddings = encoding(rows,out)
    fit,vs,vu = [p['roles'][key] for key in ('fit','validation_seen','validation_unseen')]
    val = vs+vu
    ids = lambda records:np.array([r['composition_id'] for r in records])
    labels = ids(val)
    seen = np.arange(len(val))<len(vs)
    lookup = {n:i for i,n in enumerate(training.file)}
    integer_rows = lambda records:np.array([lookup[r['filename']] for r in records])
    fit_idx,val_idx = integer_rows(fit),integer_rows(val)
    assert not np.intersect1d(fit_idx,val_idx).size
    train_crop = training.crop_id.to_numpy()
    alt = crop_scores(embeddings,fit_idx,train_crop[fit_idx],val_idx)
    pri = HERE/'clip_disease_only'
    config = json.loads((BASE/'config/vlm/v11.3_lora_features_config.json').read_text())
    weight = torch.load(pri/'disease_plus_gan_head.pt',map_location='cpu',weights_only=True)['weight']
    with np.load(HERE/'runs/seed_42/features.npz') as bank:
        index = {n:i for i,n in enumerate(bank['filenames'])}
        x = torch.from_numpy(bank['embeddings'][[index[r['filename']] for r in val]])
    with torch.inference_mode():
        clip = CosineClassifier(weight,config['classifier']['logit_scale'])(x).numpy()
    with np.load(p['baseline_validation_logits']) as bank:
        dev_x,dev_alt,s,u = evidence(bank['crop'],bank['disease'],clip,alt,ci,di,
                                      np.unique(ids(fit)),np.unique(ids(vu)),labels,blend_unseen)
        dev_crop_correct = [int((z.argmax(1)==ci[labels]).sum()) for z in (bank['crop'],alt)]
    calibrated = json.loads((upstream/'crop_context_calibration.json').read_text())['development_selected']['cutoff']
    current = json.loads((upstream/'crop_context_gate.json').read_text())['selection']['models']['crop_context']
    full = np.ones(len(val),dtype=bool)
    z = gate_score(dev_x,current)-calibrated
    assert counts(z>=0,full,s,u)[0]==682
    if not blend_unseen:
        assert counts(z>=0,full,s,u)[1]==315
    rng = np.random.default_rng(42)
    groups = {pair:i%5 for group in (labels[seen],labels[~seen])
              for i,pair in enumerate(rng.permutation(np.unique(group)))}
    folds = np.array([groups[pair] for pair in labels])
    records = []
    for i in range(5):
        ev = folds == i
        fit_rows = ~ev
        score = gate_score(dev_x,fit_gate(dev_x,seen,fit_rows))
        offsets = np.unique(score[fit_rows])

        def h(offset):
            a,b = counts(score>=offset,fit_rows,s,u)
            a,b = a/(fit_rows&seen).sum(),b/(fit_rows&~seen).sum()
            return 2*a*b/(a+b) if a+b else 0.

        distance = score-max(offsets,key=h)
        width,w = choose(distance,dev_alt,s,u,fit_rows)
        route = distance+w*dev_alt*(abs(distance)<=width)>=0
        records.append(dict(rule=[width,w],current=counts(distance>=0,ev,s,u),new=counts(route,ev,s,u)))
    width,w = choose(z,dev_alt,s,u,full)
    print('Locked rule',width,w,flush=True)
    t = len(training)
    alt_test = crop_scores(embeddings,np.arange(t),train_crop,np.arange(t,len(rows)))
    y = rows.pair_id.to_numpy()[t:]
    truth_crop=ci[y]
    test_seen = np.arange(len(y))<len(test[0])
    unseen = np.array([r['is_unseen'] for r in pairs])
    path = BASE/'runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz'
    with np.load(path) as bank:
        crop,disease = [np.concatenate([bank[f'{part}_{key}'] for part in splits])
                        for key in ('crop','disease')]
    tx,ta,ts,tu = evidence(crop,disease,np.load(upstream/'clip_scores.npy'),alt_test,ci,di,
                            np.flatnonzero(~unseen),np.flatnonzero(unseen),y,blend_unseen)
    distance = gate_score(tx,current)-calibrated
    route = distance+w*ta*(abs(distance)<=width)>=0
    official = {}
    for name,decision in [('current',distance>=0),('second_opinion',route)]:
        a,b = counts(decision,np.ones(len(y),dtype=bool),ts,tu)
        sr,ur = a/len(test[0]),b/len(test[1])
        official[name] = dict(correct=[a,b],seen=sr,unseen=ur,h=2*sr*ur/(sr+ur))
    assert official['current']['correct']==[1347,1227 if blend_unseen else 1218]
    report = dict(rule=[width,w],crop_correct_development=dev_crop_correct,
                  crop_correct_official=[int((z.argmax(1)==truth_crop).sum()) for z in (crop,alt_test)],
                  held_out={k:np.sum([f[k] for f in records],axis=0).tolist() for k in ('current','new')},
                  folds=records,official=official,changed_routes=int((route!=(distance>=0)).sum()))
    write_json(out/('results_blended_unseen.json' if blend_unseen else 'results.json'),report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend-unseen',action='store_true')
    main(parser.parse_args().blend_unseen)
