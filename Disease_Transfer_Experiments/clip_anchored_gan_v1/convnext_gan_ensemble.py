"""Test crop-anchored ConvNeXt GAN heads in the fixed CLIP ensemble."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from expert_correctness_gate import BASE, DATA, DETAIL, HERE, write_json
from dpd.data import COLUMNS
from dpd.feature_gen import CosineClassifier
from independent_cnn_baseline.models import IndependentNet, transforms


SEEDS = (42,43,44)


def logits(config, checkpoint, filenames, gan_paths, head_key):
    sys.path.insert(0,str(HERE.parent/'leafgan_v1'))
    from classifier_multi import Images

    net = IndependentNet(config,'disease')
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(state.get('model_state',state))
    del state
    net.cuda().eval()
    _,tf = transforms(config)
    cache = BASE/'runs/cache/resized_256'
    rows = [dict(path=str(cache/(Path(name).stem+'.jpg')),disease_id=0) for name in filenames]
    loader = torch.utils.data.DataLoader(Images(rows,tf),batch_size=8)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        features = torch.cat([net.backbone(image.cuda()).float().cpu() for image,_ in loader]).cuda()
    result = {}
    for seed,path in zip(SEEDS,gan_paths):
        net.head.load_state_dict(torch.load(path/head_key,map_location='cpu',weights_only=True))
        with torch.inference_mode():
            result[seed] = net.head(features).cpu().numpy()
    assert all(np.isfinite(scores).all() for scores in result.values())
    return result


def compare(pairs, labels, is_seen, crop, conv, clip, fitted, si, ui, offset, *, tune=False):
    ci,di = pairs
    ids = np.sort(np.r_[si,ui])
    seen_cols = np.isin(ids,si)

    def decisions(disease):
        joint = crop[:,ci[ids]] + .5*(disease[:,di[ids]]+clip[:,di[ids]])
        a,b = joint[:,seen_cols],joint[:,~seen_cols]
        return (a.max(1)-b.max(1)).astype(np.float64),si[a.argmax(1)],ui[b.argmax(1)]

    def metrics(z,gamma):
        gap,a,b = z
        hit = np.where(gap>=gamma,a,b)==labels
        n,m = int((hit&is_seen).sum()), int((hit&~is_seen).sum())
        s,u = n/int(is_seen.sum()),m/int((~is_seen).sum())
        return dict(correct=[n,m],seen=s,unseen=u,overall=(n+m)/len(labels),h=2*s*u/(s+u) if s+u else 0.)

    base = decisions(conv)
    ref = metrics(base,offset)
    floor = ref['correct'][0]
    result = {}
    for seed,disease in fitted.items():
        z = decisions(disease)
        gap,a,_ = z
        result[seed] = dict(fixed=metrics(z,offset))
        if tune and floor <= int((is_seen&(a==labels)).sum()):
            # Largest offset retaining at least the baseline number of seen hits.
            tuned = float(np.sort(gap[is_seen&(a==labels)])[-floor])
            result[seed].update(offset=tuned, tuned=metrics(z,tuned))
    return ref,result,decisions,metrics


def main():
    torch.set_num_threads(2)
    out = HERE/'clip_disease_only/official'
    frozen = json.loads((out/'soft_gate_comparison.json').read_text())['selection']['fixed']['tuned']
    assert frozen['params']['weight']==.5
    offset = frozen['offset']
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    pairs = sorted(p['pairs'],key=lambda r:r['pair_id'])
    ci,di = [np.array([r[k] for r in pairs]) for k in ('crop_id','disease_id')]
    fit,vs,vu = [p['roles'][k] for k in ('fit','validation_seen','validation_unseen')]
    rows = vs+vu
    names = [r['filename'] for r in rows]
    labels = np.array([r['composition_id'] for r in rows])
    seen = np.arange(len(rows))<len(vs)
    si = np.unique([r['composition_id'] for r in fit])
    ui = np.unique(labels[~seen])
    scale = json.loads((BASE/'config/vlm/v11.3_lora_features_config.json').read_text())['classifier']['logit_scale']
    state = torch.load(out.parent/'disease_plus_gan_head.pt',map_location='cpu',weights_only=True)
    with np.load(HERE/'runs/seed_42/features.npz') as bank:
        order = {n:i for i,n in enumerate(bank['filenames'])}
        x = torch.from_numpy(bank['embeddings'][[order[n] for n in names]])
    with torch.inference_mode():
        clip = CosineClassifier(state['weight'],scale)(x).numpy()
    dev_root = HERE.parent/'feature_gan_anchored_v1/runs'
    source = next(k for k in p['reference_hashes']
                  if k.replace(chr(92),'/').endswith('/disease/best_epoch_15.pt'))
    print('Scoring development images with the saved GAN heads',flush=True)
    dev_gan = logits(p['baseline_config'],source,names,
                     [dev_root/f'seed_{s}' for s in SEEDS],'192_gan_head.pt')
    with np.load(p['baseline_validation_logits']) as bank:
        control,development,_,_ = compare((ci,di),labels,seen,bank['crop'],bank['disease'],
                                           clip,dev_gan,si,ui,offset,tune=True)
    print('Locked offsets',json.dumps({s:v.get('offset') for s,v in development.items()}),flush=True)
    splits = ('test_seen','test_unseen')
    frames = [pd.read_csv(DATA/f'DPD_Small_{s}.csv',header=None,names=COLUMNS) for s in splits]
    for split,frame in zip(splits,frames):
        with np.load(out/f'{split}.npz') as bank:
            assert np.array_equal(bank['filenames'],frame.file.to_numpy())
            assert np.array_equal(bank['labels'],frame.pair_id.to_numpy())
    labels = np.concatenate([f.pair_id.to_numpy() for f in frames])
    seen = np.arange(len(labels))<len(frames[0])
    official_root = HERE.parent/'feature_gan_anchored_official_192/runs'
    base = BASE/'runs/baseline/independent_cnn_baseline_convnext_base_seed42'
    config = json.loads((base/'config.json').read_text())
    print('Scoring official images with the saved GAN heads',flush=True)
    test_paths = [official_root/f'seed_{s}' for s in SEEDS]
    test_gan = logits(config,base/'disease_net.pt',[n for f in frames for n in f.file],
                      test_paths,'gan_head.pt')
    with np.load(base/'factor_logits.npz') as bank:
        crop,conv = [np.concatenate([bank[f'{s}_{k}'] for s in splits]) for k in ('crop','disease')]
    for seed,path in zip(SEEDS,test_paths):
        start = 0
        for split,frame in zip(splits,frames):
            stop = start+len(frame)
            with np.load(path/f'{split}.npz') as bank:
                assert np.array_equal(bank['filenames'],frame.file.to_numpy())
                reference = crop[start:stop,ci] + test_gan[seed][start:stop,di]
                assert np.array_equal(reference.argmax(1),bank['gan'])
            start = stop
    unseen_cols = np.array([r['is_unseen'] for r in pairs])
    si,ui = np.flatnonzero(~unseen_cols),np.flatnonzero(unseen_cols)
    base_metrics,official,decisions,metrics = compare((ci,di),labels,seen,crop,conv,
        np.load(out/'clip_scores.npy'),test_gan,si,ui,offset)
    assert base_metrics['correct']==[1311,1318]
    for seed,result in official.items():
        if 'offset' in development[seed]:
            result['tuned'] = metrics(decisions(test_gan[seed]),development[seed]['offset'])
    report = dict(weight=.5,fixed_offset=offset,development=dict(control=control,heads=development),
                  official=dict(control=base_metrics,heads=official))
    write_json(out/'convnext_gan_ensemble.json',report)
    print(json.dumps(report['official']),flush=True)


if __name__ == '__main__':
    main()
