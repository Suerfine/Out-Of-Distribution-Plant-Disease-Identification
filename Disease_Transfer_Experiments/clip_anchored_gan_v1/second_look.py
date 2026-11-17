"""Test a multi-view agreement rule for borderline unseen routes."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seen_multiview import HERE, DETAIL, BASE, REF, GROUPS, Views, IndependentNet, sha256, write_json


def route(clip, crop, disease, ci, di, seen, unseen, rule):
    cnn = crop[:,ci]+disease[:,di]
    gap = clip[:,seen].max(1)-clip[:,unseen].max(1)
    cnn_gap = cnn[:,seen].max(1)-cnn[:,unseen].max(1)
    confidence = gap+rule['cnn_gate_weight']*cnn_gap/rule['cnn_gap_scale']
    unseen_pred = unseen[clip[:,unseen].argmax(1)]
    return rule['threshold']-confidence,unseen_pred


def second_look(paths, cfg, source, crop, disease, ci, di, seen):
    model = IndependentNet(cfg,'disease')
    saved = torch.load(source,map_location='cpu',weights_only=True)
    model.load_state_dict(saved.get('model_state',saved))
    del saved
    model.cuda().eval()
    loader = DataLoader(Views(paths,cfg,GROUPS['local']),batch_size=8)
    outputs = []
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
        for x in loader:
            outputs.append(model(x.flatten(0,1).cuda()).float().cpu().numpy().reshape(len(x),4,175))
    local = np.concatenate(outputs)
    assert np.isfinite(local).all()
    crop_scores = crop[:,ci[seen]]
    disease_cols = di[seen]
    per_view = (crop_scores[:,None,:]+local[:,:,disease_cols]).argmax(2)
    mixed = crop_scores+.5*(disease[:,disease_cols]+local.mean(1)[:,disease_cols])
    chosen = mixed.argmax(1)
    votes = (per_view==chosen[:,None]).sum(1)
    two = np.partition(mixed,-2,axis=1)[:,-2:]
    return seen[chosen],votes,two[:,1]-two[:,0]


def delta(pred, previous, labels, group, override):
    return int(((pred==labels)&group&override).sum()-((previous==labels)&group&override).sum())


def main():
    torch.set_num_threads(4)
    out = HERE/'second_look'
    out.mkdir(exist_ok=True)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    cfg = p['baseline_config']
    pairs = sorted(p['pairs'],key=lambda r:r['pair_id'])
    ci,di = [np.array([r[k] for r in pairs]) for k in ('crop_id','disease_id')]
    rule = json.loads((HERE/'hybrid/selection.json').read_text())
    val = p['roles']['validation_seen']+p['roles']['validation_unseen']
    index = {r['filename']:i for i,r in enumerate(val)}
    with np.load(HERE/'hybrid/development_scores.npz') as z:
        order = np.array([index[name] for name in z['filenames']])
        clip = z['clip']
    assert len(set(order))==len(val)
    targets = np.array([r['composition_id'] for r in val])[order]
    is_seen = order<len(p['roles']['validation_seen'])
    seen = np.array(sorted({r['composition_id'] for r in p['roles']['fit']}))
    unseen = np.unique(targets[~is_seen])
    with np.load(p['baseline_validation_logits']) as z:
        crop,disease = z['crop'][order],z['disease'][order]
    gap,prev = route(clip,crop,disease,ci,di,seen,unseen,rule)
    widths = (.005,.01,.02,.03,.05)
    ids = np.flatnonzero((gap>0)&(gap<=max(widths)))
    assert len(ids)
    source = next(Path(k) for k in p['reference_hashes'] if Path(k).parent.name=='disease' and Path(k).name=='best_epoch_15.pt')
    assert sha256(source)==p['reference_hashes'][str(source)]
    candidate,votes,margin = second_look([val[order[i]]['legacy_cache_path'] for i in ids],cfg,source,
                                         crop[ids],disease[ids],ci,di,seen)
    grid = []
    for width in widths:
        for agreement in (3,4):
            for min_margin in (0.,.5,1.,2.):
                override = (gap[ids]<=width)&(votes>=agreement)&(margin>=min_margin)
                grid.append(dict(width=width,votes=agreement,min_margin=min_margin,overrides=int(override.sum()),
                    seen_gain=delta(candidate,prev[ids],targets[ids],is_seen[ids],override),
                    unseen_gain=delta(candidate,prev[ids],targets[ids],~is_seen[ids],override)))
    best = max((r for r in grid if r['unseen_gain']>=0 and r['seen_gain']>0),
               key=lambda r:(r['seen_gain'],-r['overrides']),default=None)
    selection = dict(chosen=best,grid=grid,development_candidates=len(ids))
    write_json(out/'selection.json',selection)
    print('Development',len(ids),best,flush=True)
    if best is None:
        return

    seen = np.array([r['pair_id'] for r in pairs if not r['is_unseen']])
    unseen = np.array([r['pair_id'] for r in pairs if r['is_unseen']])
    cache = BASE/'runs/cache/resized_256'
    scores = {}
    with np.load(REF/'factor_logits.npz') as logits:
        for split in ('test_seen','test_unseen'):
            previous = HERE/'seen_multiview/test_seen.npz' if split=='test_seen' else HERE/'seen_expert_official/test_unseen.npz'
            with np.load(previous) as b,np.load(HERE/'official'/f'anchored_{split}.npz') as c:
                names,targets = b['filenames'],b['labels']
                pred = b['new' if split=='test_seen' else 'old'].copy()
                routed = b['routed_seen']
                assert np.array_equal(names,c['filenames']) and np.array_equal(targets,c['labels'])
                crop,disease = logits[f'{split}_crop'],logits[f'{split}_disease']
                gap,clip_pred = route(c['scores'],crop,disease,ci,di,seen,unseen,rule)
            assert np.array_equal(gap<=0,routed) and np.array_equal(clip_pred[~routed],pred[~routed])
            old = pred.copy()
            ids = np.flatnonzero((gap>0)&(gap<=best['width']))
            overridden = 0
            if len(ids):
                candidate,votes,margin = second_look([cache/(Path(names[i]).stem+'.jpg') for i in ids],
                    cfg,REF/'disease_net.pt',crop[ids],disease[ids],ci,di,seen)
                override = (votes>=best['votes'])&(margin>=best['min_margin'])
                pred[ids[override]] = candidate[override]
                overridden = int(override.sum())
            scores[split] = dict(images=len(targets),old=int((old==targets).sum()),
                                 new=int((pred==targets).sum()),overrides=overridden)
            np.savez_compressed(out/f'{split}.npz',filenames=names,labels=targets,old=old,new=pred)
    assert scores['test_seen']['old']==1348 and scores['test_unseen']['old']==1134
    write_json(out/'results.json',dict(selection=selection,official=scores))
    print('Official',scores,flush=True)


if __name__=='__main__':
    main()
