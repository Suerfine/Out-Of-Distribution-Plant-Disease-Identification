"""Evaluate the validation-selected crop disagreement rule."""
import json
import numpy as np
import torch
from second_look import HERE, DETAIL, BASE, REF, route, second_look, write_json


def main():
    torch.set_num_threads(4)
    out = HERE/'crop_veto'
    chosen = json.loads((out/'selection.json').read_text())['chosen']
    assert chosen['seen_gain']>0 and chosen['held_out_loss']==0
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    rule = json.loads((HERE/'hybrid/selection.json').read_text())
    pairs = sorted(p['pairs'],key=lambda r:r['pair_id'])
    ci,di = [np.array([r[k] for r in pairs]) for k in ('crop_id','disease_id')]
    seen = np.array([r['pair_id'] for r in pairs if not r['is_unseen']])
    unseen = np.array([r['pair_id'] for r in pairs if r['is_unseen']])
    cache = BASE/'runs/cache/resized_256'
    results = {}
    with np.load(REF/'factor_logits.npz') as logits:
        for split in ('test_seen','test_unseen'):
            path = HERE/'seen_multiview/test_seen.npz' if split=='test_seen' else HERE/'seen_expert_official/test_unseen.npz'
            with np.load(path) as b,np.load(HERE/'official'/f'anchored_{split}.npz') as c:
                names,y,routed = [b[key] for key in ('filenames','labels','routed_seen')]
                old = b['new' if split=='test_seen' else 'old']
                assert np.array_equal(names,c['filenames']) and np.array_equal(y,c['labels'])
                crop,dis = logits[f'{split}_crop'],logits[f'{split}_disease']
                distance,clip_pred = route(c['scores'],crop,dis,ci,di,seen,unseen,rule)
            assert np.array_equal(distance<=0,routed) and np.array_equal(old[~routed],clip_pred[~routed])
            margin = crop.max(1)-crop[np.arange(len(crop)),ci[clip_pred]]
            ids = np.flatnonzero((distance>0)&(distance<=chosen['width'])&(margin>=chosen['crop_margin']))
            new = old.copy()
            if len(ids):
                paths = [cache/(name.rsplit('.',1)[0]+'.jpg') for name in names[ids]]
                new[ids],_,_ = second_look(paths,p['baseline_config'],REF/'disease_net.pt',crop[ids],dis[ids],ci,di,seen)
            results[split] = {'images':len(y),'old':int((old==y).sum()),'new':int((new==y).sum()),'overrides':len(ids)}
            if split=='test_unseen':
                assert np.all(new[ids]!=y[ids])
            np.savez_compressed(out/f'{split}.npz',filenames=names,labels=y,old=old,new=new,overrides=ids)
    assert results['test_seen']['old']==1348 and results['test_unseen']['old']==1134
    write_json(out/'official.json',results)
    print(results,flush=True)


if __name__=='__main__':
    main()
