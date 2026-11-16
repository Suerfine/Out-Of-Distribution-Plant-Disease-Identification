"""Compare a within-crop disease loss with matched regular fine-tuning."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from seen_expert import HERE, DETAIL, PairImages, IndependentNet, transforms, sha256, seed_all, status, write_json
from independent_cnn_baseline.models import optimizer
from storage import CheckpointStore


WORKSPACE = HERE.parents[1]
BASE = WORKSPACE/'Out-Of-Distribution-Plant-Disease-Identification'
BASELINE = BASE/'runs/baseline/independent_cnn_baseline_convnext_base_seed42'
DATA = WORKSPACE/'Plant_Disease'
EPOCHS = 3
BATCH = 4
ACCUM = 8


def official_rows(split):
    cache = BASE/'runs/cache/resized_256'
    with (DATA/f'DPD_Small_{split}.csv').open(encoding='utf-8-sig',newline='') as f:
        return [dict(filename=r[0],path=str(cache/(Path(r[0]).stem+'.jpg')),
                     crop_id=int(r[1]),disease_id=int(r[2]),composition_id=int(r[3]))
                for r in csv.reader(f)]


def predict(model, rows, tf, out):
    loader = DataLoader(PairImages(rows,tf),batch_size=16,num_workers=2)
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
        logits = torch.cat([model(x.cuda()).float().cpu() for x,_,_,_ in loader]).numpy()
    if not np.isfinite(logits).all():
        raise RuntimeError(f'Non-finite predictions in {out}')
    return logits


def fit_model(rows, base, source, weight, epochs, out, evaluate=None, scope='development', *,
              hard_margin=None, backbone_anchor=0.):
    out.mkdir(exist_ok=True,parents=True)
    seed_all(42)
    cfg = dict(weight=weight,epochs=epochs,scope=scope,source_sha256=sha256(source),
               fit_images=len(rows),microbatch=BATCH,accumulation=ACCUM)
    if hard_margin is not None:
        cfg['hard_margin'] = hard_margin
    if backbone_anchor:
        cfg['backbone_anchor'] = backbone_anchor
    write_json(out/'recipe.json',cfg)
    fingerprint = sha256(Path(__file__)) + sha256(source) + json.dumps(cfg,sort_keys=True)
    store = CheckpointStore(out,fingerprint,6*1024**3)
    model = IndependentNet(base,'disease')
    checkpoint = torch.load(source,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
    del checkpoint
    model.cuda()
    model.backbone.set_grad_checkpointing(True)
    anchors = [(p, p.detach().clone()) for p in model.backbone.parameters()
               if backbone_anchor and p.requires_grad]
    opt = optimizer(model,base)
    scaler = torch.amp.GradScaler('cuda')
    tf,_ = transforms(base)
    loader = DataLoader(PairImages(rows,tf),batch_size=BATCH,shuffle=True,num_workers=2)
    allow = torch.zeros(55,175,dtype=torch.bool)
    allow[[r['crop_id'] for r in rows],[r['disease_id'] for r in rows]] = True
    allow = allow.cuda()
    saved = store.load()
    history = []
    if saved is not None:
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        history = saved['history']
        del saved
    for epoch in range(len(history)+1,epochs+1):
        seed_all(42+epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        for step,(x,y,c,_) in enumerate(loader,1):
            y,c = y.cuda(),c.cuda()
            with torch.autocast('cuda',dtype=torch.float16):
                logits = model(x.cuda())
                loss = F.cross_entropy(logits,y,label_smoothing=.1)
                if weight:
                    same_crop = logits.masked_fill(~allow[c],float('-inf'))
                    if hard_margin is None:
                        extra = F.cross_entropy(same_crop,y)
                    else:
                        target = logits.float().gather(1,y[:,None]).squeeze(1)
                        negatives = same_crop.float().scatter(1,y[:,None],float('-inf'))
                        extra = F.relu(hard_margin + negatives.max(1).values - target).mean()
                    loss = loss + weight*extra
            size = min(BATCH*ACCUM,len(rows)-((step-1)//ACCUM)*BATCH*ACCUM)
            scaler.scale(loss*len(x)/size).backward()
            if step%ACCUM==0 or step==len(loader):
                scaler.unscale_(opt)
                # Add the gradient of (lambda / 2) * ||backbone - initial||² once per update.
                with torch.no_grad():
                    for param, initial in anchors:
                        if param.grad is not None:
                            param.grad.add_(param - initial, alpha=backbone_anchor)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if step==1 or step%100==0:
                status(out,'training',epoch=epoch,step=step,total_steps=len(loader))
        row = {'epoch':epoch}
        if anchors:
            with torch.no_grad():
                row['backbone_distance'] = float(torch.stack([
                    torch.linalg.vector_norm(param - initial).square()
                    for param, initial in anchors
                ]).sum().sqrt())
        if evaluate:
            status(out,'validation',epoch=epoch)
            row.update(evaluate(model,out))
        history.append(row)
        store.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                        history=history,epoch=epoch,cursor=0,stage='finetune',fingerprint=fingerprint))
        write_json(out/'history.json',history)
        status(out,'epoch_complete',**row)
        if (out/'STOP.request').exists() or (out.parent/'STOP.request').exists():
            status(out,'paused',epoch=epoch)
            return None,history
    status(out,'complete',epochs=epochs)
    return model,history


def main():
    torch.set_num_threads(4)
    out = HERE/'within_crop_disease'
    out.mkdir(exist_ok=True)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    base = p['baseline_config']
    fit,seen = (p['roles'][r] for r in ('fit','validation_seen'))
    assert len(fit)==5798 and len(seen)==862
    source = next(Path(k) for k in p['reference_hashes'] if Path(k).parent.name=='disease' and Path(k).name=='best_epoch_15.pt')
    assert sha256(source)==p['reference_hashes'][str(source)]
    seen_pairs = np.array(sorted({r['composition_id'] for r in fit}))
    pairs = sorted(p['pairs'],key=lambda r:r['pair_id'])
    ci,di = [np.array([r[k] for r in pairs]) for k in ('crop_id','disease_id')]
    target = np.array([r['composition_id'] for r in seen])
    _,test_tf = transforms(base)
    with np.load(p['baseline_validation_logits']) as bank:
        scores = bank['crop'][:len(seen),ci[seen_pairs]]
        ref = scores+bank['disease'][:len(seen),di[seen_pairs]]
    baseline = int((seen_pairs[ref.argmax(1)]==target).sum())
    assert baseline==773

    def evaluate(model,arm):
        logits = predict(model,seen,test_tf,arm)
        pred = seen_pairs[(scores+logits[:,di[seen_pairs]]).argmax(1)]
        return {'correct':int((pred==target).sum()),'images':len(target)}

    completed = {}
    for weight in (0.,.5,1.):
        model,hist = fit_model(fit,base,source,weight,EPOCHS,out/f'weight_{weight}',evaluate)
        if model is None:
            return
        completed[str(weight)] = hist
        del model
        torch.cuda.empty_cache()
    candidates = [(row['correct'],-row['epoch'],-float(w)) for w,h in completed.items()
                  if float(w)>0 for row in h]
    best,epoch,weight = max(candidates)
    epoch,weight = -epoch,-weight
    control = max(row['correct'] for row in completed['0.0'])
    choice = {'weight':weight,'epochs':epoch,'correct':best,'baseline_correct':baseline,
              'control_best_correct':control,'all_development':completed}
    write_json(out/'selection.json',choice)
    print('Development comparison',choice,flush=True)
    if best <= max(control,baseline):
        status(out,'complete',decision='no development improvement for official refit')
        return

    full = official_rows('train')
    assert len(full)==8000 and len({r['composition_id'] for r in full})==323
    original = BASELINE/'disease_net.pt'
    model,_ = fit_model(full,base,original,weight,epoch,out/'official',scope='full-data refit')
    if model is None:
        return
    rows = official_rows('test_seen')
    assert len(rows)==2000 and {r['filename'] for r in full}.isdisjoint(r['filename'] for r in rows)
    names = [r['filename'] for r in rows]
    y = np.array([r['composition_id'] for r in rows])
    with np.load(HERE/'seen_expert_official/test_seen.npz') as bank:
        assert np.array_equal(bank['filenames'],names) and np.array_equal(bank['labels'],y)
        old,routed = bank['old'],bank['routed_seen']
    codes = np.array(sorted({r['composition_id'] for r in full}))
    with np.load(BASELINE/'factor_logits.npz') as bank:
        crop_score = bank['test_seen_crop'][routed][:,ci[codes]]
        old_head = bank['test_seen_disease'][routed][:,di[codes]]
    assert np.array_equal(codes[(crop_score+old_head).argmax(1)],old[routed])
    logits = predict(model,[r for r,route in zip(rows,routed) if route],test_tf,out)
    new = old.copy()
    new[routed] = codes[(crop_score+logits[:,di[codes]]).argmax(1)]
    with np.load(HERE/'seen_expert_official/test_unseen.npz') as bank:
        unseen = int((bank['labels']==bank['old']).sum())
    assert unseen==1134
    result = dict(seen_correct=int((new==y).sum()),old_seen_correct=int((old==y).sum()),
                  unseen_correct=unseen,seen_images=2000,unseen_images=1448)
    np.savez_compressed(out/'official/test_seen.npz',filenames=names,labels=y,old=old,new=new,routed_seen=routed)
    write_json(out/'official/results.json',result)
    status(out,'complete',official=result)


if __name__=='__main__':
    main()
