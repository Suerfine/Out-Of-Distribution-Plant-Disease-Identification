"""Conditional feature GAN on frozen ConvNeXt-Base development features."""
import argparse
import csv
from collections import Counter
from copy import deepcopy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'leafgan_v1'))
from classifier_multi import DETAIL, IndependentNet, features, transforms, composed, pd, seed_all, status, sha256, write_json


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 512), nn.LeakyReLU(.2),
                         nn.Linear(512, 512), nn.LeakyReLU(.2), nn.Linear(512, outputs))


def condition(crop, disease):
    return torch.cat((nn.functional.one_hot(crop,55), nn.functional.one_hot(disease,175)),1).float()


def main(seed):
    run = HERE/'runs'/f'seed_{seed}'
    run.mkdir(exist_ok=True, parents=True)
    seed_all(seed)
    torch.set_num_threads(4)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    project = HERE.parents[1]/'Out-Of-Distribution-Plant-Disease-Identification'
    baseline = project/'runs/baseline/independent_cnn_baseline_convnext_base_seed42'
    ref = baseline/'disease_net.pt'
    cfg = json.loads((baseline/'config.json').read_text())
    model = IndependentNet(cfg,'disease')
    model.load_state_dict(torch.load(ref,map_location='cpu',weights_only=True))
    model.cuda().eval().requires_grad_(False)
    cache = project/'runs/cache/resized_256'
    def row(filename,crop,disease,pair):
        return {'filename':filename,'crop_id':int(crop),'disease_id':int(disease),
                'composition_id':int(pair),'path':str(cache/(Path(filename).stem+'.jpg'))}
    manifest = next(Path(k) for k in p['reference_hashes'] if Path(k).name=='all_train.csv')
    assert sha256(manifest)==p['reference_hashes'][str(manifest)]
    with manifest.open(encoding='utf-8-sig',newline='') as f:
        fit = [row(r['file'],r['crop_id'],r['disease_id'],r['pair_id']) for r in csv.DictReader(f)]
    assert len(fit)==8000 and len({r['filename'] for r in fit})==8000
    held = {r['pair_id']:r for r in p['pairs'] if r['is_unseen']}
    assert len(held)==10 and not set(held).intersection(r['composition_id'] for r in fit)
    assert {r['crop_id'] for r in held.values()} <= {r['crop_id'] for r in fit}
    assert {r['disease_id'] for r in held.values()} <= {r['disease_id'] for r in fit}
    write_json(run/'selection.json',{'generated':[r['pair'] for r in held.values()],
        'vectors_per_pair':192,'epochs':3,'gan_steps':1500,'checkpoint_sha256':sha256(ref),
        'candidate_source':'prepared task ontology'})
    _, transform = transforms(cfg)
    x,y = features(model,fit,transform,run,'fit')
    x,y = x.cuda(),y.cuda()
    crops = torch.tensor([r['crop_id'] for r in fit],device='cuda')
    mean,scale = x.mean(0),x.std(0).clamp_min(.01)
    normalized = (x-mean)/scale
    cond = condition(crops,y)
    gen,disc = mlp(294,1024).cuda(),mlp(1254,1).cuda()
    go,do = (torch.optim.Adam(net.parameters(),lr=2e-4,betas=(.5,.999)) for net in (gen,disc))
    mse,ce = nn.MSELoss(),nn.CrossEntropyLoss()
    for step in range(1,1501):
        ids = torch.randint(len(y),(128,),device='cuda')
        c = cond[ids]
        fake = gen(torch.cat((torch.randn(128,64,device='cuda'),c),1))
        real_logits = disc(torch.cat((normalized[ids],c),1))
        fake_logits = disc(torch.cat((fake.detach(),c),1))
        dl = .5*(mse(real_logits,torch.ones_like(real_logits))+mse(fake_logits,torch.zeros_like(fake_logits)))
        do.zero_grad(set_to_none=True)
        dl.backward()
        do.step()
        disc.requires_grad_(False)
        logits = disc(torch.cat((fake,c),1))
        gl = mse(logits,torch.ones_like(logits))+.2*ce(model.head(fake*scale+mean),y[ids])
        go.zero_grad(set_to_none=True)
        gl.backward()
        go.step()
        disc.requires_grad_(True)
        if step%100==0:
            assert torch.isfinite(gl) and torch.isfinite(dl)
            status(run,'training_gan',step=step,total_steps=1500,generator=float(gl.detach()),discriminator=float(dl.detach()))
    labels = y.cpu().numpy()
    selected = sorted(held)
    seed_all(seed)
    sy = torch.tensor([held[k]['disease_id'] for k in selected for _ in range(192)],device='cuda')
    sc = torch.tensor([held[k]['crop_id'] for k in selected for _ in range(192)],device='cuda')
    with torch.no_grad():
        synthetic = gen(torch.cat((torch.randn(1920,64,device='cuda'),condition(sc,sy)),1))*scale+mean
    assert torch.isfinite(synthetic).all()
    torch.save({'generator':gen.state_dict(),'mean':mean,'scale':scale},run/'generator.pt')
    pools = {d:np.flatnonzero(labels==d) for d in set(sy.tolist())}
    rng = np.random.default_rng(seed)
    ids = [int(rng.choice(pools[d])) for d in sy.tolist()]
    train_y = torch.cat((y,sy))
    heads = {}
    for name,extra in [('gan',synthetic),('real_only',x[ids])]:
        status(run,'training_head',arm=name)
        seed_all(seed)
        head = deepcopy(model.head).requires_grad_(True)
        optimizer = torch.optim.AdamW(head.parameters(),lr=1e-4)
        train_x = torch.cat((x,extra))
        for epoch in range(3):
            head.train()
            for batch in torch.randperm(len(train_y),device='cuda').split(64):
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(head(train_x[batch]),train_y[batch],label_smoothing=.1).backward()
                optimizer.step()
        heads[name] = head.eval()
        torch.save(head.state_dict(),run/f'{name}_head.pt')

    pairs = sorted(p['pairs'],key=lambda r:r['pair_id'])
    assert [r['pair_id'] for r in pairs]==list(range(333))
    ci = [r['crop_id'] for r in pairs]
    di = [r['disease_id'] for r in pairs]
    results = {}
    with np.load(baseline/'factor_logits.npz') as factors, np.load(baseline/'scores.npz') as saved:
        for split,count,expected in [('test_seen',2000,1476),('test_unseen',1448,203)]:
            path = HERE.parents[1]/'Plant_Disease'/f'DPD_Small_{split}.csv'
            with path.open(encoding='utf-8-sig',newline='') as f:
                test = [row(*r[:4]) for r in csv.reader(f)]
            targets = np.array([r['composition_id'] for r in test])
            assert len(test)==count and np.array_equal(targets,saved[f'{split}_labels'])
            assert not {r['filename'] for r in fit}.intersection(r['filename'] for r in test)
            crop = factors[f'{split}_crop']
            old = factors[f'{split}_disease']
            def predict(disease):
                return (crop[:,ci]+disease[:,di]).argmax(1)
            reference = predict(old)
            assert (reference==targets).sum()==expected
            vx,_ = features(model,test,transform,run,split)
            preds = {'baseline':reference}
            with torch.inference_mode():
                for name,head in heads.items():
                    logits = head(vx.cuda()).cpu().numpy()
                    assert np.isfinite(logits).all()
                    preds[name] = predict(logits)
            np.savez_compressed(run/f'{split}.npz',labels=targets,filenames=[r['filename'] for r in test],**preds)
            results[split] = {name:{'correct':int((pred==targets).sum()),'images':count,
                'per_pair':{pairs[k]['pair']:{'correct':int(((pred==targets)&(targets==k)).sum()),
                            'images':int((targets==k).sum())} for k in np.unique(targets)}}
                for name,pred in preds.items()}
    write_json(run/'results.json',results)
    status(run,'complete',seed=seed,synthetic_vectors=1920)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    main(parser.parse_args().seed)

