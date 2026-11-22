"""Conditional feature GAN on frozen ConvNeXt-Base development features."""
import argparse
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
    ref = next(k for k in p['reference_hashes'] if Path(k).parent.name=='disease' and Path(k).name=='best_epoch_15.pt')
    for path in (ref,p['baseline_validation_logits']):
        assert sha256(Path(path)) == p['reference_hashes'][path]
    cfg = p['baseline_config']
    model = IndependentNet(cfg,'disease')
    model.load_state_dict(torch.load(ref,map_location='cpu',weights_only=True)['model_state'])
    model.cuda().eval()
    model.requires_grad_(False)
    fit, seen, unseen = (p['roles'][role] for role in ('fit','validation_seen','validation_unseen'))
    fit_counts = Counter(r['composition_id'] for r in fit)
    held = {r['composition_id']:r for r in unseen}
    assert not (held.keys() & fit_counts.keys())
    crop_counts = Counter(r['crop_id'] for r in fit)
    disease_counts = Counter(r['disease_id'] for r in fit)
    eligible = [k for k,r in sorted(held.items()) if crop_counts[r['crop_id']]>=10 and disease_counts[r['disease_id']]>=10]
    rng = np.random.default_rng(42)
    selected = sorted(rng.choice(eligible,len(eligible)//2,replace=False).tolist())
    assert selected and len(selected)<len(held)
    write_json(run/'selection.json',{'generated':[held[k]['composition_name'] for k in selected],
                                      'untouched':[r['composition_name'] for k,r in sorted(held.items()) if k not in selected]})
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
    sy = torch.tensor([held[k]['disease_id'] for k in selected for _ in range(192)],device='cuda')
    sc = torch.tensor([held[k]['crop_id'] for k in selected for _ in range(192)],device='cuda')
    with torch.no_grad():
        synthetic = gen(torch.cat((torch.randn(len(sy),64,device='cuda'),condition(sc,sy)),1))*scale+mean
    assert torch.isfinite(synthetic).all()
    labels = y.cpu().numpy()
    pools = {d: np.flatnonzero(labels == d) for d in set(sy.tolist())}
    control_rng = np.random.default_rng(seed)
    control_ids = np.array([control_rng.choice(pools[d]) for d in sy.tolist()])
    torch.save({'generator':gen.state_dict(),'mean':mean,'scale':scale,'synthetic':synthetic,'labels':sy},run/'generator.pt')
    vx,_ = features(model,seen+unseen,transform,run,'validation')
    vx = vx.cuda()
    model.backbone.cpu()
    pairs = pd.DataFrame(p['pairs'])
    targets = np.array([r['composition_id'] for r in seen+unseen])
    is_seen = np.arange(len(targets))<len(seen)
    generated = ~is_seen & np.isin(targets,selected)
    masks = {'seen':is_seen,'held_out':~is_seen,'synthesized_pairs':generated,'untouched_pairs':~is_seen & ~generated}
    def metrics(pred):
        return {key:{'correct':int((pred[mask]==targets[mask]).sum()),'images':int(mask.sum())} for key,mask in masks.items()}
    with np.load(p['baseline_validation_logits']) as bank:
        crop = bank['crop'].copy()
        results = {'baseline':metrics(composed(crop,bank['disease'],pairs).argmax(1))}
    for budget in (32,96,192):
        indices = np.concatenate([np.arange(i*192,i*192+budget) for i in range(len(selected))])
        status(run,'comparing_heads',seed=seed,vectors_per_pair=budget)
        train_y = torch.cat((y,sy[indices]))
        for name,extra in [('gan',synthetic[indices]),('real_only',x[control_ids[indices]])]:
            seed_all(seed)
            head = deepcopy(model.head).requires_grad_(True)
            optimizer = torch.optim.AdamW(head.parameters(),lr=1e-4)
            train_x = torch.cat((x,extra))
            history = []
            for epoch in range(1,4):
                head.train()
                for ids in torch.randperm(len(train_y), device='cuda').split(64):
                    optimizer.zero_grad(set_to_none=True)
                    nn.functional.cross_entropy(head(train_x[ids]),train_y[ids],label_smoothing=.1).backward()
                    optimizer.step()
                head.eval()
                with torch.no_grad():
                    pred = composed(crop,head(vx).cpu().numpy(),pairs).argmax(1)
                history.append(metrics(pred))
            results[f"{budget}_{name}"] = history
            torch.save(head.state_dict(),run/f'{budget}_{name}_head.pt')
        write_json(run/'results.json',results)
    status(run,'complete',seed=seed,synthesized_pairs=len(selected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    main(parser.parse_args().seed)
