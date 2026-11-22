"""Learn disease residuals on healthy same-crop ConvNeXt-Base features."""
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
sys.path.insert(0,str(next(parent for parent in HERE.parents if (parent/'dpd').is_dir())))
sys.path.insert(0, str(HERE.parent/'leafgan_v1'))
from classifier_multi import DETAIL, IndependentNet, features, transforms, composed, pd, seed_all, status, sha256, write_json


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 512), nn.LeakyReLU(.2),
                         nn.Linear(512, 512), nn.LeakyReLU(.2), nn.Linear(512, outputs))


def condition(crop, disease):
    return torch.cat((nn.functional.one_hot(crop,55), nn.functional.one_hot(disease,175)),1).float()


def main(seed, within_crop=False):
    run = HERE/('within_crop' if within_crop else 'runs')/f'seed_{seed}'
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
    selected = sorted(held)
    assert len(selected) == 20
    assert {r['crop_id'] for r in held.values()} <= {r['crop_id'] for r in fit}
    assert {r['disease_id'] for r in held.values()} <= {r['disease_id'] for r in fit}
    write_json(run/'selection.json',{'pairs':[held[k]['composition_name'] for k in selected]})
    _, transform = transforms(cfg)
    x,y = features(model,fit,transform,run,'fit')
    x,y = x.cuda(),y.cuda()
    crops = torch.tensor([r['crop_id'] for r in fit],device='cuda')
    same_crop = torch.zeros(55,175,device='cuda',dtype=torch.bool)
    same_crop[crops,y] = True
    mean,scale = x.mean(0),x.std(0).clamp_min(.01)
    normalized = (x-mean)/scale
    cond = condition(crops,y)
    healthy = torch.tensor([r['disease_name']=='healthy' for r in fit],device='cuda')
    pools_by_crop = {c:torch.where(healthy & (crops==c))[0] for c in crops.unique().tolist()}
    assert all(len(pools_by_crop[held[k]['crop_id']]) for k in selected)
    eligible = torch.tensor([i for i,r in enumerate(fit)
                            if r['disease_name']!='healthy' and len(pools_by_crop[r['crop_id']])],
                           device='cuda')
    write_json(run/'anchor_audit.json',{'eligible_disease_examples':len(eligible),
        'healthy_examples_by_target_pair':{held[k]['composition_name']:len(pools_by_crop[held[k]['crop_id']])
                                            for k in selected}})
    anchor_table = torch.zeros((55,max(map(len,pools_by_crop.values()))),dtype=torch.long,device='cuda')
    anchor_counts = torch.zeros(55,dtype=torch.long,device='cuda')
    for crop_id,pool in pools_by_crop.items():
        anchor_table[crop_id,:len(pool)] = pool
        anchor_counts[crop_id] = len(pool)

    def anchors(crop_ids):
        choices = (torch.rand(len(crop_ids),device='cuda')*anchor_counts[crop_ids]).long()
        return normalized[anchor_table[crop_ids,choices]]

    crop_probe = nn.Linear(1024,55).cuda()
    probe_opt = torch.optim.AdamW(crop_probe.parameters(),lr=.003)
    for _ in range(500):
        ids = torch.randint(len(y),(128,),device='cuda')
        probe_opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(crop_probe(normalized[ids]),crops[ids]).backward()
        probe_opt.step()
    crop_probe.eval().requires_grad_(False)
    gen,disc = mlp(1318,1024).cuda(),mlp(1254,1).cuda()

    def generate(anchor, labels):
        noise = torch.randn(len(anchor),64,device='cuda')
        return anchor + gen(torch.cat((anchor,noise,labels),1))
    go,do = (torch.optim.Adam(net.parameters(),lr=2e-4,betas=(.5,.999)) for net in (gen,disc))
    mse,ce = nn.MSELoss(),nn.CrossEntropyLoss()
    for step in range(1,1501):
        ids = eligible[torch.randint(len(eligible),(128,),device='cuda')]
        c = cond[ids]
        anchor = anchors(crops[ids])
        fake = generate(anchor,c)
        real_logits = disc(torch.cat((normalized[ids],c),1))
        fake_logits = disc(torch.cat((fake.detach(),c),1))
        dl = .5*(mse(real_logits,torch.ones_like(real_logits))+mse(fake_logits,torch.zeros_like(fake_logits)))
        do.zero_grad(set_to_none=True)
        dl.backward()
        do.step()
        disc.requires_grad_(False)
        logits = disc(torch.cat((fake,c),1))
        disease_logits = model.head(fake*scale+mean)
        gl = mse(logits,torch.ones_like(logits))+.2*ce(disease_logits,y[ids])
        if within_crop:
            local = disease_logits.masked_fill(~same_crop[crops[ids]],float('-inf'))
            gl = gl + .2*ce(local,y[ids])
        gl = gl + .2*ce(crop_probe(fake),crops[ids]) + .05*(fake-anchor).abs().mean()
        go.zero_grad(set_to_none=True)
        gl.backward()
        go.step()
        disc.requires_grad_(True)
        if step%100==0:
            assert torch.isfinite(gl) and torch.isfinite(dl)
            status(run,'training_gan',step=step,total_steps=1500,generator=float(gl.detach()),discriminator=float(dl.detach()))
    labels = y.cpu().numpy()
    torch.save({'generator':gen.state_dict(),'mean':mean,'scale':scale},run/'generator.pt')
    vx,_ = features(model,seen+unseen,transform,run,'validation')
    vx = vx.cuda()
    model.backbone.cpu()
    pairs = pd.DataFrame(p['pairs'])
    targets = np.array([r['composition_id'] for r in seen+unseen])
    is_seen = np.arange(len(targets))<len(seen)
    with np.load(p['baseline_validation_logits']) as bank:
        crop = bank['crop'].copy()
        reference = composed(crop,bank['disease'],pairs).argmax(1)

    def metrics(pred):
        hit = pred == targets
        per_pair = {held[k]['composition_name']: {'correct':int(hit[targets==k].sum()),
                     'images':int((targets==k).sum())} for k in sorted(held)}
        return {'seen':int(hit[is_seen].sum()),'held_out':int(hit[~is_seen].sum()),
                'held_out_macro':float(np.mean([v['correct']/v['images'] for v in per_pair.values()])),
                'per_pair':per_pair}

    results = {'baseline':metrics(reference)}
    trained_heads = {}
    for count in (192,):
        total = count*len(selected)

        seed_all(seed)
        sy = torch.tensor([held[k]['disease_id'] for k in selected for _ in range(count)],device='cuda')
        sc = torch.tensor([held[k]['crop_id'] for k in selected for _ in range(count)],device='cuda')
        with torch.no_grad():
            synthetic = generate(anchors(sc),condition(sc,sy))*scale+mean
        assert torch.isfinite(synthetic).all()
        pools = {d:np.flatnonzero(labels==d) for d in set(sy.tolist())}
        control_rng = np.random.default_rng(seed)
        ids = np.array([control_rng.choice(pools[d]) for d in sy.tolist()])
        train_y = torch.cat((y,sy))
        status(run,'comparing_heads',seed=seed,vectors_per_pair=count)
        for name,extra in [('gan',synthetic),('real_only',x[ids])]:
            seed_all(seed)
            head = deepcopy(model.head).requires_grad_(True)
            optimizer = torch.optim.AdamW(head.parameters(),lr=1e-4)
            train_x = torch.cat((x,extra))
            history = []
            for epoch in range(1,4):
                head.train()
                for batch in torch.randperm(len(train_y),device='cuda').split(64):
                    optimizer.zero_grad(set_to_none=True)
                    nn.functional.cross_entropy(head(train_x[batch]),train_y[batch],label_smoothing=.1).backward()
                    optimizer.step()
                head.eval()
                with torch.no_grad():
                    pred = composed(crop,head(vx).cpu().numpy(),pairs).argmax(1)
                history.append(metrics(pred))
            results[f'{count}_{name}'] = history
            torch.save(head.state_dict(),run/f'{count}_{name}_head.pt')
            if within_crop and name=='gan':
                trained_heads['within_crop'] = head
        write_json(run/'results.json',results)
    if within_crop:
        control = deepcopy(model.head)
        control.load_state_dict(torch.load(HERE/'runs'/f'seed_{seed}'/'192_gan_head.pt',
                                           map_location='cpu',weights_only=True))
        control.eval()
        trained_heads['original'] = control
        selected_pairs = np.array(sorted(set(fit_counts)|set(held)))
        ci,di = [pairs[key].to_numpy() for key in ('crop_id','disease_id')]
        seen_cols = np.isin(selected_pairs,list(fit_counts))
        tracking = {}
        scopes = (is_seen,~is_seen)
        for name,head in trained_heads.items():
            with torch.inference_mode():
                disease = head(vx).cpu().numpy()
            ranked = crop[:,ci[selected_pairs]] + disease[:,di[selected_pairs]]
            local = np.where(ci[selected_pairs][None,:]==ci[targets][:,None],ranked,-np.inf)
            a,b = ranked[:,seen_cols],ranked[:,~seen_cols]
            seen_hit = selected_pairs[seen_cols][a.argmax(1)]==targets
            unseen_hit = selected_pairs[~seen_cols][b.argmax(1)]==targets
            gap = a.max(1)-b.max(1)
            tracking[name] = (gap,seen_hit,unseen_hit)
            crop_hit = selected_pairs[local.argmax(1)]==targets
            disease_hit = disease.argmax(1)==di[targets]
            results[name] = dict(true_crop=[int((crop_hit&m).sum()) for m in scopes],
                                 disease_only=[int((disease_hit&m).sum()) for m in scopes],
                                 zero_offset=[int((seen_hit&(gap>=0)).sum()),int((unseen_hit&(gap<0)).sum())])
        floor = results['original']['zero_offset'][0]
        for name,(gap,s,u) in tracking.items():
            if s.sum() >= floor > 0:
                cutoff = np.sort(gap[s])[-floor]
                results[name]['matched_seen_diagnostic'] = [int((s&(gap>=cutoff)).sum()),int((u&(gap<cutoff)).sum())]
        results['images'] = [len(seen),len(unseen)]
        results['within_crop_weight'] = .2
        write_json(run/'comparison.json',results)
        print({key:results[key] for key in ('images','original','within_crop')},flush=True)
    status(run,'complete',seed=seed,budgets=[3840])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument('--within-crop',action='store_true')
    args = parser.parse_args()
    main(args.seed,args.within_crop)

