"""Compare standard and crop-anchored f-CLSWGAN on fixed CLIP development splits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT = WORKSPACE/'Out-Of-Distribution-Plant-Disease-Identification'
sys.path.insert(0,str(PROJECT))
os.environ.setdefault('DPD_ROOT',str(WORKSPACE/'Plant_Disease'))

from dpd.clip_utils import clip_transforms, load_clip
from dpd.data import COLUMNS
from dpd.feature_gen import Critic, Generator, fit_classifier, gradient_penalty
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split
from train import train_lora
from train.train_feature_gen import text_conditions, seed_everything


class AnchoredGenerator(Generator):
    def __init__(self, cond_dim, noise_dim, hidden, width):
        super().__init__(cond_dim+width,noise_dim,hidden,width,False)

    def forward(self, cond, anchor):
        delta = super().forward(torch.cat((cond,anchor),dim=1))
        return F.normalize(anchor+delta,dim=1)


def train_generator(x,y,conditions,aux,options,anchor_pool=None):
    anchored = anchor_pool is not None
    width = x.shape[1]
    args = (conditions.shape[1],options['noise_dim'],options['hidden'],width)
    gen = (AnchoredGenerator(*args) if anchored else Generator(*args,True)).to(x.device)
    disc = Critic(width,conditions.shape[1],options['hidden']).to(x.device)
    opt_g,opt_d = (torch.optim.Adam(net.parameters(),lr=options['lr'],betas=tuple(options['betas']))
                   for net in (gen,disc))
    aux.requires_grad_(False).eval()
    bs = options['batch_size']

    def draw():
        ids = torch.randint(len(x),(bs,),device=x.device)
        return x[ids], y[ids], conditions[y[ids]]

    def generate(c,labels):
        return gen(c,anchor_pool(labels)) if anchored else gen(c)

    for epoch in range(1,options['epochs']+1):
        for _ in range((len(x)+bs-1)//bs):
            for _ in range(options['n_critic']):
                real, labels, cond = draw()
                with torch.no_grad():
                    fake = generate(cond,labels)
                loss_d = (disc(fake,cond).mean()-disc(real,cond).mean()
                          +options['lambda_gp']*gradient_penalty(disc,real,fake,cond))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
            _,labels,cond = draw()
            fake = generate(cond,labels)
            loss_g = -disc(fake,cond).mean()+options['cls_weight']*F.cross_entropy(aux(fake),labels)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
        if not torch.isfinite(loss_d) or not torch.isfinite(loss_g):
            raise RuntimeError('Non-finite GAN loss')
        print(f'GAN epoch {epoch}/{options["epochs"]} anchored={anchored} G={loss_g.item():.4f}',flush=True)
    return generate


def build_anchor_pool(x, fit, pairs):
    crops = torch.as_tensor(fit.crop_id.to_numpy(),device=x.device)
    healthy = torch.as_tensor((fit.disease=='healthy').to_numpy(),device=x.device)
    pair_crop = torch.as_tensor(pairs.crop_id.to_numpy(),device=x.device)
    pools = {c:torch.where(healthy & (crops==c))[0] for c in sorted(set(pairs.crop_id))}
    # A zero anchor lets the branch fall back to label-conditioned generation
    # when this split has no real healthy fitting example for the requested crop.
    table = torch.full((len(pools),max(1,max(map(len,pools.values())))),len(x),device=x.device,dtype=torch.long)
    counts = torch.ones(len(pools),device=x.device,dtype=torch.long)
    for c,pool in pools.items():
        if len(pool):
            table[c,:len(pool)] = pool
            counts[c] = len(pool)
    padded = torch.cat((x,torch.zeros_like(x[:1])))

    def sample(labels):
        c = pair_crop[labels]
        j = (torch.rand(len(c),device=x.device)*counts[c]).long()
        return padded[table[c,j]]
    return sample,{c:len(v) for c,v in pools.items()}


def smoke():
    torch.manual_seed(42)
    device = 'cuda'
    x = F.normalize(torch.randn(24,16,device=device),dim=1)
    y = torch.arange(24,device=device)%3
    cond = torch.randn(3,8,device=device)
    aux = nn.Linear(16,3).to(device)
    opts = dict(noise_dim=8,hidden=32,lr=.0001,betas=[.5,.999],batch_size=8,
                n_critic=2,lambda_gp=10.,cls_weight=.01,epochs=1)
    for pool in (None,lambda ids:x[ids]):
        g = train_generator(x,y,cond,aux,opts,pool)
        with torch.no_grad():
            sample = g(cond,torch.arange(3,device=device))
            assert sample.shape == (3,16) and torch.isfinite(sample).all()
            torch.testing.assert_close(sample.norm(dim=1),torch.ones(3,device=device))
    print('Both GAN variants passed a GPU update and synthesis check.',flush=True)


def main():
    torch.set_num_threads(4)
    base = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())
    lora = json.loads((PROJECT/'config/vlm/v10.3_text_anchor_three_splits_config.json').read_text())
    weights = HERE/'ViT-B-16.pt'
    with weights.open('rb') as f:
        digest = hashlib.file_digest(f,'sha256').hexdigest()
    if digest != '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f':
        raise RuntimeError('Unexpected OpenAI CLIP checkpoint checksum')
    state_path = HERE/'ViT-B-16-state.pt'
    if not state_path.exists():
        state = torch.jit.load(str(weights),map_location='cpu').state_dict()
        for key in ('input_resolution','context_length','vocab_size'):
            state.pop(key,None)
        temp = state_path.with_suffix('.tmp')
        torch.save(state,temp)
        temp.replace(state_path)
    for cfg in (base,lora):
        cfg['model']['pretrained'] = str(state_path)
    train = pd.read_csv(Path(os.environ['DPD_ROOT'])/'DPD_Small_train.csv',header=None,names=COLUMNS)
    train['row'] = np.arange(len(train))
    train['source'] = train.file.str.split('_').str[0]
    d = WORKSPACE/'Disease_Transfer_Experiments/cnn_detail_base_v1/runs/development_seed42/prepared.json'
    pairs = pd.DataFrame(json.loads(d.read_text())['pairs']).sort_values('pair_id').reset_index(drop=True)
    assert len(train)==8000 and pairs.pair_id.tolist()==list(range(333))
    hidden = torch.as_tensor(pairs.is_unseen.to_numpy(),device='cuda')
    clip,_,tokenizer = load_clip(base['model'],'cuda')
    prompt,cond = [torch.as_tensor(a,device='cuda') for a in text_conditions(base,clip,tokenizer,pairs,'cuda')]
    del clip
    v = base['validation']
    output = HERE/'runs'
    output.mkdir(exist_ok=True)
    for seed in (42,43,44):
        run = output/f'seed_{seed}'
        run.mkdir(exist_ok=True)
        result_path = run/'results.json'
        if result_path.is_file():
            continue
        fit, seen, unseen, held = make_validation_split(train,v['n_holdout_pairs'],v['min_holdout_images'],
            v['val_fraction'],seed,tuple(v['exclude_holdout_diseases']),v['max_images_per_holdout_pair'])
        fit_ids = sorted(set(fit.pair_id))
        print(f'Split {seed} | fitting={len(fit)} seen={len(seen)} held out={len(unseen)}',flush=True)
        feature_path = run/'features.npz'
        if feature_path.is_file():
            with np.load(feature_path) as bank:
                assert np.array_equal(train.file.to_numpy(),bank['filenames'])
                x_all = torch.as_tensor(bank['embeddings'],device='cuda')
        else:
            cache = ensure_resized(train.file,lora['input']['resize_cache'])
            tfs = clip_transforms(lora['input'],lora['augmentation'])
            seed_everything(seed)
            model, params = train_lora.build(lora,pairs,'cuda')
            train_lora.fit(model,params,fit,fit_ids,8,lora,cache,tfs,log=lambda r:print(r,flush=True))
            dl = train_lora.loader(train,cache,tfs[1],lora,shuffle=False)
            embeddings,_ = train_lora.encode(model,dl,'cuda',lora['runtime']['amp'],
                                            tuple(lora['evaluation'].get('tta',['identity'])))
            temp = feature_path.with_suffix('.tmp.npz')
            np.savez(temp,embeddings=embeddings,filenames=train.file.to_numpy(dtype=str))
            temp.replace(feature_path)
            del model,params,dl
            torch.cuda.empty_cache()
            x_all = torch.as_tensor(embeddings,device='cuda')
        x = x_all[fit.row.to_numpy()]
        y = torch.as_tensor(fit.pair_id.to_numpy(),device='cuda')
        xv = x_all[np.r_[seen.row.to_numpy(),unseen.row.to_numpy()]]
        labels = np.r_[seen.pair_id.to_numpy(),unseen.pair_id.to_numpy()]
        trained = torch.zeros(len(pairs),dtype=torch.bool,device='cuda')
        trained[fit_ids] = True
        pool,audit = build_anchor_pool(x,fit,pairs)
        output_rows = {'fallback_pairs':[pairs.iloc[k]['pair'] for k in held if audit[int(pairs.iloc[k].crop_id)]==0]}
        for arm,anchors in [('original',None),('anchored',pool)]:
            seed_everything(seed)
            aux_cfg = base['aux_classifier']
            aux = fit_classifier(x,y,trained,prompt,aux_cfg,aux_cfg['epochs'],tag='aux')
            generate = train_generator(x,y,cond,aux,base['gan'],anchors)
            sy = torch.as_tensor(held,device='cuda').repeat_interleave(base['synthesis']['n_per_unseen'])
            with torch.no_grad():
                sx = generate(cond[sy],sy)
            scores = []
            def score(epoch, head):
                with torch.no_grad():
                    pred = head(xv).masked_fill(hidden,float('-inf')).argmax(1).cpu().numpy()
                hits = pred==labels
                s,u = hits[:len(seen)].mean(),hits[len(seen):].mean()
                scores.append(dict(epoch=epoch,seen=float(s),unseen=float(u),h=float(2*s*u/(s+u) if s+u else 0)))
            seed_everything(seed)
            head_cfg = base['classifier']
            fit_classifier(torch.cat((x,sx)),torch.cat((y,sy)),~hidden,prompt,head_cfg,head_cfg['max_epochs'],score)
            output_rows[arm] = scores
            del generate,aux
        result_path.write_text(json.dumps(output_rows,indent=2))
    print('All development splits complete.',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    smoke() if parser.parse_args().smoke else main()

