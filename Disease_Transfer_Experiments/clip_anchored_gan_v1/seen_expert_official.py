"""Development trial of a direct ConvNeXt seen-pair head with auxiliary labels."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'leafgan_v1'))
from classifier_multi import DETAIL, Images, IndependentNet, transforms, seed_all, sha256, status, write_json
from independent_cnn_baseline.models import optimizer
from storage import CheckpointStore
from learned_gate import decisions, metrics


class PairImages(Images):
    def __getitem__(self,i):
        x,disease = super().__getitem__(i)
        row = self.rows[i]
        return x,disease,row['crop_id'],row['composition_id']


class Model(nn.Module):
    def __init__(self,base,n_pairs):
        super().__init__()
        self.disease_model = IndependentNet(base,'disease')
        width = self.disease_model.backbone.num_features
        self.head_pair = nn.Linear(width,n_pairs)
        self.head_crop = nn.Linear(width,55)

    def forward(self,x):
        f = self.disease_model.backbone(x)
        return self.head_pair(f),self.head_crop(f),self.disease_model.head(f)


def main():
    torch.set_num_threads(4)
    seed_all(42)
    out = HERE/'seen_expert_official'
    out.mkdir(exist_ok=True)
    prepared = DETAIL/'runs/development_seed42/prepared.json'
    p = json.loads(prepared.read_text())
    base = p['baseline_config']
    project = HERE.parents[1]/'Out-Of-Distribution-Plant-Disease-Identification'
    data = HERE.parents[1]/'Plant_Disease'
    baseline = project/'runs/baseline/independent_cnn_baseline_convnext_base_seed42'
    source = baseline/'disease_net.pt'
    cache = project/'runs/cache/resized_256'
    def load(split):
        with (data/f'DPD_Small_{split}.csv').open(encoding='utf-8-sig',newline='') as f:
            return [dict(filename=r[0],crop_id=int(r[1]),disease_id=int(r[2]),
                         composition_id=int(r[3]),path=str(cache/(Path(r[0]).stem+'.jpg'))) for r in csv.reader(f)]
    fit = load('train')
    codes = sorted({r['composition_id'] for r in fit})
    table = sorted(p['pairs'],key=lambda r:r['pair_id'])
    unseen = np.array([r['pair_id'] for r in table if r['is_unseen']])
    assert len(codes)==323 and len(fit)==8000 and not set(codes).intersection(unseen)
    assert len({r['filename'] for r in fit})==8000 and all(Path(r['path']).is_file() for r in fit)
    columns = torch.full((333,),-1,device='cuda',dtype=torch.long)
    columns[codes] = torch.arange(len(codes),device='cuda')
    selected = json.loads((HERE/'seen_expert/results.json').read_text())['best']['epoch']
    cfg = dict(epochs=selected,microbatch=4,accumulate=8,aux_crop=.25,aux_disease=.25)
    gate_path = HERE/'hybrid/selection.json'
    rule = json.loads(gate_path.read_text())
    write_json(out/'selection.json',{'epochs':selected,'fit_images':len(fit),'seen_pairs':len(codes),
                                     'gate':rule,'initial_checkpoint_sha256':sha256(source)})
    model = Model(base,len(codes))
    model.disease_model.load_state_dict(torch.load(source,weights_only=True,map_location='cpu'))
    model.disease_model.backbone.set_grad_checkpointing(True)
    model.cuda()
    # Match the baseline backbone and head rates, with the new auxiliary heads in the head group.
    config = base['optimizer']
    backbone = list(model.disease_model.backbone.parameters())
    heads = list(model.head_pair.parameters())+list(model.head_crop.parameters())+list(model.disease_model.head.parameters())
    opt = torch.optim.AdamW([{'params':backbone,'lr':config['lr_backbone']},
                             {'params':heads,'lr':config['lr_head']}],
                            weight_decay=config['weight_decay'],betas=tuple(config['betas']),eps=config['eps'])
    scaler = torch.amp.GradScaler('cuda')
    train_tf, eval_tf = transforms(base)
    train = DataLoader(PairImages(fit,train_tf),batch_size=cfg['microbatch'],shuffle=True,num_workers=2)
    fingerprint = (sha256(Path(__file__))+sha256(prepared)+sha256(source)
                   +sha256(data/'DPD_Small_train.csv')+sha256(gate_path)+str(selected))
    store = CheckpointStore(out,fingerprint,6*1024**3)
    saved = store.load()
    history = []
    if saved:
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        history = saved['history']
        del saved
    criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    for epoch in range(len(history)+1,cfg['epochs']+1):
        seed_all(42+epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        for step,(x,d,c,pair) in enumerate(train,1):
            d,c,pair = d.cuda(),c.cuda(),pair.cuda()
            update_size = min(32,len(fit)-((step-1)//8)*32)
            with torch.autocast('cuda',dtype=torch.float16):
                logits,cl,dl = model(x.cuda())
                loss = (criterion(logits,columns[pair])+.25*criterion(cl,c)+.25*criterion(dl,d))*len(x)/update_size
            scaler.scale(loss).backward()
            if step%8==0 or step==len(train):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if step==1 or step%100==0:
                status(out,'training',epoch=epoch,step=step,total_steps=len(train))
        history.append({'epoch':epoch})
        store.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                        history=history,epoch=epoch,cursor=0,stage='finetune',fingerprint=fingerprint))
        write_json(out/'history.json',history)
        status(out,'epoch_complete',epoch=epoch)
        if (out/'STOP.request').exists():
            status(out,'paused',epoch=epoch)
            return
    model.eval()
    predictions = {'old_seen_expert':[],'direct_seen_expert':[]}
    labels,groups = [],[]
    crops = np.array([r['crop_id'] for r in table])
    fit_names = {r['filename'] for r in fit}
    with np.load(baseline/'scores.npz') as n:
        for split,count in [('test_seen',2000),('test_unseen',1448)]:
            status(out,'evaluating',split=split)
            rows = load(split)
            names = [r['filename'] for r in rows]
            y = np.array([r['composition_id'] for r in rows])
            assert len(rows)==count and fit_names.isdisjoint(names) and np.array_equal(y,n[f'{split}_labels'])
            with np.load(HERE/'official'/f'anchored_{split}.npz') as c:
                assert np.array_equal(names,c['filenames']) and np.array_equal(y,c['labels'])
                a,b,features = decisions(c['scores'],n[f'{split}_scores'],np.array(codes),unseen,crops)
            use_seen = features[:,0]+rule['cnn_gate_weight']*features[:,1]/rule['cnn_gap_scale']>=rule['threshold']
            model_pred = a.copy()
            chosen_rows = [r for r,use in zip(rows,use_seen) if use]
            valid = DataLoader(PairImages(chosen_rows,eval_tf),batch_size=16,num_workers=2)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                values = torch.cat([model(x.cuda())[0].float().cpu() for x,_,_,_ in valid]).numpy()
            assert np.isfinite(values).all()
            model_pred[use_seen] = np.array(codes)[values.argmax(1)]
            old,new = np.where(use_seen,a,b),np.where(use_seen,model_pred,b)
            predictions['old_seen_expert'].append(old)
            predictions['direct_seen_expert'].append(new)
            labels.append(y)
            groups.append(np.full(count,split=='test_seen'))
            np.savez_compressed(out/f'{split}.npz',labels=y,filenames=names,old=old,new=new,routed_seen=use_seen)
    results = {k:metrics(np.concatenate(v),np.concatenate(labels),np.concatenate(groups))
               for k,v in predictions.items()}
    assert results['old_seen_expert']['seen_correct']==1338 and results['old_seen_expert']['unseen_correct']==1134
    assert results['old_seen_expert']['unseen_correct']==results['direct_seen_expert']['unseen_correct']
    write_json(out/'results.json',{'epochs':selected,'official':results})
    status(out,'complete',epochs=cfg['epochs'],official=results)


if __name__=='__main__':
    main()
