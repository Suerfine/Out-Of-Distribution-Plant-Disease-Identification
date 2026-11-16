"""Development trial of a direct ConvNeXt seen-pair head with auxiliary labels."""
from pathlib import Path
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
    out = HERE/'seen_expert'
    out.mkdir(exist_ok=True)
    prepared = DETAIL/'runs/development_seed42/prepared.json'
    p = json.loads(prepared.read_text())
    base = p['baseline_config']
    fit,seen = (p['roles'][name] for name in ('fit','validation_seen'))
    codes = sorted({r['composition_id'] for r in fit})
    assert len(codes)==303 and len(fit)==5798 and len(seen)==862
    columns = torch.full((333,),-1,device='cuda',dtype=torch.long)
    columns[codes] = torch.arange(len(codes),device='cuda')
    cfg = dict(epochs=5,microbatch=4,accumulate=8,aux_crop=.25,aux_disease=.25)
    source = next(Path(k) for k in p['reference_hashes'] if Path(k).parent.name=='disease' and Path(k).name=='best_epoch_15.pt')
    assert sha256(source)==p['reference_hashes'][str(source)]
    model = Model(base,len(codes))
    model.disease_model.load_state_dict(torch.load(source,weights_only=True,map_location='cpu')['model_state'])
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
    valid = DataLoader(PairImages(seen,eval_tf),batch_size=16,num_workers=2)
    targets = np.array([r['composition_id'] for r in seen])
    table = sorted(p['pairs'],key=lambda r:r['pair_id'])
    crop,disease = [np.array([r[k] for r in table])[codes] for k in ('crop_id','disease_id')]
    with np.load(p['baseline_validation_logits']) as z:
        old = z['crop'][:len(seen),crop]+z['disease'][:len(seen),disease]
    reference = int((np.array(codes)[old.argmax(1)]==targets).sum())
    fingerprint = sha256(Path(__file__))+sha256(prepared)+sha256(source)
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
        model.eval()
        status(out,'validating',epoch=epoch)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            scores = torch.cat([model(x.cuda())[0].float().cpu() for x,_,_,_ in valid]).numpy()
        assert np.isfinite(scores).all()
        correct = int((np.array(codes)[scores.argmax(1)]==targets).sum())
        history.append(dict(epoch=epoch,correct=correct,images=len(seen),baseline_correct=reference))
        np.save(out/f'epoch_{epoch}_scores.npy',scores)
        store.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                        history=history,epoch=epoch,cursor=0,stage='finetune',fingerprint=fingerprint))
        write_json(out/'history.json',history)
        status(out,'epoch_complete',**history[-1])
        if (out/'STOP.request').exists():
            status(out,'paused',epoch=epoch)
            return
    write_json(out/'results.json',dict(best=max(history,key=lambda r:r['correct']),history=history,candidate_ids=codes))
    status(out,'complete',epochs=cfg['epochs'])


if __name__=='__main__':
    main()
