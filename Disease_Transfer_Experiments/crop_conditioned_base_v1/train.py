"""Warm-started crop-conditioned disease training and frozen evaluation."""
import argparse
import csv
import gc
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import ConditionalDiseaseCNN,conditional_scores
from protocol import (HERE,EpochBatches,StudyLock,classification_metrics,harmonic,
                      load_config,loader,metrics,paired_bootstrap,prepare,restore_rng,
                      rng_state,save_checkpoint,seed_all,sha256,stamp,status,tensor_hash,
                      transforms_for,write_json)

ARM='conditioned'


def autocast(cfg):
    return torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp'])


def images_to(images,cfg):
    return images.to(cfg['device'],non_blocking=True,memory_format=torch.contiguous_format)


def make_model(p,cfg,pretrained=True):
    return ConditionalDiseaseCNN(cfg,p['ontology'],warm_start=pretrained).to(cfg['device'])


def optimizer_for(model,cfg):
    groups={}
    for name,p in model.named_parameters():
        head=name.startswith(('net.head.','conditioner.'))
        decay=p.ndim>1 and not name.endswith('.bias')
        groups.setdefault((head,decay),[]).append(p)
    return torch.optim.AdamW([{'params':params,'lr':cfg['head_lr' if head else 'backbone_lr'],
        'base_lr':cfg['head_lr' if head else 'backbone_lr'],'weight_decay':cfg['weight_decay'] if decay else 0.}
        for (head,decay),params in groups.items()],betas=(.9,.999),eps=1e-8)


def lr_factor(epoch,cfg):
    from independent_cnn_baseline.engine import lr_factor as base_factor
    return base_factor(epoch+1,{'warmup_epochs':cfg['warmup_epochs'],'max_epochs':cfg['schedule_epochs'],'min_lr_ratio':.01})


def selected_summary(checkpoint):
    return {'complete':checkpoint['complete'],'epochs':checkpoint['epoch'],'selected_epoch':checkpoint['epoch'] if checkpoint['complete'] else None,
        'selection':'fixed final epoch, no validation selection','seconds':sum(h['seconds'] for h in checkpoint['history']),'initial_hash':checkpoint['initial_hash']}


def recover_reports(directory,checkpoint):
    write_json(directory/'history.json',checkpoint['history'])
    write_json(directory/'summary.json',selected_summary(checkpoint))
    if checkpoint['complete'] and not (directory/'final.pt').exists():
        save_checkpoint(directory/'final.pt',{'model':checkpoint['model'],'epoch':checkpoint['epoch'],'fingerprint':checkpoint['fingerprint'],'initial_hash':checkpoint['initial_hash']})


@torch.inference_mode()
def predict(model,rows,p,cfg,transform,crop):
    model.eval(); values={k:[] for k in ('pair','crop','disease','conditional','log_crop','log_conditional')}; offset=0
    for images,_ in loader(rows,{**cfg,'batch_size':cfg['eval_batch_size']},p['ontology'],transform):
        with autocast(cfg): logits=model(images_to(images,cfg))
        cp=crop[offset:offset+len(images)]; offset+=len(images)
        pair,disease=conditional_scores(cp,logits.float().cpu(),p['ontology'])
        for k,v in dict(pair=pair,crop=cp,disease=disease,conditional=logits.float().cpu(),log_crop=F.log_softmax(cp.float(),1),log_conditional=F.log_softmax(logits.float().cpu(),2)).items(): values[k].append(v)
    values={k:torch.cat(v) for k,v in values.items()}
    if not all(torch.isfinite(v).all() for v in values.values()): raise RuntimeError('Nonfinite prediction')
    return values



def run_epoch(model,p,cfg,optimizer,scaler,transform,run,epoch):
    model.train(); rows=p['roles']['fit']
    sampler=EpochBatches(rows,cfg,grouped=False); sampler.epoch=epoch
    indices=sampler.indices(); data=loader(rows,cfg,p['ontology'],transform,sampler,epoch)
    totals={'loss':0.}
    n=correct_disease=skips=window=0
    optimizer.zero_grad(set_to_none=True)
    for batch,(images,labels) in enumerate(data,1):
        labels=labels.to(cfg['device'],non_blocking=True)
        with autocast(cfg):
            output=model(images_to(images,cfg),labels[:,0])
            loss=F.cross_entropy(output,labels[:,1],label_smoothing=cfg['label_smoothing'],reduction='sum')
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        denominator=min(cfg['batch_size']*cfg['accumulation_steps'],len(rows)-((batch-1)//cfg['accumulation_steps'])*cfg['batch_size']*cfg['accumulation_steps'])
        scaler.scale(loss/denominator).backward(); window+=len(labels)
        if batch%cfg['accumulation_steps']==0 or batch==len(data):
            scale=scaler.get_scale(); scaler.unscale_(optimizer)
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not cfg['amp'] and not torch.isfinite(norm): raise RuntimeError('Nonfinite gradients')
            scaler.step(optimizer); scaler.update(); skips+=int(scaler.get_scale()<scale)
            optimizer.zero_grad(set_to_none=True); window=0
        totals['loss']+=float(loss.detach())
        correct_disease+=int(output.detach().argmax(1).eq(labels[:,1]).sum()); n+=len(labels)
        if batch==1 or batch%100==0 or batch==len(data):
            status(run,'training',arm=ARM,epoch=epoch+1,batch=batch,total_batches=len(data),loss=totals['loss']/n)
    steps=math.ceil(len(data)/cfg['accumulation_steps'])
    if skips==steps: raise RuntimeError('All optimizer updates overflowed')
    exposure=run/ARM/f'exposures_epoch{epoch+1:02}.npz'
    np.savez_compressed(exposure,indices=np.array(indices),image_counts=np.bincount(indices,minlength=len(rows)))
    return {**{k:v/n for k,v in totals.items()},'samples':n,'conditional_disease_top1_training_crop':correct_disease/n,'amp_skipped_steps':skips,
            'optimizer_steps':steps,'unique_images':len(set(indices)),'exposure_sha256':sha256(exposure)}


def train_model(p,cfg,run):
    directory=run/ARM; directory.mkdir(exist_ok=True)
    saved=torch.load(directory/'last.pt',map_location='cpu',weights_only=True) if (directory/'last.pt').exists() else None
    if saved and saved['fingerprint']!=p['fingerprint']: raise ValueError('Training fingerprint changed')
    if saved and saved['complete']:
        recover_reports(directory,saved); status(run,'reusing_completed_model',arm=ARM); return
    seed_all(cfg['model_seed']); model=make_model(p,cfg,pretrained=saved is None)
    initial_hash=saved['initial_hash'] if saved else tensor_hash(model.state_dict())
    if saved: model.load_state_dict(saved['model'])
    train_transform,_,preprocess=transforms_for(model.net,cfg['image_size'])
    optimizer=optimizer_for(model,cfg); scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'],init_scale=cfg['amp_initial_scale'])
    history=saved['history'] if saved else []
    if saved:
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng'])
    write_json(directory/'initialization.json',{'initial_hash':initial_hash,'model_seed':cfg['model_seed'],
        'disease_checkpoint_sha256':p['disease_checkpoint_sha256'],'model':cfg['model'],
        'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),
        'trainable_backbone_parameters':sum(x.numel() for x in model.net.backbone.parameters() if x.requires_grad),
        'preprocessing':preprocess,'train_transform':str(train_transform),'selection':'fixed 15 additional epochs; no validation'})
    del saved; gc.collect()
    for epoch in range(len(history),cfg['epochs']):
        start=time.perf_counter()
        for group in optimizer.param_groups: group['lr']=group['base_lr']*lr_factor(epoch,cfg)
        training=run_epoch(model,p,cfg,optimizer,scaler,train_transform,run,epoch)
        history.append({'epoch':epoch+1,'training':training,'seconds':time.perf_counter()-start,'backbone_lr':optimizer.param_groups[0]['lr']})
        complete=epoch+1==cfg['epochs']
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'rng':rng_state(),'history':history,'epoch':epoch+1,'complete':complete,'fingerprint':p['fingerprint'],'initial_hash':initial_hash}
        save_checkpoint(directory/'last.pt',checkpoint); recover_reports(directory,checkpoint)
        status(run,'epoch_complete',arm=ARM,epoch=epoch+1,fixed_epochs=cfg['epochs'],seconds=history[-1]['seconds'],complete=complete)
        del checkpoint
    del model,optimizer,scaler; gc.collect(); torch.cuda.empty_cache()


def preflight(p,cfg,run):
    seed_all(cfg['model_seed']); model=make_model(p,cfg)
    backbone_parameters=sum(x.numel() for x in model.net.backbone.parameters())
    if backbone_parameters<=80_000_000: raise RuntimeError('Expected Base backbone; refusing smaller model')
    transform,_,_=transforms_for(model.net,cfg['image_size'])
    sampler=EpochBatches(p['roles']['fit'],cfg)
    images,labels=next(iter(loader(p['roles']['fit'],{**cfg,'workers':0},p['ontology'],transform,sampler)))
    labels=labels.to(cfg['device']); images=images_to(images,cfg)
    model.eval()
    with torch.inference_mode(),autocast(cfg):
        reference=model.net(images); conditioned=model(images)
        identity=bool(torch.equal(conditioned,reference[:,None,:].expand_as(conditioned)))
    if not identity: raise RuntimeError('Identity initialization does not reproduce saved disease model')
    model.train(); optimizer=optimizer_for(model,cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'],init_scale=cfg['amp_initial_scale'])
    before_net=tensor_hash(model.net.state_dict()); before_head=tensor_hash(model.conditioner.state_dict())
    torch.cuda.reset_peak_memory_stats()
    for attempt in range(14):
        optimizer.zero_grad(set_to_none=True)
        with autocast(cfg):
            output=model(images,labels[:,0]); loss=F.cross_entropy(output,labels[:,1],label_smoothing=cfg['label_smoothing'])
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        gradients={'backbone':float(model.net.backbone.stem[0].weight.grad.norm()),'conditioner':float(model.conditioner.affine.weight.grad.norm()),'disease_classifier':float(model.net.head[1].weight.grad.norm())}
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer); scaler.update()
        if torch.isfinite(norm): break
    else: raise RuntimeError('GPU probe failed to obtain finite applied update')
    after_net=tensor_hash(model.net.state_dict()); after_head=tensor_hash(model.conditioner.state_dict())
    if before_net==after_net or before_head==after_head or not all(math.isfinite(x) and x>0 for x in gradients.values()): raise RuntimeError('GPU probe did not update disease CNN and conditioner')
    write_json(run/'preflight.json',{'passed':True,'at_utc':stamp(),'model':cfg['model'],'feature_width':model.net.backbone.num_features,'backbone_parameters':backbone_parameters,'gradients':gradients,'backbone_changed':True,'conditioner_changed':True,'identity_initialization_exact':identity,'amp_skipped_probe_steps':attempt,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'discarded_probe_updates':True})
    status(run,'preflight_passed',gradients=gradients)
    del model,optimizer,scaler,output,loss; gc.collect(); torch.cuda.empty_cache()


def cache_crop(p,cfg,run,roles):
    # Reuse verified saved logits from the unchanged Base crop checkpoint.
    base=Path(cfg['baseline_run']); bank=np.load(base/'factor_logits.npz'); scores=np.load(base/'scores.npz'); result={}
    for role in roles:
        split='test_seen' if role=='official_seen' else 'test_unseen'; rows=p['roles'][role]
        with (base/f'predictions_{split}.csv').open(newline='',encoding='utf-8-sig') as f: originals=list(csv.DictReader(f))
        if [r['file'] for r in originals]!=[r['filename'] for r in rows]: raise ValueError('Base prediction filename alignment changed')
        if scores[split+'_labels'].tolist()!=[r['composition_id'] for r in rows]: raise ValueError('Base official label alignment changed')
        result[role]=torch.from_numpy(bank[split+'_crop'].copy())
        np.savez_compressed(run/f'crop_{role}.npz',logits=result[role].numpy(),filenames=[r['filename'] for r in rows],checkpoint_hash=p['crop_checkpoint_sha256'],source_logits_sha256=p['historical_hashes'][str(base/'factor_logits.npz')])
    return result


def freeze(p,run):
    last=torch.load(run/ARM/'last.pt',map_location='cpu',weights_only=True)
    if not last['complete']: raise ValueError('Cannot freeze incomplete training')
    best=torch.load(run/ARM/'final.pt',map_location='cpu',weights_only=True)
    value={'frozen_utc':stamp(),'fingerprint':p['fingerprint'],'checkpoint_sha256':sha256(run/ARM/'final.pt'),
           'selected_epoch':best['epoch'],'selection':'fixed 15 additional epochs, no validation','official_scores_used':False}
    path=run/'selection_frozen_before_test.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['checkpoint_sha256']!=value['checkpoint_sha256']: raise ValueError('Frozen checkpoint changed')
    else: write_json(path,value)


def evaluate(p,cfg,run):
    saved=torch.load(run/ARM/'final.pt',map_location='cpu',weights_only=True)
    model=make_model(p,cfg,False); model.load_state_dict(saved['model']); del saved
    crop_cache=cache_crop(p,cfg,run,('official_seen','official_unseen'))
    _,transform,_=transforms_for(model.net,cfg['image_size'])
    result={'candidate_count':333,'score':'log P(crop|image) + log P(disease|image,candidate crop)','metric_units':'fractions'}
    pids=np.array([c['composition_id'] for c in p['ontology']['candidates']])
    for role in ('official_seen','official_unseen'):
        rows=p['roles'][role]; path=run/f'logits_{role}.npz'
        status(run,'final_inference',role=role)
        if path.exists():
            bank=np.load(path)
            if bank['checkpoint_hash'].item()!=sha256(run/ARM/'final.pt'): raise ValueError('Logit cache mismatch')
            output={k:torch.from_numpy(bank[k].copy()) for k in ('pair','crop','disease','conditional','log_crop','log_conditional')}
        else:
            output=predict(model,rows,p,cfg,transform,crop_cache[role])
            np.savez_compressed(path,**{k:v.numpy() for k,v in output.items()},filenames=[r['filename'] for r in rows],
                crop_targets=[r['crop_id'] for r in rows],disease_targets=[r['disease_id'] for r in rows],
                composition_targets=[r['composition_id'] for r in rows],checkpoint_hash=sha256(run/ARM/'final.pt'))
        result[role]=metrics(output['crop'],output['disease'],rows,p['ontology'],pair=output['pair'])
        seen={r['composition_id'] for r in p['roles']['fit']}
        seen_mask=torch.tensor([c['composition_id'] in seen for c in p['ontology']['candidates']])
        calibrated=output['pair']-cfg['baseline_gamma']*seen_mask.double()[None,:]
        result[role+'_fixed_gamma']=metrics(output['crop'],output['disease'],rows,p['ontology'],pair=calibrated)
        ranking=output['pair'].argsort(dim=1,descending=True,stable=True)[:,:5].numpy()
        with (run/f'predictions_{role}.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f); writer.writerow(['filename','true_composition_id','predicted_composition_id','top5_ids'])
            for row,ids in zip(rows,ranking):
                ranked=pids[ids].tolist(); writer.writerow([row['filename'],row['composition_id'],ranked[0],json.dumps(ranked)])
    result['H']={k:harmonic(result['official_seen']['composition'][k],result['official_unseen']['composition'][k])
                 for k in ('top1','top5','macro_top1')}
    result['H_fixed_gamma']={k:harmonic(result['official_seen_fixed_gamma']['composition'][k],result['official_unseen_fixed_gamma']['composition'][k]) for k in ('top1','top5','macro_top1')}
    result['fixed_gamma']=cfg['baseline_gamma']
    write_json(run/'metrics.json',result)
    del model; gc.collect(); torch.cuda.empty_cache()
    return result


def baseline_correct(p,role,gamma):
    base=Path(p['config']['baseline_run']); bank=np.load(base/'scores.npz')
    split='test_seen' if role=='official_seen' else 'test_unseen'; seen={r['composition_id'] for r in p['roles']['fit']}
    mask=np.array([c['composition_id'] in seen for c in p['ontology']['candidates']])
    scores=bank[split+'_scores'].astype(np.float64)-gamma*mask[None,:]
    return scores.argmax(1)==bank[split+'_labels']


def analyze(p,cfg,run,result):
    out={}; seen={r['composition_id'] for r in p['roles']['fit']}
    mask=np.array([c['composition_id'] in seen for c in p['ontology']['candidates']]); ids=np.array([c['composition_id'] for c in p['ontology']['candidates']])
    for gamma in (0.,cfg['baseline_gamma']):
        point={}; draws={}
        for role in ('official_seen','official_unseen'):
            bank=np.load(run/f'logits_{role}.npz'); a=ids[(bank['pair']-gamma*mask[None,:]).argmax(1)]==bank['composition_targets']; b=baseline_correct(p,role,gamma)
            point[role]={'new_correct':int(a.sum()),'baseline_correct':int(b.sum()),'delta_pp':float(100*(a.mean()-b.mean()))}
            for cluster in (False,True): draws[(role,cluster)]=paired_bootstrap(a,b,p['roles'][role],cfg,cluster)
        intervals={}
        for cluster in (False,True):
            sa,sb=draws[('official_seen',cluster)]; ua,ub=draws[('official_unseen',cluster)]
            ha=np.divide(2*sa*ua,sa+ua,out=np.zeros_like(sa),where=sa+ua>0); hb=np.divide(2*sb*ub,sb+ub,out=np.zeros_like(sb),where=sb+ub>0)
            intervals['composition' if cluster else 'image']={'unseen_delta_95_pp':np.percentile(100*(ua-ub),[2.5,97.5]).tolist(),'H_delta_95_pp':np.percentile(100*(ha-hb),[2.5,97.5]).tolist()}
        out[str(gamma)]={'point':point,'intervals':intervals}
    write_json(run/'paired_analysis.json',out); return out


def reports(p,cfg,run,result,analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=json.loads((run/ARM/'history.json').read_text()); epochs=[h['epoch'] for h in history]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(epochs,[h['training']['loss'] for h in history]); axes[0].set_title('Training conditional CE (smoothed)')
    axes[1].plot(epochs,[100*h['training']['conditional_disease_top1_training_crop'] for h in history]); axes[1].set_title('Training disease accuracy (%)')
    for ax in axes: ax.set_xlabel('Additional epoch'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run/'training_curves.png',dpi=160); plt.close(fig)
    base=Path(cfg['baseline_run']); calibrated=json.loads((base/'metrics.json').read_text()); raw=json.loads((base/'uncalibrated_metrics.json').read_text())
    lines=['# ConvNeXt-Base crop-conditioned disease continuation','', '| Model / scoring | Seen Top-1 | Unseen Top-1 | H |','|---|---:|---:|---:|']
    for name,b in [('Saved Base, gamma 0',raw),('Saved Base, fixed gamma 6.2',calibrated)]: lines.append(f"| {name} | {100*b['test_seen']['top1']:.2f} | {100*b['test_unseen']['top1']:.2f} | {100*b['harmonic_mean_top1']:.2f} |")
    for name,suffix,hkey in [('Conditional Base, gamma 0','','H'),('Conditional Base, same fixed gamma 6.2','_fixed_gamma','H_fixed_gamma')]: lines.append(f"| {name} | {100*result['official_seen'+suffix]['composition']['top1']:.2f} | {100*result['official_unseen'+suffix]['composition']['top1']:.2f} | {100*result[hkey]['top1']:.2f} |")
    lines+=['','One new Base disease model, all disease parameters trainable; unchanged Base crop model/logits reused. No control retraining.',
        'Initialized from the saved full-data Base disease checkpoint. Fixed 15 additional epochs on all original 8,000 training images; no new validation, pseudo-unseen partition, early stopping or checkpoint search.',
        'The saved Base checkpoint already saw every training image. A subset of those images would not provide a fresh validation set. The fixed budget was specified before this continuation trained.',
        'Original Base 256-pixel JPEG cache, transforms, dropout .2, DropPath .1, batch4/accumulation8, optimizer grouping and 30-epoch LR horizon are retained. New optimizer/scheduler states are used for this 15-epoch continuation.',
        'Candidate-specific disease normalization and all-crop inference use no true crop. All 333 candidates are considered. FP32 log probabilities and pair addition match the original Base score precision; exact ties prefer lower pair ID.',
        'The existing gamma6.2 is frozen and reported alongside gamma0; no new calibration search, test-selected gamma, epoch or rescue scoring is used.',
        'This compares a complete conditional fine-tuning recipe against the saved Base baseline. Without a matched continuation control it cannot isolate conditioning from extra fitting. One seed; official tests are familiar.',
        'Earlier Tiny experiments remain preserved and are not evidence of improvement or failure on Base. Only this current experiment was replaced.',
        'Training-plot visual review remains pending until inspected. Numerical verification follows automatically.','',json.dumps(analysis,indent=2)]
    (run/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default=str(HERE/'config.json'))
    parser.add_argument('--preflight-only',action='store_true'); args=parser.parse_args()
    cfg=load_config(args.config); run=HERE/'runs'/cfg['run_name']; run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(2); os.environ['HF_HUB_OFFLINE']='1'
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    with StudyLock(run/'study.lock'):
        try:
            p=prepare(cfg,run)
            if shutil.disk_usage(run).free<3*1024**3: raise RuntimeError('Need 3 GiB free for atomic Base checkpoints')
            if not (run/'preflight.json').exists(): preflight(p,cfg,run)
            if args.preflight_only: return
            train_model(p,cfg,run); freeze(p,run)
            result=evaluate(p,cfg,run); analysis=analyze(p,cfg,run,result); reports(p,cfg,run,result,analysis)
            from verify import verify_run
            check=verify_run(run); status(run,'complete',verification_checks=check['checks'],results=str(run/'results.md'))
        except BaseException as error:
            status(run,'failed',error=repr(error),traceback=traceback.format_exc()); raise


if __name__=='__main__': main()
