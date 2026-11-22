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
    head={id(x) for x in model.net.get_classifier().parameters()} | {id(x) for x in model.conditioner.parameters()}
    return torch.optim.AdamW([
        {'params':[x for x in model.parameters() if id(x) not in head],'lr':cfg['backbone_lr'],'base_lr':cfg['backbone_lr']},
        {'params':[x for x in model.parameters() if id(x) in head],'lr':cfg['head_lr'],'base_lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])


def lr_factor(epoch,cfg):
    if epoch<cfg['warmup_epochs']: return (epoch+1)/cfg['warmup_epochs']
    progress=(epoch-cfg['warmup_epochs'])/max(1,cfg['epochs']-cfg['warmup_epochs']-1)
    return .01+.99*.5*(1+math.cos(math.pi*progress))


def selected_summary(checkpoint):
    return {'complete':checkpoint['complete'],'epochs':checkpoint['epoch'],
        'selected_epoch':checkpoint['best_epoch'],'selected_score':checkpoint['best_key'][0],
        'seconds':sum(h['seconds'] for h in checkpoint['history']),
        'initial_hash':checkpoint['initial_hash']}


def recover_reports(directory,checkpoint):
    write_json(directory/'history.json',checkpoint['history'])
    write_json(directory/'summary.json',selected_summary(checkpoint))


@torch.inference_mode()
def predict(model,rows,p,cfg,transform,crop):
    model.eval(); values={k:[] for k in ('pair','crop','disease','conditional')}; offset=0
    for images,_ in loader(rows,cfg,p['ontology'],transform):
        with autocast(cfg): logits=model(images_to(images,cfg))
        cp=crop[offset:offset+len(images)]; offset+=len(images)
        pair,disease=conditional_scores(cp,logits.float().cpu(),p['ontology'])
        for k,v in dict(pair=pair,crop=cp,disease=disease,conditional=logits.float().cpu()).items(): values[k].append(v)
    values={k:torch.cat(v) for k,v in values.items()}
    if not all(torch.isfinite(v).all() for v in values.values()): raise RuntimeError('Nonfinite prediction')
    return values


def validation(model,p,cfg,transform,crop):
    rows=p['roles']['seen_validation']; output=predict(model,rows,p,cfg,transform,crop)
    dm={v:i for i,v in enumerate(p['ontology']['disease'])}
    y=torch.tensor([dm[r['disease_id']] for r in rows])
    value=metrics(output['crop'],output['disease'],rows,p['ontology'],pair=output['pair'])
    return {'selection_score':value['disease']['top1'],
            'mean_ce':float(F.nll_loss(output['disease'],y)), 'metrics':value}


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
        scaler.scale(loss).backward(); window+=len(labels)
        if batch%cfg['accumulation_steps']==0 or batch==len(data):
            scale=scaler.get_scale(); scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None: parameter.grad.div_(window)
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not cfg['amp'] and not torch.isfinite(norm): raise RuntimeError('Nonfinite gradients')
            scaler.step(optimizer); scaler.update(); skips+=int(scaler.get_scale()<scale)
            optimizer.zero_grad(set_to_none=True); window=0
        totals['loss']+=float(loss.detach())
        correct_disease+=int(output.detach().argmax(1).eq(labels[:,1]).sum()); n+=len(labels)
        if batch==1 or batch%100==0 or batch==len(data):
            status(run,'training',arm=ARM,epoch=epoch+1,batch=batch,total_batches=len(data),loss=totals['loss']/n)
    exposure=run/ARM/f'exposures_epoch{epoch+1:02}.npz'
    np.savez_compressed(exposure,indices=np.array(indices),image_counts=np.bincount(indices,minlength=len(rows)))
    return {**{k:v/n for k,v in totals.items()},'samples':n,'conditional_disease_top1_training_crop':correct_disease/n,'amp_skipped_steps':skips,
            'unique_images':len(set(indices)),'exposure_sha256':sha256(exposure)}


def train_model(p,cfg,run,crop):
    directory=run/ARM; directory.mkdir(exist_ok=True)
    saved=torch.load(directory/'last.pt',map_location='cpu',weights_only=True) if (directory/'last.pt').exists() else None
    if saved and saved['fingerprint']!=p['fingerprint']: raise ValueError('Training fingerprint changed')
    if saved and saved['complete']:
        recover_reports(directory,saved); status(run,'reusing_completed_model',arm=ARM); return
    seed_all(cfg['seed']+1)
    model=make_model(p,cfg,pretrained=saved is None)
    initial_hash=saved['initial_hash'] if saved else tensor_hash(model.state_dict())
    if saved: model.load_state_dict(saved['model'])
    train_transform,eval_transform,preprocess=transforms_for(model.net,cfg['image_size'])
    optimizer=optimizer_for(model,cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'],init_scale=cfg['amp_initial_scale'])
    history=saved['history'] if saved else []
    initial_validation=saved['initial_validation'] if saved else validation(model,p,cfg,eval_transform,crop)
    best_key=tuple(saved['best_key']) if saved else (initial_validation['selection_score'],-initial_validation['mean_ce'])
    best_epoch=saved['best_epoch'] if saved else 0; stale=saved['stale'] if saved else 0
    if saved:
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng'])
    write_json(directory/'initialization.json',{'initial_hash':initial_hash,'model_seed':cfg['seed']+1,
        'disease_checkpoint_sha256':p['disease_checkpoint_sha256'],'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),
        'trainable_backbone_parameters':sum(x.numel() for name,x in model.net.named_parameters() if x.requires_grad and not name.startswith('head.fc.')),
        'preprocessing':preprocess,'train_transform':str(train_transform),'eval_transform':str(eval_transform)})
    if saved is None:
        save_checkpoint(directory/'best.pt',{'model':model.state_dict(),'epoch':0,'validation':initial_validation,'fingerprint':p['fingerprint'],'initial_hash':initial_hash})
    del saved; gc.collect()
    for epoch in range(len(history),cfg['epochs']):
        start=time.perf_counter()
        for group in optimizer.param_groups: group['lr']=group['base_lr']*lr_factor(epoch,cfg)
        training=run_epoch(model,p,cfg,optimizer,scaler,train_transform,run,epoch)
        status(run,'validating',arm=ARM,epoch=epoch+1)
        val=validation(model,p,cfg,eval_transform,crop)
        key=(val['selection_score'],-val['mean_ce']); improved=key>best_key
        if improved:
            best_key=key; best_epoch=epoch+1; stale=0
            save_checkpoint(directory/'best.pt',{'model':model.state_dict(),'epoch':epoch+1,
                'validation':val,'fingerprint':p['fingerprint'],'initial_hash':initial_hash})
        else: stale+=1
        complete=epoch+1==cfg['epochs'] or stale>=cfg['patience']
        history.append({'epoch':epoch+1,'training':training,'validation':val,'improved':improved,
                        'seconds':time.perf_counter()-start,'backbone_lr':optimizer.param_groups[0]['lr']})
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'rng':rng_state(),'history':history,'best_key':list(best_key),'best_epoch':best_epoch,'stale':stale,
            'epoch':epoch+1,'complete':complete,'fingerprint':p['fingerprint'],'initial_hash':initial_hash,'initial_validation':initial_validation}
        save_checkpoint(directory/'last.pt',checkpoint); recover_reports(directory,checkpoint)
        status(run,'epoch_complete',arm=ARM,epoch=epoch+1,selected_epoch=best_epoch,
               selection_score=val['selection_score'],seconds=history[-1]['seconds'],complete=complete)
        del checkpoint
        if complete: break
    del model,optimizer,scaler; gc.collect(); torch.cuda.empty_cache()


def preflight(p,cfg,run):
    seed_all(cfg['seed']+1); model=make_model(p,cfg)
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
        gradients={'backbone':float(model.net.stem[0].weight.grad.norm()),'conditioner':float(model.conditioner.affine.weight.grad.norm()),'disease_classifier':float(model.net.get_classifier().weight.grad.norm())}
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer); scaler.update()
        if torch.isfinite(norm): break
    else: raise RuntimeError('GPU probe failed to obtain finite applied update')
    after_net=tensor_hash(model.net.state_dict()); after_head=tensor_hash(model.conditioner.state_dict())
    if before_net==after_net or before_head==after_head or not all(math.isfinite(x) and x>0 for x in gradients.values()): raise RuntimeError('GPU probe did not update disease CNN and conditioner')
    write_json(run/'preflight.json',{'passed':True,'at_utc':stamp(),'gradients':gradients,'backbone_changed':True,'conditioner_changed':True,'identity_initialization_exact':identity,'amp_skipped_probe_steps':attempt,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'discarded_probe_updates':True})
    status(run,'preflight_passed',gradients=gradients)
    del model,optimizer,scaler,output,loss; gc.collect(); torch.cuda.empty_cache()


def cache_crop(p,cfg,run,roles):
    import timm
    source=torch.load(p['crop_checkpoint'],map_location='cpu',weights_only=True)
    if source['label_ids']!=p['ontology']['crop']: raise ValueError('Crop label order changed')
    net=timm.create_model(source['model_name'],pretrained=False,num_classes=len(source['label_ids'])).to(cfg['device']).eval()
    net.load_state_dict(source['model_state'])
    for parameter in net.parameters(): parameter.requires_grad_(False)
    _,transform,_=transforms_for(net,cfg['image_size']); result={}
    for role in roles:
        path=run/f'crop_{role}.npz'; rows=p['roles'][role]
        if path.exists():
            bank=np.load(path)
            if bank['checkpoint_hash'].item()!=p['crop_checkpoint_sha256'] or bank['filenames'].tolist()!=[r['filename'] for r in rows]: raise ValueError('Crop cache mismatch')
            result[role]=torch.from_numpy(bank['logits'].copy()); continue
        status(run,'caching_original_crop',role=role); values=[]
        with torch.inference_mode():
            for images,_ in loader(rows,cfg,p['ontology'],transform):
                with autocast(cfg): logits=net(images_to(images,cfg))
                values.append(logits.float().cpu())
        result[role]=torch.cat(values)
        np.savez_compressed(path,logits=result[role].numpy(),filenames=[r['filename'] for r in rows],checkpoint_hash=p['crop_checkpoint_sha256'])
    del net,source; gc.collect(); torch.cuda.empty_cache(); return result


def freeze(p,run):
    last=torch.load(run/ARM/'last.pt',map_location='cpu',weights_only=True)
    if not last['complete']: raise ValueError('Cannot freeze incomplete training')
    best=torch.load(run/ARM/'best.pt',map_location='cpu',weights_only=True)
    value={'frozen_utc':stamp(),'fingerprint':p['fingerprint'],'checkpoint_sha256':sha256(run/ARM/'best.pt'),
           'selected_epoch':best['epoch'],'validation':best['validation'],'official_scores_used':False}
    path=run/'selection_frozen_before_test.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['checkpoint_sha256']!=value['checkpoint_sha256']: raise ValueError('Frozen checkpoint changed')
    else: write_json(path,value)


def evaluate(p,cfg,run):
    saved=torch.load(run/ARM/'best.pt',map_location='cpu',weights_only=True)
    model=make_model(p,cfg,False); model.load_state_dict(saved['model']); del saved
    crop_cache=cache_crop(p,cfg,run,('seen_validation','official_seen','official_unseen'))
    _,transform,_=transforms_for(model.net,cfg['image_size'])
    result={'candidate_count':333,'score':'log P(crop|image) + log P(disease|image,candidate crop)','metric_units':'fractions'}
    pids=np.array([c['composition_id'] for c in p['ontology']['candidates']])
    for role in ('seen_validation','official_seen','official_unseen'):
        rows=p['roles'][role]; path=run/f'logits_{role}.npz'
        status(run,'final_inference',role=role)
        if path.exists():
            bank=np.load(path)
            if bank['checkpoint_hash'].item()!=sha256(run/ARM/'best.pt'): raise ValueError('Logit cache mismatch')
            output={k:torch.from_numpy(bank[k].copy()) for k in ('pair','crop','disease','conditional')}
        else:
            output=predict(model,rows,p,cfg,transform,crop_cache[role])
            np.savez_compressed(path,**{k:v.numpy() for k,v in output.items()},filenames=[r['filename'] for r in rows],
                crop_targets=[r['crop_id'] for r in rows],disease_targets=[r['disease_id'] for r in rows],
                composition_targets=[r['composition_id'] for r in rows],checkpoint_hash=sha256(run/ARM/'best.pt'))
        result[role]=metrics(output['crop'],output['disease'],rows,p['ontology'],pair=output['pair'])
        ranking=output['pair'].argsort(dim=1,descending=True,stable=True)[:,:5].numpy()
        with (run/f'predictions_{role}.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f); writer.writerow(['filename','true_composition_id','predicted_composition_id','top5_ids'])
            for row,ids in zip(rows,ranking):
                ranked=pids[ids].tolist(); writer.writerow([row['filename'],row['composition_id'],ranked[0],json.dumps(ranked)])
    result['H']={k:harmonic(result['official_seen']['composition'][k],result['official_unseen']['composition'][k])
                 for k in ('top1','top5','macro_top1')}
    write_json(run/'metrics.json',result)
    del model; gc.collect(); torch.cuda.empty_cache()
    return result


def baseline_correct(p,role):
    name='seen' if role=='official_seen' else 'unseen'
    with (Path(p['config']['baseline_run'])/f'predictions_{name}.csv').open(encoding='utf-8',newline='') as f:
        bank={r['filename']:r for r in csv.DictReader(f)}
    rows=p['roles'][role]
    if len(bank)!=len(rows): raise ValueError('Baseline prediction length changed')
    for r in rows:
        if int(bank[r['filename']]['true_composition_id'])!=r['composition_id']: raise ValueError('Baseline labels changed')
    correct=np.array([int(bank[r['filename']]['top1_id'])==r['composition_id'] for r in rows])
    expected=json.loads((Path(p['config']['baseline_run'])/'metrics.json').read_text())[name]['composition']['top1']
    if int(correct.sum())!=round(expected*len(rows)): raise ValueError('Baseline correct count mismatch')
    return correct


def analyze(p,cfg,run,result):
    pids=np.array([c['composition_id'] for c in p['ontology']['candidates']]); point={}; draws={}
    for role in ('official_seen','official_unseen'):
        bank=np.load(run/f'logits_{role}.npz')
        a=pids[bank['pair'].argmax(1)]==bank['composition_targets']; b=baseline_correct(p,role)
        point[role]={'new_correct':int(a.sum()),'baseline_correct':int(b.sum()),'delta_pp':float(100*(a.mean()-b.mean()))}
        for clustered in (False,True): draws[(role,clustered)]=paired_bootstrap(a,b,p['roles'][role],cfg,clustered)
    intervals={}
    for clustered in (False,True):
        sa,sb=draws[('official_seen',clustered)]; ua,ub=draws[('official_unseen',clustered)]
        ha=np.divide(2*sa*ua,sa+ua,out=np.zeros_like(sa),where=sa+ua>0)
        hb=np.divide(2*sb*ub,sb+ub,out=np.zeros_like(sb),where=sb+ub>0)
        intervals['composition' if clustered else 'image']={
            'unseen_delta_95_pp':np.percentile(100*(ua-ub),[2.5,97.5]).tolist(),
            'H_delta_95_pp':np.percentile(100*(ha-hb),[2.5,97.5]).tolist()}
    value={'reference':'saved original independent baseline','point':point,'intervals':intervals,'full_training_seeds':1}
    write_json(run/'paired_analysis.json',value); return value


def reports(p,cfg,run,result,analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=json.loads((run/ARM/'history.json').read_text()); summary=json.loads((run/ARM/'summary.json').read_text())
    epochs=[h['epoch'] for h in history]; fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot(epochs,[h['training']['loss'] for h in history],label='Training conditional CE (smoothed)')
    for key in ('crop','disease','composition'): axes[1].plot(epochs,[100*h['validation']['metrics'][key]['top1'] for h in history],label=key)
    axes[2].plot(epochs,[h['validation']['mean_ce'] for h in history],label='Marginal disease validation CE')
    for ax,title in zip(axes,('Training loss','Seen-validation Top-1 (%)','Checkpoint tie-break loss')):
        ax.axvline(summary['selected_epoch'],color='gray',ls=':',label='Selected epoch'); ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run/'training_curves.png',dpi=160); plt.close(fig)
    base=json.loads((Path(cfg['baseline_run'])/'metrics.json').read_text())
    lines=['# Crop-conditioned disease recognition results','', '| Model | Seen Top-1 | Unseen Top-1 | H |','|---|---:|---:|---:|',
        f"| Saved baseline | {100*base['seen']['composition']['top1']:.2f} | {100*base['unseen']['composition']['top1']:.2f} | {100*base['harmonic_mean']['top1']:.2f} |",
        f"| Crop-conditioned disease | {100*result['official_seen']['composition']['top1']:.2f} | {100*result['official_unseen']['composition']['top1']:.2f} | {100*result['H']['top1']:.2f} |",'',
        f"Trained {summary['epochs']} epochs; selected epoch {summary['selected_epoch']}. Epoch 0 is the identity-initialized saved baseline.",
        f"Recorded training/validation epoch time: {summary['seconds']/3600:.2f} hours.",'',
        'Original 6,746/1,254 split; no pseudo-unseen validation. Reused frozen crop checkpoint; fully trainable disease CNN starts from the saved disease baseline.',
        'Shared crop embeddings produce bounded feature scale/bias adjustments; one shared disease classifier. This is a plant-specific adaptation, not a reproduction of a published method.',
        'Training CE uses the fitting crop label. Prediction considers all 55 crops, normalizes disease probabilities separately for each crop, and scores all 333 candidate pairs. No true crop or seen/unseen membership supplied at inference.',
        'Checkpoint selection: original seen-validation disease Top-1, then lower ordinary disease CE; use the predicted-crop-weighted disease marginal. Includes the initial checkpoint. Official scores do not choose any setting.',
        'This tests warm-started conditional fine-tuning as a complete recipe; improvements cannot be attributed to conditioning alone without a separately authorized matched continuation control.',
        'One training seed; familiar official tests and bootstrap intervals do not measure new-seed variance. Plot visual review remains pending.',
        '',json.dumps(analysis,indent=2),'','Historical baseline artifacts are unchanged. No additional training is queued.']
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
            if shutil.disk_usage(run).free<2*1024**3: raise RuntimeError('Need 2 GiB free for atomic checkpoints')
            if not (run/'preflight.json').exists(): preflight(p,cfg,run)
            if args.preflight_only: return
            crop=cache_crop(p,cfg,run,('seen_validation',))['seen_validation']
            train_model(p,cfg,run,crop); freeze(p,run)
            result=evaluate(p,cfg,run); analysis=analyze(p,cfg,run,result); reports(p,cfg,run,result,analysis)
            from verify import verify_run
            check=verify_run(run); status(run,'complete',verification_checks=check['checks'],results=str(run/'results.md'))
        except BaseException as error:
            status(run,'failed',error=repr(error),traceback=traceback.format_exc()); raise


if __name__=='__main__': main()
