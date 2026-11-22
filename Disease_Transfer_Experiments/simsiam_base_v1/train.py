"""Fixed-budget label-free SimSiam adaptation, then supervised Base training."""
import argparse
import copy
import gc
import json
import math
import os
import threading
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from protocol import (HERE,StudyLock,load_config,prepare,seed_all,rng_state,restore_rng,status,
                      tensor_hash,transforms_for,sha256,stamp,write_json)
from storage import CheckpointStore,checked_save,LowDiskSpace,IntegrityError,atomic_json
from data import training_loader,epoch_indices
from model import SimSiam,ssl_optimizer,cache_backward
from independent_cnn_baseline.models import IndependentNet,optimizer as base_optimizer
from independent_cnn_baseline.engine import lr_factor as base_lr_factor


class ScientificFailure(RuntimeError): pass


def amp(cfg): return torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp'])


def make_scaler(cfg): return torch.amp.GradScaler('cuda',enabled=cfg['amp'],init_scale=cfg['amp_initial_scale'])


def apply_step(model,opt,scaler,cfg):
    previous=scaler.get_scale(); scaler.unscale_(opt)
    norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not cfg['amp'] and not torch.isfinite(norm): raise ScientificFailure('Nonfinite unscaled gradients')
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    return int(scaler.get_scale()<previous)


def finetune_backward(model,images,labels,micro,cfg,scaler):
    loss_sum=correct=0
    for start in range(0,len(images),micro):
        y=labels[start:start+micro].to(cfg['device'])
        with amp(cfg):
            logits=model(images[start:start+micro].to(cfg['device'],non_blocking=True))
            loss=F.cross_entropy(logits.float(),y,label_smoothing=cfg['label_smoothing'],reduction='sum')
        if not torch.isfinite(loss): raise ScientificFailure('Nonfinite supervised loss')
        scaler.scale(loss/len(images)).backward()
        loss_sum+=float(loss.detach()); correct+=int(logits.detach().argmax(1).eq(y).sum())
    return loss_sum,correct


class MonitorRows(Dataset):
    def __init__(self,rows): self.rows=rows; self.transform=transforms_for(None,224)[1]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['path']) as im: return self.transform(im.convert('RGB'))


@torch.inference_mode()
def monitor(model,p,cfg):
    # Fitting images only, no labels, accuracy, validation or official inputs.
    ids=np.random.default_rng(cfg['seed']+999).choice(len(p['roles']['fit']),cfg['monitor_images'],replace=False)
    rows=[p['roles']['fit'][i] for i in ids]; features=[]; projected=[]; model.eval()
    for images in DataLoader(MonitorRows(rows),batch_size=cfg['micro_batch'],num_workers=0):
        with amp(cfg):
            f=model.encoder(images.to(cfg['device'])); z=model.projector(f.float())
        features.append(F.normalize(f.float(),dim=1).cpu()); projected.append(F.normalize(z.float(),dim=1).cpu())
    f=torch.cat(features).double(); z=torch.cat(projected).double(); centered=f-f.mean(0)
    eigen=torch.linalg.eigvalsh(centered@centered.T).clamp_min(0); probabilities=eigen/eigen.sum().clamp_min(1e-20)
    rank=float(torch.exp(-(probabilities*probabilities.clamp_min(1e-20).log()).sum()))
    return {'encoder_std':float(f.std(0,unbiased=False).mean()),'projector_std':float(z.std(0,unbiased=False).mean()),
        'encoder_effective_rank':rank,'images':len(rows),'source':'unlabelled fitting images only'}


def check_monitor(value,cfg):
    if min(value['encoder_std'],value['projector_std'])<cfg['collapse_std_floor'] or value['encoder_effective_rank']<cfg['collapse_effective_rank_floor']:
        raise ScientificFailure('SSL representation collapse detected: '+str(value))


def initial_ssl(p,cfg):
    seed_all(cfg['seed']); model=SimSiam(p['baseline_config'],cfg,True).to(cfg['device'])
    for m in model.encoder.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm) or getattr(m,'drop_prob',0)>0 or (isinstance(m,nn.Dropout) and m.p>0):
            raise IntegrityError('Gradient caching requires deterministic, batch-independent encoder')
    return model


def initialise_finetune(p,cfg,encoder):
    seed_all(cfg['seed']); model=IndependentNet(p['baseline_config'],'disease',pretrained=True)
    head_hash=tensor_hash(model.head.state_dict()); model.backbone.load_state_dict(encoder,strict=True)
    return model.to(cfg['device']),head_hash


def preflight(p,cfg,run):
    model=initial_ssl(p,cfg); opt=ssl_optimizer(model,cfg); scaler=make_scaler(cfg)
    local={**cfg,'workers':0}; data,_=training_loader(p['roles']['fit'],'ssl',0,0,local,p['baseline_config'])
    v1,v2,_=next(iter(data)); del data
    initial=tensor_hash(model.encoder.state_dict()); head_sizes=[]
    hooks=[m.register_forward_pre_hook(lambda m,args:head_sizes.append(len(args[0]))) for m in model.projector.modules() if isinstance(m,nn.BatchNorm1d)]
    applied=0; torch.cuda.reset_peak_memory_stats()
    for attempt in range(12):
        model.train(); opt.zero_grad(set_to_none=True)
        loss,detail=cache_backward(model,v1,v2,cfg['micro_batch'],cfg['device'],cfg['amp'],scaler,True)
        skipped=apply_step(model,opt,scaler,cfg)
        if not skipped: applied+=1
        if applied==2: break
    else: raise ScientificFailure('SSL GPU probe could not apply two finite updates')
    for h in hooks: h.remove()
    if initial==tensor_hash(model.encoder.state_dict()) or set(head_sizes)!={64}: raise ScientificFailure('SSL backbone update/head batch check failed')
    # Actual GPU optimizer/scaler/RNG state must survive a checked disk round trip.
    store=CheckpointStore(run/'preflight_checkpoints',p['fingerprint'],cfg['min_free_bytes'])
    payload={'fingerprint':p['fingerprint'],'stage':'probe','epoch':0,'cursor':0,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state()}
    store.save(payload); del payload
    def next_update():
        model.train(); opt.zero_grad(set_to_none=True)
        cache_backward(model,v1,v2,cfg['micro_batch'],cfg['device'],cfg['amp'],scaler,True); apply_step(model,opt,scaler,cfg)
        return tensor_hash(model.state_dict())
    expected=next_update(); saved=store.load(); model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng']); del saved
    replay=next_update()
    if expected!=replay: raise ScientificFailure('GPU checkpoint restart changed the next update')
    value=monitor(model,p,cfg); check_monitor(value,cfg)
    encoder={k:v.detach().cpu() for k,v in model.encoder.state_dict().items()}
    ssl_peak=torch.cuda.max_memory_allocated(); del model,opt,scaler,v1,v2; gc.collect(); torch.cuda.empty_cache()
    ft,head_hash=initialise_finetune(p,cfg,encoder); del encoder
    opt=base_optimizer(ft,p['baseline_config']); scaler=make_scaler(cfg)
    data,_=training_loader(p['roles']['fit'],'finetune',0,0,local,p['baseline_config']); images,labels,_=next(iter(data)); del data
    before=tensor_hash(ft.backbone.state_dict()); opt.zero_grad(set_to_none=True)
    for attempt in range(12):
        opt.zero_grad(set_to_none=True); ft.train(); finetune_backward(ft,images,labels,cfg['micro_batch'],cfg,scaler)
        if not apply_step(ft,opt,scaler,cfg): break
    else: raise ScientificFailure('Supervised GPU probe did not apply a finite update')
    if before==tensor_hash(ft.backbone.state_dict()): raise ScientificFailure('Supervised backbone did not update')
    result={'passed':True,'utc':stamp(),'model':cfg['model'],'backbone_parameters':sum(x.numel() for x in ft.backbone.parameters()),
        'head_batch_sizes':sorted(set(head_sizes)),'gradient_cache_replay_exact':True,'checkpoint_next_update_exact':True,
        'ssl_peak_bytes':ssl_peak,'feature_monitor':value,'supervised_backbone_changed':True,'updates_discarded':True}
    write_json(run/'preflight.json',result); status(run,'preflight_passed',**{k:v for k,v in result.items() if k not in ('utc',)})
    del ft,opt,scaler; gc.collect(); torch.cuda.empty_cache()


def run_training(p,cfg,run,micro):
    store=CheckpointStore(run,p['fingerprint'],cfg['min_free_bytes']); saved=store.load()
    if saved:
        for label,history in saved['histories'].items():
            (run/label).mkdir(exist_ok=True); write_json(run/label/'history.json',history)
            write_json(run/label/'summary.json',{'completed_epochs':len(history),'last':history[-1] if history else None})
    if saved and saved['stage']=='evaluate': return
    if saved:
        state={k:v for k,v in saved.items() if k not in ('model','optimizer','scaler','rng')}
        if state['stage']=='ssl': model=SimSiam(p['baseline_config'],cfg,False).to(cfg['device']); opt=ssl_optimizer(model,cfg)
        else: model=IndependentNet(p['baseline_config'],'disease',pretrained=False).to(cfg['device']); opt=base_optimizer(model,p['baseline_config'])
        scaler=make_scaler(cfg); model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng']); del saved
    else:
        model=initial_ssl(p,cfg); opt=ssl_optimizer(model,cfg); scaler=make_scaler(cfg)
        init=tensor_hash(model.encoder.state_dict())
        state={'fingerprint':p['fingerprint'],'stage':'ssl','epoch':0,'cursor':0,'histories':{'ssl':[],'finetune':[]},
            'totals':{'samples':0,'loss_sum':0.,'correct':0,'skips':0,'steps':0,'seconds':0.},'initial_encoder_hash':init}
        write_json(run/'ssl_initialization.json',{'encoder_hash':init,'source_pretrained_sha256':p['pretrained_sha256'],'model':cfg['model'],'ssl_labels_used':False})
    last_save=time.monotonic()
    def commit():
        nonlocal last_save
        status(run,'checkpointing',stage=state['stage'],epoch=state['epoch'],step=state['cursor'])
        payload={**state,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state()}
        store.save(payload); last_save=time.monotonic()
    if not store.pointer.exists(): commit()
    while True:
        stage=state['stage']; limit=cfg['ssl_epochs' if stage=='ssl' else 'finetune_epochs']
        if state['epoch']==limit:
            if stage=='finetune':
                final={'model':model.state_dict(),'fingerprint':p['fingerprint'],'ssl_epochs':cfg['ssl_epochs'],'finetune_epochs':limit}
                checked_save(run/'disease_final.pt',final,cfg['min_free_bytes'],run/'checkpoint_events.jsonl')
                frozen={'frozen_utc':stamp(),'fingerprint':p['fingerprint'],'checkpoint_sha256':sha256(run/'disease_final.pt'),'official_scores_used':False,'selection':'fixed20 SSL +15 supervised epochs'}
                write_json(run/'selection_frozen_before_test.json',frozen)
                state['stage']='evaluate'; store.save(state)
                del model,opt,scaler; gc.collect(); torch.cuda.empty_cache(); return
            encoder={k:v.detach().cpu() for k,v in model.encoder.state_dict().items()}
            checked_save(run/'ssl_encoder.pt',{'encoder':encoder,'fingerprint':p['fingerprint'],'epochs':limit},cfg['min_free_bytes'],run/'checkpoint_events.jsonl')
            encoder_hash=tensor_hash(encoder); del model,opt,scaler; gc.collect(); torch.cuda.empty_cache()
            model,head_hash=initialise_finetune(p,cfg,encoder); del encoder
            opt=base_optimizer(model,p['baseline_config']); scaler=make_scaler(cfg)
            write_json(run/'finetune_initialization.json',{'encoder_hash':encoder_hash,'head_hash':head_hash,'source_ssl_encoder_sha256':sha256(run/'ssl_encoder.pt'),'seed':cfg['seed']})
            state.update(stage='finetune',epoch=0,cursor=0,totals={'samples':0,'loss_sum':0.,'correct':0,'skips':0,'steps':0,'seconds':0.})
            commit(); continue
        epoch=state['epoch']; model.train()
        for group in opt.param_groups:
            if stage=='ssl':
                factor=(epoch+1)/cfg['ssl_warmup_epochs'] if epoch<cfg['ssl_warmup_epochs'] else .5*(1+math.cos(math.pi*(epoch-cfg['ssl_warmup_epochs'])/max(1,cfg['ssl_epochs']-cfg['ssl_warmup_epochs']-1)))
                group['lr']=group['base_lr']*(1. if group['constant'] else factor)
            else:
                factor=base_lr_factor(epoch+1,p['baseline_config']['training']); group['lr']=group['initial_lr']*factor
        data,indices=training_loader(p['roles']['fit'],stage,epoch,state['cursor'],cfg,p['baseline_config'])
        size=cfg['ssl_batch' if stage=='ssl' else 'finetune_batch']; total_steps=math.ceil(len(indices)/size)
        for batch in data:
            started=time.monotonic(); step=state['cursor']; opt.zero_grad(set_to_none=True)
            expected=indices[step*size:(step+1)*size]
            if not np.array_equal(batch[2].numpy(),expected): raise IntegrityError('Resumed image order differs')
            if stage=='ssl':
                loss,_=cache_backward(model,batch[0],batch[1],micro,cfg['device'],cfg['amp'],scaler)
                loss_sum=loss*len(expected); correct=0
            else: loss_sum,correct=finetune_backward(model,batch[0],batch[1],micro,cfg,scaler)
            skipped=apply_step(model,opt,scaler,cfg); t=state['totals']
            t['samples']+=len(expected); t['loss_sum']+=loss_sum; t['correct']+=correct; t['skips']+=skipped; t['steps']+=1; t['seconds']+=time.monotonic()-started
            state['cursor']=step+1
            if step==0 or (step+1)%10==0 or step+1==total_steps:
                status(run,'training',stage=stage,epoch=epoch+1,epochs=limit,step=step+1,total_steps=total_steps,loss=t['loss_sum']/t['samples'],micro_batch=micro)
            if time.monotonic()-last_save>=cfg['checkpoint_seconds']: commit()
            if (run/'STOP.request').exists():
                commit(); status(run,'paused',stage=stage,epoch=epoch+1,step=state['cursor']); return 'paused'
        del data
        t=state['totals']
        if t['samples']!=8000 or t['steps']!=total_steps or t['skips']==t['steps']: raise ScientificFailure('Incomplete fitting exposure or no applied updates')
        value={'epoch':epoch+1,'training':dict(t),'loss':t['loss_sum']/t['samples'],'seconds':t['seconds'],'micro_batch':micro}
        if stage=='ssl':
            value['monitor']=monitor(model,p,{**cfg,'micro_batch':micro}); check_monitor(value['monitor'],cfg)
        directory=run/stage; directory.mkdir(exist_ok=True)
        exposure=directory/f'exposures_epoch{epoch+1:02}.npz'
        np.savez_compressed(exposure,indices=indices,image_counts=np.bincount(indices,minlength=8000)); value['exposure_sha256']=sha256(exposure)
        state['histories'][stage].append(value); state['epoch']+=1; state['cursor']=0
        state['totals']={'samples':0,'loss_sum':0.,'correct':0,'skips':0,'steps':0,'seconds':0.}
        commit()
        for label,history in state['histories'].items():
            (run/label).mkdir(exist_ok=True); write_json(run/label/'history.json',history)
            write_json(run/label/'summary.json',{'completed_epochs':len(history),'last':history[-1] if history else None})
        status(run,'epoch_complete',stage=stage,epoch=state['epoch'],epochs=limit,loss=value['loss'])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default=str(HERE/'config.json')); parser.add_argument('--micro-batch',type=int); parser.add_argument('--preflight-only',action='store_true'); args=parser.parse_args()
    cfg=load_config(args.config); micro=args.micro_batch or cfg['micro_batch']
    if micro not in cfg['allowed_micro_batches']: raise ValueError('Unapproved microbatch')
    run=HERE/'runs'/cfg['run_name']; run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(2); os.environ['HF_HUB_OFFLINE']='1'
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    with StudyLock(run/'worker.lock'):
        try:
            status(run,'preparing',micro_batch=micro); p=prepare(cfg,run)
            if not (run/'preflight.json').exists(): preflight(p,cfg,run)
            if args.preflight_only: return
            outcome=run_training(p,cfg,run,micro)
            if outcome=='paused': return
            from evaluation import evaluate_and_report
            result=evaluate_and_report(p,cfg,run)
            from verify import verify_run
            check=verify_run(run); status(run,'complete',verification_checks=check['checks'],results=str(run/'results.md'))
        except torch.OutOfMemoryError as error:
            status(run,'retryable_cuda_oom',error=repr(error),micro_batch=micro); raise SystemExit(42)
        except LowDiskSpace as error:
            status(run,'waiting_for_disk_space',error=repr(error)); raise SystemExit(43)
        except (ScientificFailure,FloatingPointError) as error:
            status(run,'scientific_failure',error=repr(error),traceback=traceback.format_exc()); raise SystemExit(65)
        except (IntegrityError,ValueError,AssertionError) as error:
            status(run,'integrity_failure',error=repr(error),traceback=traceback.format_exc()); raise SystemExit(78)
        except BaseException as error:
            status(run,'failed',error=repr(error),traceback=traceback.format_exc()); raise


if __name__=='__main__': main()
