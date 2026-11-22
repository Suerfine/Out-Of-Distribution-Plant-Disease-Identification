"""Train one of the two declared Base detail arms; select on raw development H."""
import argparse,gc,json,math,os,time,traceback
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from protocol import HERE,StudyLock,load_config,prepare,arm_fingerprint,rng_state,restore_rng,status,tensor_hash,sha256,stamp,write_json
from storage import CheckpointStore,checked_save,LowDiskSpace,IntegrityError
from data import training_loader
from model import DetailCNN,initialize,optimizer
from evaluation import validate_epoch,finalize,is_improvement,should_stop
from independent_cnn_baseline.engine import lr_factor


class ScientificFailure(RuntimeError):pass


def scaler_for(cfg):return torch.amp.GradScaler('cuda',enabled=cfg['amp'],init_scale=cfg['amp_initial_scale'])


def backward_batch(model,batch,cfg,micro,scaler=None):
    n=len(batch['label']);loss_sum=correct=0
    for start in range(0,n,micro):
        global_view=batch['global'][start:start+micro].to(cfg['device'],non_blocking=True)
        local=batch['local'][start:start+micro].to(cfg['device'],non_blocking=True) if 'local' in batch else None
        labels=batch['label'][start:start+micro].to(cfg['device'])
        context=torch.autocast('cuda',dtype=torch.float16) if cfg['amp'] else nullcontext()
        with context:
            logits=model(global_view,local)
            loss=F.cross_entropy(logits.float(),labels,label_smoothing=cfg['label_smoothing'],reduction='sum')
        if not torch.isfinite(loss):raise ScientificFailure('Nonfinite disease loss')
        (scaler.scale(loss/n) if scaler else loss/n).backward()
        loss_sum+=float(loss.detach());correct+=int((logits.detach().argmax(1)==labels).sum())
    return loss_sum,correct


def apply_step(model,opt,scaler):
    previous=scaler.get_scale();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.)
    scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
    return int(scaler.get_scale()<previous)


def preflight(p,cfg,root,arm,micro):
    run=root/arm;run.mkdir(exist_ok=True);status(run,'gpu_preflight',arm=arm)
    model=initialize(p,cfg,arm);opt=optimizer(model,p['baseline_config']);scaler=scaler_for(cfg)
    loader,_=training_loader(p,{**cfg,'workers':0},arm,0,0);batch=next(iter(loader));del loader
    names=['backbone','head']+(['attention','local_head'] if arm=='global_local' else [])
    before={name:tensor_hash(getattr(model,name).state_dict()) for name in names};attention_gradient=False
    torch.cuda.reset_peak_memory_stats();applied=0
    for attempt in range(12):
        model.train();opt.zero_grad(set_to_none=True);backward_batch(model,batch,cfg,micro,scaler)
        if arm=='global_local':
            attention_gradient|=any(x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(torch.count_nonzero(x.grad)) for x in model.attention.parameters())
        if not apply_step(model,opt,scaler):applied+=1
        if applied==2:break
    else:raise ScientificFailure('GPU probe could not apply two finite updates')
    changed={name:before[name]!=tensor_hash(getattr(model,name).state_dict()) for name in names}
    if not all(changed.values()) or (arm=='global_local' and not attention_gradient):raise ScientificFailure('Required branch did not learn: '+str(changed))
    fp=arm_fingerprint(p,arm);store=CheckpointStore(run/'preflight_checkpoints',fp,cfg['min_free_bytes'])
    store.save({'fingerprint':fp,'stage':'probe','epoch':0,'cursor':0,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state()})
    def next_update():
        model.train();opt.zero_grad(set_to_none=True);backward_batch(model,batch,cfg,micro,scaler);apply_step(model,opt,scaler)
        return tensor_hash(model.state_dict())
    expected=next_update();saved=store.load();model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler']);restore_rng(saved['rng']);del saved
    if next_update()!=expected:raise ScientificFailure('GPU checkpoint next-update replay differs')
    result={'passed':True,'utc':stamp(),'arm':arm,'fingerprint':fp,'backbone_parameters':sum(v.numel() for v in model.backbone.parameters()),
        'component_weights_changed':changed,'attention_has_gradient':attention_gradient if arm=='global_local' else None,
        'checkpoint_next_update_exact':True,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'updates_discarded':True,
        'micro_batch':micro,'global_size':cfg['image_size'],'patch_size':cfg['patch_size'] if arm=='global_local' else None}
    write_json(run/'preflight.json',result);status(run,'preflight_passed',**{k:v for k,v in result.items() if k!='utc'})
    del model,opt,scaler,batch;gc.collect();torch.cuda.empty_cache()


def totals():return {'samples':0,'loss_sum':0.,'correct':0,'steps':0,'skips':0,'seconds':0.}


def sync_history(run,state):
    h=state['history'];write_json(run/'history.json',h);write_json(run/'summary.json',{'completed_epochs':len(h),'last':h[-1] if h else None})


def run_training(p,cfg,root,arm,micro):
    run=root/arm;fp=arm_fingerprint(p,arm);store=CheckpointStore(run,fp,cfg['min_free_bytes']);saved=store.load()
    if saved:
        sync_history(run,saved)
        if saved['stage']=='report':return
        if saved['arm']!=arm:raise IntegrityError('Arm changed on resume')
        state={k:v for k,v in saved.items() if k not in ['model','optimizer','scaler','rng']}
        model=DetailCNN(p['baseline_config'],cfg,arm,pretrained=False).to(cfg['device']);opt=optimizer(model,p['baseline_config']);scaler=scaler_for(cfg)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);restore_rng(saved['rng']);del saved
    else:
        model=initialize(p,cfg,arm);opt=optimizer(model,p['baseline_config']);scaler=scaler_for(cfg)
        state={'fingerprint':fp,'arm':arm,'stage':'finetune','epoch':0,'cursor':0,'history':[],'best':None,'stale_epochs':0,'totals':totals()}
        write_json(run/'initialization.json',{'backbone_hash':tensor_hash(model.backbone.state_dict()),'head_hash':tensor_hash(model.head.state_dict()),
            'pretrained_sha256':p['pretrained_sha256'],'fresh_optimizer':True,'seed':cfg['seed'],'source':'pinned ImageNet; no plant-finetuned weights'})
    last_save=time.monotonic()
    def commit():
        nonlocal last_save
        status(run,'checkpointing',arm=arm,epoch=state['epoch'],step=state['cursor'])
        store.save({**state,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state()});last_save=time.monotonic()
    if not store.pointer.exists():commit()
    while not should_stop(state['epoch'],state['stale_epochs'],cfg):
        epoch=state['epoch'];model.train()
        for group in opt.param_groups:group['lr']=group['initial_lr']*lr_factor(epoch+1,p['baseline_config']['training'])
        loader,indices=training_loader(p,cfg,arm,epoch,state['cursor']);nsteps=math.ceil(len(indices)/cfg['finetune_batch'])
        for batch in loader:
            start=time.monotonic();step=state['cursor'];expected=indices[step*cfg['finetune_batch']:(step+1)*cfg['finetune_batch']]
            if not np.array_equal(batch['index'].numpy(),expected):raise IntegrityError('Primary image order changed')
            opt.zero_grad(set_to_none=True);loss,correct=backward_batch(model,batch,cfg,micro,scaler);skipped=apply_step(model,opt,scaler)
            t=state['totals'];t['samples']+=len(expected);t['loss_sum']+=loss;t['correct']+=correct;t['steps']+=1;t['skips']+=skipped;t['seconds']+=time.monotonic()-start
            state['cursor']+=1
            if step==0 or state['cursor']%10==0 or state['cursor']==nsteps:
                status(run,'training',arm=arm,epoch=epoch+1,epochs=cfg['finetune_epochs'],step=state['cursor'],total_steps=nsteps,loss=t['loss_sum']/t['samples'],micro_batch=micro)
            if time.monotonic()-last_save>=cfg['checkpoint_seconds']:commit()
            if (run/'STOP.request').exists() or (root/'STOP.request').exists():
                commit();status(run,'paused',arm=arm,epoch=epoch+1,step=state['cursor']);return 'paused'
        del loader
        t=state['totals']
        if t['samples']!=5798 or t['steps']!=nsteps or t['skips']==t['steps']:raise ScientificFailure('Incomplete epoch or no applied update')
        prior_rng=rng_state()
        try:validation,bank,bank_hash=validate_epoch(model,p,cfg,run,epoch+1)
        finally:restore_rng(prior_rng)
        score=validation['objective']['harmonic_mean_top1']
        if not math.isfinite(score):raise ScientificFailure('Nonfinite validation objective')
        if is_improvement(score,state['best']):
            directory=run/'best_models';directory.mkdir(exist_ok=True);path=directory/f'epoch_{epoch+1:02}.pt'
            digest=checked_save(path,{'model':model.state_dict(),'fingerprint':fp,'arm':arm,'epoch':epoch+1,'objective':validation['objective']},cfg['min_free_bytes'],run/'checkpoint_events.jsonl')
            state['best']={'epoch':epoch+1,'harmonic_mean_top1':score,'path':str(path),'sha256':digest,'validation_bank':bank,'validation_bank_sha256':bank_hash};state['stale_epochs']=0
        else:state['stale_epochs']+=1
        exposure=run/f'exposures_epoch{epoch+1:02}.npz';np.savez_compressed(exposure,indices=indices,image_counts=np.bincount(indices,minlength=5798))
        value={'epoch':epoch+1,'training':t,'loss':t['loss_sum']/5798,'seconds':t['seconds'],'micro_batch':micro,'validation':validation,'exposure_sha256':sha256(exposure)}
        state['history'].append(value);state['epoch']+=1;state['cursor']=0;state['totals']=totals();commit();sync_history(run,state)
        status(run,'epoch_complete',arm=arm,epoch=state['epoch'],epochs=cfg['finetune_epochs'],validation_H=score,best_epoch=state['best']['epoch'],stale_epochs=state['stale_epochs'])
    if state['best'] is None:raise ScientificFailure('No selected checkpoint')
    write_json(run/'selection_frozen.json',{'frozen_utc':stamp(),'fingerprint':fp,'arm':arm,'best':state['best'],'official_scores_used':False,
        'completed_epochs':state['epoch'],'selection':'maximum gamma0 validation H; earliest exact tie','stop_reason':'patience8' if state['stale_epochs']>=cfg['patience'] else 'max30'})
    state['stage']='report';store.save(state);del model,opt,scaler;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(HERE/'config.json'));parser.add_argument('--arm',choices=['resolution384','global_local'],default='resolution384')
    parser.add_argument('--micro-batch',type=int);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    cfg=load_config(args.config);micro=args.micro_batch or cfg['micro_batch'];arm=args.arm
    if micro not in cfg['allowed_micro_batches']:raise ValueError('Unapproved microbatch')
    root=HERE/'runs'/cfg['run_name'];root.mkdir(parents=True,exist_ok=True);run=root/arm;run.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(2);os.environ['HF_HUB_OFFLINE']='1'
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    with StudyLock(root/'worker.lock'):
        try:
            status(run,'preparing',arm=arm);p=prepare(cfg,root)
            if args.prepare_only:return
            for candidate in cfg['arms']:
                if not (root/candidate/'preflight.json').exists():
                    status(run,'preflighting',arm=arm,candidate=candidate)
                    preflight(p,cfg,root,candidate,micro)
            if args.preflight_only:return
            if run_training(p,cfg,root,arm,micro)=='paused':return
            check=finalize(p,cfg,run,arm);status(run,'complete',arm=arm,verification_checks=check['checks'],results=str(run/'results.md'))
        except torch.OutOfMemoryError as error:status(run,'retryable_cuda_oom',arm=arm,error=repr(error));raise SystemExit(42)
        except LowDiskSpace as error:status(run,'waiting_for_disk_space',arm=arm,error=repr(error));raise SystemExit(43)
        except (ScientificFailure,FloatingPointError) as error:status(run,'scientific_failure',arm=arm,error=repr(error),traceback=traceback.format_exc());raise SystemExit(65)
        except (IntegrityError,ValueError,AssertionError) as error:status(run,'integrity_failure',arm=arm,error=repr(error),traceback=traceback.format_exc());raise SystemExit(78)
        except BaseException as error:status(run,'failed',arm=arm,error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
