"""Full end-to-end compositional training, frozen evaluation and reporting."""
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

from model import CompositionalCNN,training_loss
from protocol import (HERE,EpochBatches,StudyLock,classification_metrics,harmonic,
                      load_config,loader,metrics,paired_bootstrap,prepare,restore_rng,
                      rng_state,save_checkpoint,seed_all,sha256,stamp,status,tensor_hash,
                      transforms_for,write_json)

ARM='compositional'


def autocast(cfg):
    return torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp'])


def images_to(images,cfg):
    return images.to(cfg['device'],non_blocking=True,memory_format=torch.contiguous_format)


def make_model(p,cfg,pretrained=True):
    seen={r['composition_id'] for r in p['roles']['fit']}
    return CompositionalCNN(cfg,p['ontology'],seen,pretrained).to(cfg['device'])


def optimizer_for(model,cfg):
    backbone={id(x) for x in model.net.parameters()}
    return torch.optim.AdamW([
        {'params':list(model.net.parameters()),'lr':cfg['backbone_lr'],'base_lr':cfg['backbone_lr']},
        {'params':[x for x in model.parameters() if id(x) not in backbone],
         'lr':cfg['head_lr'],'base_lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])


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
def predict(model,rows,p,cfg,transform):
    model.eval(); values={k:[] for k in ('pair','crop','disease')}
    for images,_ in loader(rows,cfg,p['ontology'],transform):
        with autocast(cfg): output=model(images_to(images,cfg))
        for k in values: values[k].append(output[k].float().cpu())
    values={k:torch.cat(v) for k,v in values.items()}
    if not all(torch.isfinite(v).all() for v in values.values()): raise RuntimeError('Nonfinite prediction')
    return values


def validation(model,p,cfg,transform):
    rows=p['roles']['seen_validation']; output=predict(model,rows,p,cfg,transform)
    dm={v:i for i,v in enumerate(p['ontology']['disease'])}
    y=torch.tensor([dm[r['disease_id']] for r in rows])
    pm={c['composition_id']:i for i,c in enumerate(p['ontology']['candidates'])}
    py=torch.tensor([pm[r['composition_id']] for r in rows])
    value=metrics(output['crop'],output['disease'],rows,p['ontology'],pair=output['pair'])
    return {'selection_score':value['composition']['top1'],
            'mean_ce':float(F.cross_entropy(output['pair'],py)),
            'disease_ce':float(F.cross_entropy(output['disease'],y)),'metrics':value}


def run_epoch(model,p,cfg,optimizer,scaler,transform,run,epoch):
    model.train(); rows=p['roles']['fit']
    sampler=EpochBatches(rows,cfg,grouped=False); sampler.epoch=epoch
    indices=sampler.indices(); data=loader(rows,cfg,p['ontology'],transform,sampler,epoch)
    totals={k:0. for k in ('loss','pair_ce','crop_ce','disease_ce')}
    n=correct_pair=correct_crop=correct_disease=skips=window=0
    optimizer.zero_grad(set_to_none=True)
    for batch,(images,labels) in enumerate(data,1):
        labels=labels.to(cfg['device'],non_blocking=True)
        with autocast(cfg):
            output=model(images_to(images,cfg))
            loss,parts=training_loss(output,labels,model.seen_indices,model.raw_to_global,model.global_to_seen,cfg)
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
        for k,v in parts.items(): totals[k]+=float(v.detach())
        pair_y=model.global_to_seen[model.raw_to_global[labels[:,2]]]
        correct_pair+=int(output['pair'][:,model.seen_indices].detach().argmax(1).eq(pair_y).sum())
        correct_crop+=int(output['crop'].detach().argmax(1).eq(labels[:,0]).sum())
        correct_disease+=int(output['disease'].detach().argmax(1).eq(labels[:,1]).sum()); n+=len(labels)
        if batch==1 or batch%100==0 or batch==len(data):
            status(run,'training',arm=ARM,epoch=epoch+1,batch=batch,total_batches=len(data),loss=totals['loss']/n)
    exposure=run/ARM/f'exposures_epoch{epoch+1:02}.npz'
    np.savez_compressed(exposure,indices=np.array(indices),image_counts=np.bincount(indices,minlength=len(rows)))
    return {**{k:v/n for k,v in totals.items()},'samples':n,'pair_top1_seen_training_candidates':correct_pair/n,
            'crop_top1':correct_crop/n,'disease_top1':correct_disease/n,'amp_skipped_steps':skips,
            'unique_images':len(set(indices)),'exposure_sha256':sha256(exposure)}


def train_model(p,cfg,run):
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
    best_key=tuple(saved['best_key']) if saved else (-1.,-float('inf'))
    best_epoch=saved['best_epoch'] if saved else None; stale=saved['stale'] if saved else 0
    if saved:
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng'])
    write_json(directory/'initialization.json',{'initial_hash':initial_hash,'model_seed':cfg['seed']+1,
        'pretrained_sha256':p['pretrained_sha256'],'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),
        'trainable_backbone_parameters':sum(x.numel() for x in model.net.parameters() if x.requires_grad),
        'preprocessing':preprocess,'train_transform':str(train_transform),'eval_transform':str(eval_transform)})
    del saved; gc.collect()
    for epoch in range(len(history),cfg['epochs']):
        start=time.perf_counter()
        for group in optimizer.param_groups: group['lr']=group['base_lr']*lr_factor(epoch,cfg)
        training=run_epoch(model,p,cfg,optimizer,scaler,train_transform,run,epoch)
        status(run,'validating',arm=ARM,epoch=epoch+1)
        val=validation(model,p,cfg,eval_transform)
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
            'epoch':epoch+1,'complete':complete,'fingerprint':p['fingerprint'],'initial_hash':initial_hash}
        save_checkpoint(directory/'last.pt',checkpoint); recover_reports(directory,checkpoint)
        status(run,'epoch_complete',arm=ARM,epoch=epoch+1,selected_epoch=best_epoch,
               selection_score=val['selection_score'],seconds=history[-1]['seconds'],complete=complete)
        del checkpoint
        if complete: break
    del model,optimizer,scaler; gc.collect(); torch.cuda.empty_cache()


def preflight(p,cfg,run):
    seed_all(cfg['seed']+1); model=make_model(p,cfg)
    transform,_,_=transforms_for(model.net,cfg['image_size'])
    sampler=EpochBatches(p['roles']['fit'],cfg); sampler.epoch=0
    images,labels=next(iter(loader(p['roles']['fit'],{**cfg,'workers':0},p['ontology'],transform,sampler)))
    labels=labels.to(cfg['device']); optimizer=optimizer_for(model,cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'],init_scale=cfg['amp_initial_scale'])
    before_net=tensor_hash(model.net.state_dict()); before_head=tensor_hash(model.composer.state_dict())
    torch.cuda.reset_peak_memory_stats()
    for attempt in range(14):
        optimizer.zero_grad(set_to_none=True)
        with autocast(cfg):
            output=model(images_to(images,cfg))
            loss,_=training_loss(output,labels,model.seen_indices,model.raw_to_global,model.global_to_seen,cfg)
        scaler.scale(loss/len(labels)).backward(); scaler.unscale_(optimizer)
        gradients={'backbone':float(model.net.stem[0].weight.grad.norm()),
            'composer':float(model.composer.combine[-1].weight.grad.norm()),
            'crop_embeddings':float(model.composer.crop.weight.grad.norm()),
            'disease_embeddings':float(model.composer.disease.weight.grad.norm())}
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer); scaler.update()
        if torch.isfinite(norm): break
    else: raise RuntimeError('GPU probe could not obtain finite applied update')
    after_net=tensor_hash(model.net.state_dict()); after_head=tensor_hash(model.composer.state_dict())
    if before_net==after_net or before_head==after_head or not all(math.isfinite(x) and x>0 for x in gradients.values()):
        raise RuntimeError('GPU probe failed to train encoder, primitives and composer')
    write_json(run/'preflight.json',{'passed':True,'at_utc':stamp(),'gradients':gradients,
        'backbone_changed':before_net!=after_net,'composer_changed':before_head!=after_head,
        'amp_skipped_probe_steps':attempt,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'pair_score_shape':list(output['pair'].shape),'discarded_probe_updates':True})
    status(run,'preflight_passed',peak_allocated_bytes=torch.cuda.max_memory_allocated())
    del model,optimizer,scaler,output,loss; gc.collect(); torch.cuda.empty_cache()


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
    _,transform,_=transforms_for(model.net,cfg['image_size'])
    result={'candidate_count':333,'score':'learned compositional cosine compatibility','metric_units':'fractions'}
    pids=np.array([c['composition_id'] for c in p['ontology']['candidates']])
    for role in ('seen_validation','official_seen','official_unseen'):
        rows=p['roles'][role]; path=run/f'logits_{role}.npz'
        status(run,'final_inference',role=role)
        if path.exists():
            bank=np.load(path)
            if bank['checkpoint_hash'].item()!=sha256(run/ARM/'best.pt'): raise ValueError('Logit cache mismatch')
            output={k:torch.from_numpy(bank[k].copy()) for k in ('pair','crop','disease')}
        else:
            output=predict(model,rows,p,cfg,transform)
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
    for key in ('pair_ce','crop_ce','disease_ce'):
        axes[0].plot(epochs,[h['training'][key] for h in history],label=key+' (smoothed)')
    for key in ('crop','disease','composition'):
        axes[1].plot(epochs,[100*h['validation']['metrics'][key]['top1'] for h in history],label=key)
    axes[2].plot(epochs,[h['validation']['mean_ce'] for h in history],label='Validation pair CE')
    for ax,title in zip(axes,('Training loss components','Validation Top-1 (%)','Checkpoint tie-break loss')):
        ax.axvline(summary['selected_epoch'],color='gray',ls=':',label='Selected epoch')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run/'training_curves.png',dpi=160); plt.close(fig)
    s=result['official_seen']['composition']['top1']; u=result['official_unseen']['composition']['top1']; h=result['H']['top1']
    baseline=json.loads((Path(cfg['baseline_run'])/'metrics.json').read_text())
    lines=['# Full learned compositional classifier results','',
        'Original 6,746/1,254 manifests; no pseudo-unseen validation. One fully trainable joint CNN, no retrained control.',
        'All 333 pair candidates use learned composition scores, with no calibration or split-membership information.','',
        '| Model | Seen Top-1 | Unseen Top-1 | H |','|---|---:|---:|---:|',
        f"| Saved baseline | {100*baseline['seen']['composition']['top1']:.2f} | {100*baseline['unseen']['composition']['top1']:.2f} | {100*baseline['harmonic_mean']['top1']:.2f} |",
        f'| Learned compositional classifier | {100*s:.2f} | {100*u:.2f} | {100*h:.2f} |','',
        f"Completed {summary['epochs']} epochs; selected epoch {summary['selected_epoch']} by seen-validation pair Top-1 then pair CE.",
        f"Recorded epoch time: {summary['seconds']/3600:.2f} hours, including training and validation.",
        f"Dynamic AMP skipped {sum(x['training']['amp_skipped_steps'] for x in history)} optimizer steps; final tensors are verified separately.",
        '', '## Paired comparison','',json.dumps(analysis,indent=2),'',
        'One seed and familiar official tests: intervals do not estimate variation across new trainings. All settings were fixed before training.',
        'This plant-specific shared-MLP composition architecture is inspired by CZSL principles; it is not an exact CGE or CompCos reproduction.',
        'Pair CE uses only 323 seen training candidates. All 333 candidates are scored at evaluation; primitive auxiliary heads do not replace the pair score.',
        'The experimental model has one shared image encoder; the saved baseline has two independent encoders. Protocol and model differences are reported, not attributed to a single isolated loss.',
        'Checkpoint adaptation: the original accuracy-first, lower-loss-tie-break rule is applied to the primary pair output, rather than the historical separate factor heads. This was fixed before training, without pseudo-unseen validation.',
        'All encoder and compositional parameters were trainable. The official tests did not choose the checkpoint or a scoring rule.',
        'Automatic numerical verification follows this report. Training plot visual review remains pending until inspected.',
        'The historical baseline remains unchanged. No further training is automatically queued after this study.']
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
            train_model(p,cfg,run); freeze(p,run)
            result=evaluate(p,cfg,run); analysis=analyze(p,cfg,run,result); reports(p,cfg,run,result,analysis)
            from verify import verify_run
            check=verify_run(run); status(run,'complete',verification_checks=check['checks'],results=str(run/'results.md'))
        except BaseException as error:
            status(run,'failed',error=repr(error),traceback=traceback.format_exc()); raise


if __name__=='__main__': main()
