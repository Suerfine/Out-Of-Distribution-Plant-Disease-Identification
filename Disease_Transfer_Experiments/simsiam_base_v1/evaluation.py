"""Unchanged Base factor scoring, frozen gamma comparison and automatic report."""
import csv,gc,json
from pathlib import Path
import numpy as np
import torch
from protocol import loader,transforms_for,status,sha256,metrics,harmonic,paired_bootstrap,write_json
from independent_cnn_baseline.models import IndependentNet


@torch.inference_mode()
def evaluate_and_report(p,cfg,run):
    frozen=json.loads((run/'selection_frozen_before_test.json').read_text())
    if frozen['checkpoint_sha256']!=sha256(run/'disease_final.pt'): raise ValueError('Frozen disease checkpoint changed')
    saved=torch.load(run/'disease_final.pt',map_location='cpu',weights_only=True)
    model=IndependentNet(p['baseline_config'],'disease',pretrained=False)
    model.load_state_dict(saved['model']); del saved
    model=model.to(cfg['device']).eval(); transform=transforms_for(None,224)[1]
    crops=cache_crop(p,cfg,run,('official_seen','official_unseen'))
    result={'candidate_count':333,'fixed_gamma':cfg['baseline_gamma'],'model':cfg['model'],'ssl_epochs':cfg['ssl_epochs'],'finetune_epochs':cfg['finetune_epochs']}
    cm={v:i for i,v in enumerate(p['ontology']['crop'])}; dm={v:i for i,v in enumerate(p['ontology']['disease'])}
    candidates=p['ontology']['candidates']; ids=np.array([c['composition_id'] for c in candidates])
    ci=np.array([cm[c['crop_id']] for c in candidates]); di=np.array([dm[c['disease_id']] for c in candidates])
    seen={r['composition_id'] for r in p['roles']['fit']}; seen_mask=torch.tensor([c['composition_id'] in seen for c in candidates])
    for role in ('official_seen','official_unseen'):
        rows=p['roles'][role]; path=run/f'logits_{role}.npz'
        status(run,'final_inference',role=role)
        if path.exists():
            bank=np.load(path)
            if bank['checkpoint_hash'].item()!=frozen['checkpoint_sha256'] or bank['filenames'].tolist()!=[r['filename'] for r in rows]: raise ValueError('Prediction cache differs')
            crop=torch.from_numpy(bank['crop'].copy()); disease=torch.from_numpy(bank['disease'].copy()); pair=torch.from_numpy(bank['pair'].copy())
        else:
            values=[]
            for step,(images,_) in enumerate(loader(rows,{**cfg,'batch_size':1},p['ontology'],transform),1):
                with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']): logits=model(images.to(cfg['device'],non_blocking=True))
                values.append(logits.float().cpu())
                if step%100==0: status(run,'final_inference',role=role,step=step,total_steps=len(rows))
            disease=torch.cat(values); crop=crops[role]
            lc=torch.log_softmax(crop.float(),1); ld=torch.log_softmax(disease.float(),1); pair=lc[:,ci]+ld[:,di]
            np.savez_compressed(path,crop=crop.numpy(),disease=disease.numpy(),pair=pair.numpy(),log_crop=lc.numpy(),log_disease=ld.numpy(),
                filenames=[r['filename'] for r in rows],composition_targets=[r['composition_id'] for r in rows],checkpoint_hash=frozen['checkpoint_sha256'])
        if not torch.isfinite(pair).all() or not torch.isfinite(disease).all(): raise ValueError('Nonfinite inference scores')
        result[role]=metrics(crop,disease,rows,p['ontology'],pair)
        result[role+'_fixed_gamma']=metrics(crop,disease,rows,p['ontology'],pair.double()-cfg['baseline_gamma']*seen_mask.double()[None,:])
        ranking=pair.argsort(1,descending=True,stable=True)[:,:5].numpy()
        with (run/f'predictions_{role}.csv').open('w',newline='',encoding='utf-8') as f:
            out=csv.writer(f); out.writerow(['filename','true_composition_id','predicted_composition_id','top5_ids'])
            for row,rank in zip(rows,ranking): out.writerow([row['filename'],row['composition_id'],int(ids[rank[0]]),json.dumps(ids[rank].tolist())])
    for suffix,key in (('','H'),('_fixed_gamma','H_fixed_gamma')):
        result[key]={metric:harmonic(result['official_seen'+suffix]['composition'][metric],result['official_unseen'+suffix]['composition'][metric]) for metric in ('top1','top5','macro_top1')}
    write_json(run/'metrics.json',result); analysis=analyze(p,cfg,run,result)
    make_report(p,cfg,run,result,analysis)
    del model; gc.collect(); torch.cuda.empty_cache(); return result


def make_report(p,cfg,run,result,analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ssl=json.loads((run/'ssl/history.json').read_text()); ft=json.loads((run/'finetune/history.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot([h['epoch'] for h in ssl],[h['loss'] for h in ssl]); axes[0].set_title('SSL negative cosine loss')
    axes[1].plot([h['epoch'] for h in ssl],[h['monitor']['encoder_std'] for h in ssl],label='Encoder')
    axes[1].plot([h['epoch'] for h in ssl],[h['monitor']['projector_std'] for h in ssl],label='Projector'); axes[1].set_title('Fitting feature standard deviation'); axes[1].legend()
    axes[2].plot([h['epoch'] for h in ft],[h['loss'] for h in ft]); axes[2].set_title('Supervised disease CE (smoothed)')
    for ax in axes: ax.set_xlabel('Stage epoch'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run/'training_curves.png',dpi=160); plt.close(fig)
    base=Path(cfg['baseline_run']); old0=json.loads((base/'uncalibrated_metrics.json').read_text()); oldg=json.loads((base/'metrics.json').read_text())
    lines=['# SimSiam → supervised ConvNeXt-Base','', '| Model / scoring | Seen Top-1 | Unseen Top-1 | H |','|---|---:|---:|---:|']
    for name,v in [('Saved Base, gamma0',old0),('Saved Base, fixed gamma6.2',oldg)]: lines.append(f"| {name} | {100*v['test_seen']['top1']:.2f} | {100*v['test_unseen']['top1']:.2f} | {100*v['harmonic_mean_top1']:.2f} |")
    for name,suffix,key in [('SSL Base, gamma0','','H'),('SSL Base, same fixed gamma6.2','_fixed_gamma','H_fixed_gamma')]: lines.append(f"| {name} | {100*result['official_seen'+suffix]['composition']['top1']:.2f} | {100*result['official_unseen'+suffix]['composition']['top1']:.2f} | {100*result[key]['top1']:.2f} |")
    lines+=['', 'One experimental disease pipeline, starting from the same supervised ImageNet-pretrained Base encoder as the saved baseline. No new control or crop-model training.',
        '20 fixed epochs of label-free plant-image SimSiam adaptation, then15 supervised disease epochs with the original Base recipe. Both stages use only the8,000 official training images.',
        'SSL uses two views of the same image; no crop/disease labels, negative pairs, queue or teacher. Gradient caching retains a genuine64-image projector/predictor BatchNorm batch with small physical encoder microbatches. SSL DropPath is zero for exact replay.',
        'Supervised classification discards the SSL heads, restores original Base DropPath/dropout and initializes the disease head with the original seed. The entire disease encoder and head are fine-tuned.',
        'No validation partition, pseudo-unseen selection, test-image SSL, new calibration search or checkpoint selection. The existing crop model/logits and all333 candidate pairs are reused.',
        'This is a ConvNeXt-Base adaptation of SimSiam, not a reproduction of its ImageNet/ResNet100-epoch benchmark. The complete recipe uses extra SSL training; this comparison cannot isolate individual design choices.',
        'One seed; familiar official tests and paired bootstrap intervals do not quantify new-seed variation. Numerical verification follows. Training plot visual review is pending.',
        '',json.dumps(analysis,indent=2)]
    (run/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

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

