"""Independent probability, ranking, training and Base provenance checks."""
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from model import ConditionalDiseaseCNN
from protocol import seed_all, sha256, stamp, tensor_hash, write_json


def numpy_metrics(scores, targets):
    ranking=np.argsort(-scores,axis=1,kind='stable'); correct=ranking[:,0]==targets
    return {'top1':float(correct.mean()),'top5':float((ranking[:,:min(5,scores.shape[1])]==targets[:,None]).any(1).mean()),
            'macro_top1':float(np.mean([correct[targets==c].mean() for c in np.unique(targets)]))},ranking


def logsoftmax64(x,axis):
    x=x.astype(np.float64); x=x-x.max(axis=axis,keepdims=True)
    return x-np.log(np.exp(x).sum(axis=axis,keepdims=True))


def verify_run(run):
    run=Path(run); p=json.loads((run/'prepared.json').read_text()); cfg=p['config']; roles=p['roles']; o=p['ontology']
    report=json.loads((run/'metrics.json').read_text()); frozen=json.loads((run/'selection_frozen_before_test.json').read_text())
    checks=0
    def check(value,why):
        nonlocal checks
        checks+=1
        if not value: raise AssertionError(why)
    def close(a,b,why): check(abs(float(a)-float(b))<1e-10,why)
    for field in ('source_hashes','historical_hashes','csv_hashes','manifest_hashes','image_hashes','cache_hashes'):
        for path,h in p[field].items(): check(sha256(path)==h,'Changed '+field+': '+path)
    check(cfg['model']=='convnext_base.fb_in22k_ft_in1k','Base architecture required')
    check(set(roles)=={'fit','official_seen','official_unseen'},'No validation subset of already-fitted data')
    check([len(roles[k]) for k in ('fit','official_seen','official_unseen')]==[8000,2000,1448],'Official Base counts')
    manifest=next(iter(p['manifest_hashes']))
    with open(manifest,encoding='utf-8-sig',newline='') as f: original=list(csv.DictReader(f))
    check([r['file'] for r in original]==[r['filename'] for r in roles['fit']],'Exact Base all-training membership/order')
    train_hashes={p['image_hashes'][r['original_path']] for r in roles['fit']}
    for role in ('official_seen','official_unseen'):
        check(not train_hashes & {p['image_hashes'][r['original_path']] for r in roles[role]},'No exact training/test duplicate content')
    fit_comp={r['composition_id'] for r in roles['fit']}; unseen_comp={r['composition_id'] for r in roles['official_unseen']}
    check(len(fit_comp)==323 and len(unseen_comp)==10 and not fit_comp&unseen_comp,'Compositional split')
    check(len(o['candidates'])==333,'All candidates retained')
    for key in ('crop','disease'): check(set(o[key])=={r[key+'_id'] for r in roles['fit']},'All factors learned from fitting')
    check(frozen['fingerprint']==p['fingerprint'] and not frozen['official_scores_used'],'Checkpoint frozen without official selection')
    check(sha256(run/'conditioned/final.pt')==frozen['checkpoint_sha256'],'Frozen final checkpoint unchanged')
    final=torch.load(run/'conditioned/final.pt',map_location='cpu',weights_only=True)
    last=torch.load(run/'conditioned/last.pt',map_location='cpu',weights_only=True)
    history=json.loads((run/'conditioned/history.json').read_text())
    check(last['complete'] and last['epoch']==final['epoch']==frozen['selected_epoch']==cfg['epochs']==15,'Exactly15 additional epochs; final selection')
    check(last['history']==history and len(history)==15,'Complete training history')
    check(last['fingerprint']==final['fingerprint']==p['fingerprint'],'Training fingerprint')
    check(tensor_hash(last['model'])==tensor_hash(final['model']),'Final checkpoint is last training state')
    for v in final['model'].values(): check(bool(torch.isfinite(v).all()),'Finite final weights')
    seed_all(cfg['model_seed']); initial=ConditionalDiseaseCNN(cfg,o,warm_start=True)
    check(initial.net.backbone.num_features==1024,'Base feature width, not Tiny')
    check(sum(v.numel() for v in initial.net.backbone.parameters())>80_000_000,'Base backbone parameter count')
    check(all(v.requires_grad for v in initial.parameters()),'Whole modified disease network trainable')
    check(tensor_hash(initial.state_dict())==final['initial_hash'],'Warm-start initialization reproduced')
    source=torch.load(p['disease_checkpoint'],map_location='cpu',weights_only=True)
    check(tensor_hash(initial.net.state_dict())==tensor_hash(source),'Exact saved Base disease warm-start')
    changed=sum(int(torch.count_nonzero(v!=final['model']['net.backbone.'+k])) for k,v in initial.net.backbone.state_dict().items())
    check(changed>0,'Base disease backbone trained')
    trained_conditioner={k.removeprefix('conditioner.'):v for k,v in final['model'].items() if k.startswith('conditioner.')}
    check(tensor_hash(trained_conditioner)!=tensor_hash(initial.conditioner.state_dict()),'Conditioner trained')
    check(not any(k.startswith('crop_model.') for k in final['model']),'Crop CNN outside trainable model')
    del initial,source,last
    from independent_cnn_baseline.engine import lr_factor
    for h in history:
        path=run/'conditioned'/f"exposures_epoch{h['epoch']:02}.npz"; bank=np.load(path)
        check(sha256(path)==h['training']['exposure_sha256'],'Exposure hash')
        check(np.array_equal(np.sort(bank['indices']),np.arange(8000)),'Every fitting image once each epoch')
        check(np.array_equal(bank['image_counts'],np.ones(8000,dtype=int)),'No replacement sampling')
        check(h['training']['samples']==h['training']['unique_images']==8000,'Full exposure')
        check(h['training']['optimizer_steps']>h['training']['amp_skipped_steps'],'Applied training updates')
        expected=cfg['backbone_lr']*lr_factor(h['epoch'],{'warmup_epochs':3,'max_epochs':30,'min_lr_ratio':.01})
        close(h['backbone_lr'],expected,'Original Base LR horizon retained')
    cm={v:i for i,v in enumerate(o['crop'])}; dm={v:i for i,v in enumerate(o['disease'])}
    pm={c['composition_id']:i for i,c in enumerate(o['candidates'])}; ids=np.array(list(pm))
    ci=np.array([cm[c['crop_id']] for c in o['candidates']]); di=np.array([dm[c['disease_id']] for c in o['candidates']])
    seen_mask=np.array([c['composition_id'] in fit_comp for c in o['candidates']]); freeze_time=datetime.fromisoformat(frozen['frozen_utc']).timestamp()
    analysis=json.loads((run/'paired_analysis.json').read_text()); base=Path(cfg['baseline_run']); base_scores=np.load(base/'scores.npz')
    for role,split in (('official_seen','test_seen'),('official_unseen','test_unseen')):
        rows=roles[role]; path=run/f'logits_{role}.npz'; b=np.load(path)
        check(path.stat().st_mtime>=freeze_time,'New official disease inference after freeze')
        check(b['checkpoint_hash'].item()==frozen['checkpoint_sha256'],'Raw scores from final checkpoint')
        check(b['filenames'].tolist()==[r['filename'] for r in rows],'Image alignment')
        check(b['composition_targets'].tolist()==[r['composition_id'] for r in rows],'Label alignment')
        check(b['conditional'].shape==(len(rows),55,175),'All crop conditions exported')
        for key in ('pair','crop','disease','conditional','log_crop','log_conditional'): check(np.isfinite(b[key]).all(),'Finite scores '+key)
        # Independently validate normalizers in float64, then reconstruct the
        # deliberately rounded Base-compatible pair additions exactly in FP32.
        check(np.allclose(b['log_crop'],logsoftmax64(b['crop'],1),atol=1e-5,rtol=0),'Crop log probabilities')
        check(np.allclose(b['log_conditional'],logsoftmax64(b['conditional'],2),atol=1e-5,rtol=0),'Candidate-specific disease normalizers')
        pair=(b['log_crop'][:,ci]+b['log_conditional'][:,ci,di]).astype(np.float64)
        check(np.array_equal(pair,b['pair']),'Independent exact FP32 pair reconstruction')
        joint=b['log_crop'].astype(np.float64)[:,:,None]+b['log_conditional'].astype(np.float64); mx=joint.max(axis=1)
        marginal=mx+np.log(np.exp(joint-mx[:,None,:]).sum(axis=1))
        check(np.allclose(marginal,b['disease'],atol=1e-12,rtol=0),'Independent all-crop disease marginal')
        cropbank=np.load(run/f'crop_{role}.npz')
        check(cropbank['checkpoint_hash'].item()==p['crop_checkpoint_sha256'],'Frozen Base crop cache')
        check(np.array_equal(cropbank['logits'],b['crop']),'Unchanged Base crop logits actually used')
        check(cropbank['filenames'].tolist()==b['filenames'].tolist(),'Crop filename alignment')
        cy=np.array([cm[r['crop_id']] for r in rows]); dy=np.array([dm[r['disease_id']] for r in rows]); py=np.array([pm[r['composition_id']] for r in rows])
        for gamma,suffix in ((0.,''),(cfg['baseline_gamma'],'_fixed_gamma')):
            scores=pair-gamma*seen_mask[None,:]
            for key,x,y in (('composition',scores,py),('crop',b['crop'],cy),('disease',b['disease'],dy)):
                values,ranking=numpy_metrics(x,y)
                for metric,value in values.items(): close(value,report[role+suffix][key][metric],'Independent '+role+suffix+'/'+key+'/'+metric)
                if key=='composition': pair_ranking=ranking
            for cid,counts in report[role+suffix]['per_composition'].items():
                mask=b['composition_targets']==int(cid)
                check(int(mask.sum())==counts['images'],'Composition denominator')
                check(int((pair_ranking[mask,0]==py[mask]).sum())==counts['correct'],'Composition correctness')
            old=(base_scores[split+'_scores'].astype(np.float64)-gamma*seen_mask[None,:]).argmax(1)==base_scores[split+'_labels']
            point=analysis[str(gamma)]['point'][role]
            check(int(old.sum())==point['baseline_correct'],'Actual Base comparison count')
            check(int((pair_ranking[:,0]==py).sum())==point['new_correct'],'New comparison count')
            close(100*((pair_ranking[:,0]==py).mean()-old.mean()),point['delta_pp'],'Paired difference')
            reference=json.loads((base/('uncalibrated_metrics.json' if gamma==0 else 'metrics.json')).read_text())
            close(old.mean(),reference[split]['top1'],'Saved Base reference reproduced under same gamma')
            if gamma==0:
                with (run/f'predictions_{role}.csv').open(newline='',encoding='utf-8') as f: exported=list(csv.DictReader(f))
                check(len(exported)==len(rows),'Prediction count')
                for i,row in enumerate(exported):
                    check(row['filename']==rows[i]['filename'],'Prediction filename')
                    check(int(row['predicted_composition_id'])==ids[pair_ranking[i,0]],'Independent exported Top1')
                    check(json.loads(row['top5_ids'])==ids[pair_ranking[i,:5]].tolist(),'Independent exported Top5')
    for suffix,hkey in (('','H'),('_fixed_gamma','H_fixed_gamma')):
        for metric in ('top1','top5','macro_top1'):
            s=report['official_seen'+suffix]['composition'][metric]; u=report['official_unseen'+suffix]['composition'][metric]
            close(2*s*u/(s+u) if s+u else 0,report[hkey][metric],'H recomputation')
    check(report['fixed_gamma']==cfg['baseline_gamma']==6.2,'Existing frozen gamma, no retuning')
    value={'passed':True,'checks':checks,'at_utc':stamp(),'backbone_values_changed':changed,'model':cfg['model'],
        'fixed_additional_epochs':15,'original_base_artifacts_unchanged':True,'full_image_coverage':True,
        'verification':'NumPy normalizers, exact FP32 pair addition, rankings and both frozen scoring policies'}
    write_json(run/'independent_verification.json',value); return value


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path)
    torch.set_num_threads(4)
    print(json.dumps(verify_run(parser.parse_args().run),indent=2))
