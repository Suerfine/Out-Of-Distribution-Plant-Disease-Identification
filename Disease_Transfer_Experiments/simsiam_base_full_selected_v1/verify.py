"""Independent final verification of SSL provenance, coverage and Base scoring."""
import csv,json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from protocol import seed_all,sha256,tensor_hash,write_json,stamp
from storage import CheckpointStore
from model import SimSiam
from data import epoch_indices
from independent_cnn_baseline.models import IndependentNet


def numpy_metrics(scores,y):
    rank=np.argsort(-scores,axis=1,kind='stable'); hit=rank[:,0]==y
    return {'top1':float(hit.mean()),'top5':float((rank[:,:5]==y[:,None]).any(1).mean()),
            'macro_top1':float(np.mean([hit[y==c].mean() for c in np.unique(y)]))},rank


def logsoftmax64(x):
    x=x.astype(np.float64); x=x-x.max(1,keepdims=True)
    return x-np.log(np.exp(x).sum(1,keepdims=True))


def verify_run(run):
    run=Path(run); p=json.loads((run/'prepared.json').read_text()); cfg=p['config']; o=p['ontology']; roles=p['roles']
    report=json.loads((run/'metrics.json').read_text()); frozen=json.loads((run/'selection_frozen_before_test.json').read_text()); checks=0
    def check(v,why):
        nonlocal checks
        checks+=1
        if not v: raise AssertionError(why)
    def close(a,b,why): check(abs(float(a)-float(b))<1e-10,why)
    for field in ('source_hashes','historical_hashes','csv_hashes','manifest_hashes','image_hashes','cache_hashes','ssl_source_hashes'):
        for path,h in p[field].items(): check(sha256(path)==h,'Changed '+field+': '+path)
    check(sha256(p['pretrained_path'])==p['pretrained_sha256']==p['baseline_config']['model']['pretrained_sha256'],'Same original ImageNet initialization')
    check(set(roles)=={'fit','official_seen','official_unseen'},'No added validation partition')
    check([len(roles[r]) for r in ('fit','official_seen','official_unseen')]==[8000,2000,1448],'Official splits')
    with open(next(iter(p['manifest_hashes'])),newline='',encoding='utf-8-sig') as f: original=list(csv.DictReader(f))
    check([r['file'] for r in original]==[r['filename'] for r in roles['fit']],'Exact Base full-training membership')
    fit_comp={r['composition_id'] for r in roles['fit']}; unseen={r['composition_id'] for r in roles['official_unseen']}
    check(len(fit_comp)==323 and len(unseen)==10 and not fit_comp&unseen,'Unseen pairs excluded from fitting')
    check(len(o['candidates'])==333,'All candidates retained')
    train_hashes={p['image_hashes'][r['original_path']] for r in roles['fit']}
    for role in ('official_seen','official_unseen'): check(not train_hashes&{p['image_hashes'][r['original_path']] for r in roles[role]},'No exact duplicate content in train/test')
    state=CheckpointStore(run,p['fingerprint']).load(); check(state['stage']=='evaluate','All training stages completed')
    for stage,epochs,size in (('ssl',cfg['ssl_epochs'],64),('finetune',cfg['finetune_epochs'],32)):
        history=json.loads((run/stage/'history.json').read_text())
        check(history==state['histories'][stage] and len(history)==epochs,'Fixed stage history '+stage)
        for h in history:
            path=run/stage/f"exposures_epoch{h['epoch']:02}.npz"; bank=np.load(path)
            check(sha256(path)==h['exposure_sha256'],'Exposure hash')
            check(np.array_equal(bank['indices'],epoch_indices(8000,cfg,stage,h['epoch']-1)),'Deterministic full shuffle')
            check(np.array_equal(np.sort(bank['indices']),np.arange(8000)),'Every fitting image once')
            check(np.array_equal(bank['image_counts'],np.ones(8000,dtype=int)),'No replacement sampling')
            t=h['training']; check(t['samples']==8000 and t['steps']==8000//size and t['skips']<t['steps'],'Full stage exposure/applied updates')
            if stage=='ssl':
                v=h['monitor']; check(v['encoder_std']>=cfg['collapse_std_floor'] and v['projector_std']>=cfg['collapse_std_floor'] and v['encoder_effective_rank']>=cfg['collapse_effective_rank_floor'],'No monitored SSL collapse')
    preflight=json.loads((run/'preflight.json').read_text())
    check(preflight['passed'] and preflight['supervised_backbone_changed'] and preflight['checkpoint_next_update_exact'],'Supervised GPU update and exact restart probe')
    source_probe=preflight['reused_ssl_preflight']
    check(source_probe['passed'] and source_probe['gradient_cache_replay_exact'] and source_probe['head_batch_sizes']==[64]
        and preflight['backbone_parameters']>80_000_000,'Verified reused64-image SSL heads on Base')
    init=json.loads((run/'ssl_initialization.json').read_text()); seed_all(cfg['seed']); model=SimSiam(p['baseline_config'],cfg,True)
    check(tensor_hash(model.encoder.state_dict())==init['encoder_hash'],'SSL begins from pinned ImageNet Base')
    ssl=torch.load(run/'ssl_encoder.pt',map_location='cpu',weights_only=True)
    check(ssl['epochs']==20 and ssl['fingerprint']==p['ssl_reuse']['fingerprint'],'Reused full-data SSL completion and source provenance')
    check(sha256(run/'ssl_encoder.pt')==p['ssl_reuse']['encoder_sha256'],'Reused SSL artifact copied byte identically')
    reuse=json.loads((run/'ssl_reuse.json').read_text())
    check(reuse['ssl_updates_repeated']==0 and reuse['recipe_and_full_data_match'],'Same SSL recipe and8000-image data without repeated updates')
    ssl_changed=sum(int(torch.count_nonzero(v!=ssl['encoder'][k])) for k,v in model.encoder.state_dict().items()); check(ssl_changed>0,'SSL trained the full Base encoder')
    del model
    ft_init=json.loads((run/'finetune_initialization.json').read_text()); check(tensor_hash(ssl['encoder'])==ft_init['encoder_hash'],'Exact SSL-to-supervised encoder transfer')
    check(sha256(run/'ssl_encoder.pt')==ft_init['source_ssl_encoder_sha256'],'Frozen SSL encoder source')
    seed_all(cfg['seed']); original_head=IndependentNet(p['baseline_config'],'disease',pretrained=True)
    check(tensor_hash(original_head.head.state_dict())==ft_init['head_hash'],'Original disease-head initialization seed')
    del original_head
    final=torch.load(run/'disease_final.pt',map_location='cpu',weights_only=True)
    check(final['fingerprint']==p['fingerprint'] and final['ssl_epochs']==20 and final['finetune_epochs']==cfg['finetune_epochs']==p['ssl_reuse']['selected_supervised_epochs'],'Development-selected fixed final model')
    ft_changed=sum(int(torch.count_nonzero(v!=final['model']['backbone.'+k])) for k,v in ssl['encoder'].items()); check(ft_changed>0,'Whole disease backbone fine-tuned after SSL')
    check(not any(k.startswith(('projector.','predictor.')) for k in final['model']),'SSL heads removed at inference')
    for v in final['model'].values(): check(bool(torch.isfinite(v).all()),'Finite final model')
    del ssl,final
    check(frozen['fingerprint']==p['fingerprint'] and not frozen['official_scores_used'],'No official-score selection')
    check(sha256(run/'disease_final.pt')==frozen['checkpoint_sha256'],'Final frozen checksum')
    base=Path(cfg['baseline_run']); old=np.load(base/'scores.npz'); analysis=json.loads((run/'paired_analysis.json').read_text())
    ids=np.array([c['composition_id'] for c in o['candidates']]); pm={v:i for i,v in enumerate(ids)}
    ci=np.array([c['crop_id'] for c in o['candidates']]); di=np.array([c['disease_id'] for c in o['candidates']]); mask=np.array([c in fit_comp for c in ids])
    for role,split in (('official_seen','test_seen'),('official_unseen','test_unseen')):
        path=run/f'logits_{role}.npz'; b=np.load(path); rows=roles[role]; y=np.array([pm[r['composition_id']] for r in rows])
        check(path.stat().st_mtime>=datetime.fromisoformat(frozen['frozen_utc']).timestamp(),'New disease test inference after freeze')
        check(b['checkpoint_hash'].item()==frozen['checkpoint_sha256'],'Prediction checkpoint provenance')
        check(b['filenames'].tolist()==[r['filename'] for r in rows] and b['composition_targets'].tolist()==[r['composition_id'] for r in rows],'Input/label alignment')
        check(np.allclose(b['log_crop'],logsoftmax64(b['crop']),atol=1e-5,rtol=0),'Crop probabilities')
        check(np.allclose(b['log_disease'],logsoftmax64(b['disease']),atol=1e-5,rtol=0),'Disease probabilities')
        pair=b['log_crop'][:,ci]+b['log_disease'][:,di]; check(np.array_equal(pair,b['pair']) and np.isfinite(pair).all(),'Independent exact FP32 pair product')
        crop=np.load(run/f'crop_{role}.npz'); check(np.array_equal(crop['logits'],b['crop']) and crop['checkpoint_hash'].item()==p['crop_checkpoint_sha256'],'Unchanged Base crop logits')
        for gamma,suffix in ((0.,''),(6.2,'_fixed_gamma')):
            scores=pair.astype(np.float64)-gamma*mask[None,:]
            for key,x,target in (('composition',scores,y),('crop',b['crop'],np.array([r['crop_id'] for r in rows])),('disease',b['disease'],np.array([r['disease_id'] for r in rows]))):
                values,ranking=numpy_metrics(x,target)
                for metric,value in values.items(): close(value,report[role+suffix][key][metric],'Independent metrics '+role+suffix+'/'+key+'/'+metric)
                if key=='composition': pair_rank=ranking
            prior=(old[split+'_scores'].astype(np.float64)-gamma*mask[None,:]).argmax(1)==old[split+'_labels']; point=analysis[str(gamma)]['point'][role]
            check(int(prior.sum())==point['baseline_correct'] and int((pair_rank[:,0]==y).sum())==point['new_correct'],'Paired Base comparison counts')
            reference=json.loads((base/('metrics.json' if gamma else 'uncalibrated_metrics.json')).read_text()); close(prior.mean(),reference[split]['top1'],'Saved Base score reproduced')
            if not gamma:
                with (run/f'predictions_{role}.csv').open(newline='',encoding='utf-8') as f: exported=list(csv.DictReader(f))
                check(len(exported)==len(rows),'Prediction count')
                for i,r in enumerate(exported):
                    check(r['filename']==rows[i]['filename'],'Prediction filename')
                    check(int(r['predicted_composition_id'])==ids[pair_rank[i,0]],'Independent exported Top1')
                    check(json.loads(r['top5_ids'])==ids[pair_rank[i,:5]].tolist(),'Independent exported Top5')
    for suffix,hkey in (('','H'),('_fixed_gamma','H_fixed_gamma')):
        for metric in ('top1','top5','macro_top1'):
            s=report['official_seen'+suffix]['composition'][metric]; u=report['official_unseen'+suffix]['composition'][metric]
            close(2*s*u/(s+u) if s+u else 0,report[hkey][metric],'Independent H')
    value={'passed':True,'checks':checks,'utc':stamp(),'model':cfg['model'],'ssl_backbone_values_changed':ssl_changed,'finetune_backbone_values_changed':ft_changed,
        'original_base_unchanged':True,'no_test_image_ssl':True,'fixed_epochs':[20,cfg['finetune_epochs']],
        'full_data_ssl_reused':True,'ssl_source_checks':p['ssl_reuse']['source_verification_checks'],
        'verification':'Full-data SSL reuse provenance, development-selected budget, deterministic coverage, SSL transfer and independent NumPy scoring'}
    write_json(run/'independent_verification.json',value); return value
