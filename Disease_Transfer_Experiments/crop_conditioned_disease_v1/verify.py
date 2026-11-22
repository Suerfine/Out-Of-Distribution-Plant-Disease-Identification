"""Independent NumPy ranking, provenance, split and training verification."""
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from model import ConditionalDiseaseCNN
from protocol import seed_all,sha256,stamp,tensor_hash,write_json


def numpy_metrics(scores,targets):
    ranking=np.argsort(-scores,axis=1,kind='stable')
    correct=ranking[:,0]==targets
    return {'top1':float(correct.mean()),'top5':float((ranking[:,:min(5,scores.shape[1])]==targets[:,None]).any(1).mean()),
            'macro_top1':float(np.mean([correct[targets==c].mean() for c in np.unique(targets)]))},ranking


def verify_run(run):
    run=Path(run); p=json.loads((run/'prepared.json').read_text()); cfg=p['config']; roles=p['roles']; o=p['ontology']
    report=json.loads((run/'metrics.json').read_text()); frozen=json.loads((run/'selection_frozen_before_test.json').read_text())
    checks=0
    def check(value,why):
        nonlocal checks
        checks+=1
        if not value: raise AssertionError(why)
    def close(a,b,why): check(abs(float(a)-float(b))<1e-10,why)
    for path,h in {**p['source_hashes'],**p['csv_hashes'],**p['manifest_hashes'],**p['historical_hashes']}.items():
        check(sha256(path)==h,'Changed source/reference '+path)
    check(sha256(cfg['pretrained_weights'])==p['pretrained_sha256'],'Original pretrained weights')
    for path,h in p['image_hashes'].items(): check(sha256(path)==h,'Changed image '+path)
    check(len(roles['fit'])==6746 and len(roles['seen_validation'])==1254,'Original split sizes')
    check(set(roles)=={'fit','seen_validation','official_seen','official_unseen'},'No extra holdout role')
    for role,name in (('fit','fit.csv'),('seen_validation','validation.csv')):
        with (Path(cfg['baseline_run'])/name).open(encoding='utf-8-sig',newline='') as f: original=list(csv.DictReader(f))
        check([r['filename'] for r in original]==[r['filename'] for r in roles[role]],'Exact original manifest order/membership')
    fit_hashes={p['image_hashes'][r['path']] for r in roles['fit']}
    check(not fit_hashes&{p['image_hashes'][r['path']] for r in roles['seen_validation']},'No duplicate-content leakage')
    fit_comp={r['composition_id'] for r in roles['fit']}; unseen_comp={r['composition_id'] for r in roles['official_unseen']}
    check(len(fit_comp)==323 and len(unseen_comp)==10 and not fit_comp&unseen_comp,'Compositional split')
    for key in ('crop','disease'):
        check(set(o[key])=={r[key+'_id'] for r in roles['fit']},'All primitive factors learned from fitting')
    check(frozen['fingerprint']==p['fingerprint'] and not frozen['official_scores_used'],'Frozen pre-test selection')
    check(sha256(run/'conditioned/best.pt')==frozen['checkpoint_sha256'],'Selected model unchanged')
    best=torch.load(run/'conditioned/best.pt',map_location='cpu',weights_only=True)
    last=torch.load(run/'conditioned/last.pt',map_location='cpu',weights_only=True)
    history=json.loads((run/'conditioned/history.json').read_text())
    check(last['complete'],'Full training completed')
    check(last['fingerprint']==best['fingerprint']==p['fingerprint'],'Checkpoint provenance')
    check(last['history']==history and last['epoch']==len(history)<=cfg['epochs'],'Complete history')
    records=[last['initial_validation']]+[h['validation'] for h in history]
    chosen=max(range(len(records)),key=lambda i:(records[i]['selection_score'],-records[i]['mean_ce']))
    check(chosen==best['epoch']==last['best_epoch']==frozen['selected_epoch'],'Seen-validation marginal-disease rule including epoch zero')
    check(best['validation']==records[chosen],'Selected validation record')
    if len(history)<cfg['epochs']:
        check(last['stale']>=cfg['patience'],'Valid early stopping')
    for v in best['model'].values(): check(bool(torch.isfinite(v).all()),'Finite selected tensors')
    check(sha256(p['crop_checkpoint'])==p['crop_checkpoint_sha256'],'Frozen saved crop checkpoint unchanged')
    check(sha256(p['disease_checkpoint'])==p['disease_checkpoint_sha256'],'Disease initialization checkpoint unchanged')
    seed_all(cfg['seed']+1)
    initial=ConditionalDiseaseCNN(cfg,o,warm_start=True)
    check(tensor_hash(initial.state_dict())==best['initial_hash'],'Warm-start initialization independently reproduced')
    original=torch.load(p['disease_checkpoint'],map_location='cpu',weights_only=True)
    check(tensor_hash(initial.net.state_dict())==tensor_hash(original['model_state']),'Exact saved baseline disease warm-start')
    init=initial.state_dict()
    changed=sum(int(torch.count_nonzero(v!=best['model']['net.'+k])) for k,v in initial.net.state_dict().items() if not k.startswith('head.fc.'))
    changed_last=sum(int(torch.count_nonzero(v!=last['model']['net.'+k])) for k,v in initial.net.state_dict().items() if not k.startswith('head.fc.'))
    check(changed_last>0,'Full disease backbone trained')
    check(tensor_hash(initial.conditioner.state_dict())!=tensor_hash({k.removeprefix('conditioner.'):v for k,v in last['model'].items() if k.startswith('conditioner.')}),'Conditioner trained')
    if best['epoch']>0: check(changed>0,'Selected disease backbone differs from initialization')
    else: check(tensor_hash(init)==tensor_hash(best['model']),'Selected epoch zero is exactly initialization')
    check(not any(k.startswith('crop_model.') for k in best['model']),'Crop CNN outside trainable model')
    del initial,original,init
    for h in history:
        path=run/'conditioned'/f"exposures_epoch{h['epoch']:02}.npz"; bank=np.load(path)
        check(sha256(path)==h['training']['exposure_sha256'],'Exposure hash')
        check(np.array_equal(np.sort(bank['indices']),np.arange(len(roles['fit']))),'Every fitting image once per epoch')
        check(np.array_equal(bank['image_counts'],np.ones(len(roles['fit']),dtype=int)),'No replacement sampling')
        check(h['training']['samples']==h['training']['unique_images']==6746,'Full exposure budget')
    cm={v:i for i,v in enumerate(o['crop'])}; dm={v:i for i,v in enumerate(o['disease'])}
    pm={c['composition_id']:i for i,c in enumerate(o['candidates'])}; pids=np.array(list(pm))
    freeze_time=datetime.fromisoformat(frozen['frozen_utc']).timestamp()
    for role in ('seen_validation','official_seen','official_unseen'):
        rows=roles[role]; path=run/f'logits_{role}.npz'; bank=np.load(path)
        check(bank['checkpoint_hash'].item()==frozen['checkpoint_sha256'],'Raw scores from frozen checkpoint')
        check(bank['filenames'].tolist()==[r['filename'] for r in rows],'Image alignment')
        check(bank['composition_targets'].tolist()==[r['composition_id'] for r in rows],'Ground-truth alignment')
        if role.startswith('official'): check(path.stat().st_mtime>=freeze_time,'Test inference after selection freeze')
        # Reconstruct the conditional probability model with independent NumPy.
        def logsoftmax(x,axis):
            x=x.astype(np.float64); shifted=x-x.max(axis=axis,keepdims=True)
            return shifted-np.log(np.exp(shifted).sum(axis=axis,keepdims=True))
        lc=logsoftmax(bank['crop'],1); ld=logsoftmax(bank['conditional'],2)
        ci=np.array([cm[c['crop_id']] for c in o['candidates']]); di=np.array([dm[c['disease_id']] for c in o['candidates']])
        pair=lc[:,ci]+ld[:,ci,di]
        joint=lc[:,:,None]+ld; mx=joint.max(axis=1)
        marginal=mx+np.log(np.exp(joint-mx[:,None,:]).sum(axis=1))
        check(bank['conditional'].shape==(len(rows),55,175),'All crop conditions exported')
        check(np.allclose(bank['pair'],pair,atol=1e-12,rtol=0),'Independent properly normalized conditional pair scores')
        check(np.allclose(bank['disease'],marginal,atol=1e-12,rtol=0),'Independent disease marginal over every crop')
        cropbank=np.load(run/f'crop_{role}.npz')
        check(cropbank['checkpoint_hash'].item()==p['crop_checkpoint_sha256'],'Frozen crop cache provenance')
        check(np.array_equal(cropbank['logits'],bank['crop']),'Original crop scores actually used')
        check(cropbank['filenames'].tolist()==bank['filenames'].tolist(),'Crop cache image alignment')
        for key,size,y in (('pair',333,np.array([pm[r['composition_id']] for r in rows])),
                           ('crop',55,np.array([cm[r['crop_id']] for r in rows])),
                           ('disease',175,np.array([dm[r['disease_id']] for r in rows]))):
            scores=bank[key]; check(scores.shape==(len(rows),size) and np.isfinite(scores).all(),'Raw score shapes and finite values')
            computed,ranking=numpy_metrics(scores,y)
            label='composition' if key=='pair' else key
            for metric,value in computed.items(): close(value,report[role][label][metric],'Independent '+role+'/'+label+'/'+metric)
            if key=='pair': pair_ranking=ranking; pair_y=y
        with (run/f'predictions_{role}.csv').open(encoding='utf-8',newline='') as f: exported=list(csv.DictReader(f))
        check(len(exported)==len(rows),'Export length')
        for i,r in enumerate(exported):
            check(r['filename']==rows[i]['filename'] and int(r['true_composition_id'])==rows[i]['composition_id'],'Exported row identity')
            check(int(r['predicted_composition_id'])==pids[pair_ranking[i,0]],'Exported learned-pair Top-1')
            check(json.loads(r['top5_ids'])==pids[pair_ranking[i,:5]].tolist(),'Exported learned-pair Top-5')
        for cid,counts in report[role]['per_composition'].items():
            mask=bank['composition_targets']==int(cid)
            check(int(mask.sum())==counts['images'],'Per-composition denominator')
            check(int((pair_ranking[mask,0]==pair_y[mask]).sum())==counts['correct'],'Per-composition correctness')
    close(report['seen_validation']['disease']['top1'],best['validation']['selection_score'],'Selected marginal-disease validation score re-inferred')
    for key in ('top1','top5','macro_top1'):
        s=report['official_seen']['composition'][key]; u=report['official_unseen']['composition'][key]
        close(2*s*u/(s+u) if s+u else 0,report['H'][key],'H recomputation')
    analysis=json.loads((run/'paired_analysis.json').read_text())
    for role,legacy in (('official_seen','seen'),('official_unseen','unseen')):
        with (Path(cfg['baseline_run'])/f'predictions_{legacy}.csv').open(encoding='utf-8',newline='') as f: old={r['filename']:r for r in csv.DictReader(f)}
        correct=sum(int(old[r['filename']]['top1_id'])==r['composition_id'] for r in roles[role])
        check(correct==analysis['point'][role]['baseline_correct'],'Saved baseline pairing')
        check(analysis['point'][role]['new_correct']==round(report[role]['composition']['top1']*len(roles[role])),'Paired experimental count')
        close(100*(report[role]['composition']['top1']-correct/len(roles[role])),analysis['point'][role]['delta_pp'],'Paired point difference')
    check(analysis['point']['official_seen']['baseline_correct']==1421 and analysis['point']['official_unseen']['baseline_correct']==180,'Canonical reference counts')
    value={'passed':True,'checks':checks,'at_utc':stamp(),'verified_by':'independent NumPy crop-conditional probabilities and rankings',
           'backbone_values_changed':changed,'original_artifacts_unchanged':True,'full_image_coverage':True,
           'inference_score':'normalized P(crop|image) times P(disease|image,candidate crop)'}
    write_json(run/'independent_verification.json',value); return value


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path)
    torch.set_num_threads(4)
    print(json.dumps(verify_run(parser.parse_args().run),indent=2))
