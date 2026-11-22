"""Independent NumPy ranking, provenance, split and training verification."""
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from model import CompositionalCNN
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
    check(sha256(run/'compositional/best.pt')==frozen['checkpoint_sha256'],'Selected model unchanged')
    best=torch.load(run/'compositional/best.pt',map_location='cpu',weights_only=True)
    last=torch.load(run/'compositional/last.pt',map_location='cpu',weights_only=True)
    history=json.loads((run/'compositional/history.json').read_text())
    check(last['complete'],'Full training completed')
    check(last['fingerprint']==best['fingerprint']==p['fingerprint'],'Checkpoint provenance')
    check(last['history']==history and last['epoch']==len(history)<=cfg['epochs'],'Complete history')
    chosen=max(range(len(history)),key=lambda i:(history[i]['validation']['selection_score'],-history[i]['validation']['mean_ce']))
    check(chosen+1==best['epoch']==last['best_epoch']==frozen['selected_epoch'],'Seen-validation primary-pair checkpoint rule')
    check(best['validation']==history[chosen]['validation'],'Selected validation record')
    if len(history)<cfg['epochs']:
        check(last['stale']>=cfg['patience'],'Valid early stopping')
    for v in best['model'].values(): check(bool(torch.isfinite(v).all()),'Finite selected tensors')
    seen_indices=[i for i,c in enumerate(o['candidates']) if c['composition_id'] in fit_comp]
    check(best['model']['seen_indices'].tolist()==seen_indices,'Only seen classes in pair-training denominator')
    check(int((best['model']['global_to_seen']<0).sum())==10,'Unseen training classes masked')
    seed_all(cfg['seed']+1)
    initial=CompositionalCNN(cfg,o,fit_comp,pretrained=True)
    check(tensor_hash(initial.state_dict())==best['initial_hash'],'Original initialization independently reproduced')
    initial_composer=tensor_hash(initial.composer.state_dict())
    trained_composer={k.removeprefix('composer.'):v for k,v in best['model'].items() if k.startswith('composer.')}
    check(tensor_hash(trained_composer)!=initial_composer,'Shared composer trained')
    check(not any(tuple(v.shape)==(333,cfg['embedding_dim']) for _,v in initial.named_parameters()),'No independent pair-classifier table')
    changed=sum(int(torch.count_nonzero(v!=best['model']['net.'+k])) for k,v in initial.net.state_dict().items())
    check(changed>0,'Full CNN fine-tuned')
    del initial
    for h in history:
        path=run/'compositional'/f"exposures_epoch{h['epoch']:02}.npz"; bank=np.load(path)
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
    close(report['seen_validation']['composition']['top1'],best['validation']['selection_score'],'Selected primary-pair validation score re-inferred')
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
    value={'passed':True,'checks':checks,'at_utc':stamp(),'verified_by':'independent NumPy learned-pair rankings',
           'backbone_values_changed':changed,'original_artifacts_unchanged':True,'full_image_coverage':True,
           'inference_score':'learned pair scores; not factor probability product'}
    write_json(run/'independent_verification.json',value); return value


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path)
    torch.set_num_threads(4)
    print(json.dumps(verify_run(parser.parse_args().run),indent=2))
