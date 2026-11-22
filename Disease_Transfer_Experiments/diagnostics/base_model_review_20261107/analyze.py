"""Read-only model/error audit and preparation of existing development references."""
from collections import Counter
import csv,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PROJECT=ROOT/'Out-Of-Distribution-Plant-Disease-Identification'
BASE=PROJECT/'runs/baseline/independent_cnn_baseline_convnext_base_seed42'
STUDY=PROJECT/'runs/baseline/independent_cnn_baseline_study_seed42'
DEV=PROJECT/'runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42/trials/trial_000'
SSL=ROOT/'Disease_Transfer_Experiments/simsiam_base_v1/runs/overnight_seed42'
CONDITIONED=ROOT/'Disease_Transfer_Experiments/crop_conditioned_base_v1/runs/full_seed42'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def read_csv(path):
    with open(path,newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def counts(values,names,limit=6):
    return [{'label':names[int(k)],'images':int(v)} for k,v in Counter(values.tolist()).most_common(limit)]


def main():
    p=json.loads((SSL/'prepared.json').read_text()); ontology=p['ontology']; candidates=ontology['candidates']
    ids=np.array([r['composition_id'] for r in candidates]); assert np.array_equal(ids,np.arange(333))
    names={r['composition_id']:r['composition_name'] for r in candidates}
    allrows=sum(p['roles'].values(),[])
    disease_names={r['disease_id']:r['disease_name'] for r in allrows}; crop_names={r['crop_id']:r['crop_name'] for r in allrows}
    train=p['roles']['fit']; seen_ids={r['composition_id'] for r in train}; seen_mask=np.array([v in seen_ids for v in ids])
    ci=np.array([r['crop_id'] for r in candidates]); di=np.array([r['disease_id'] for r in candidates])
    raw=np.load(BASE/'scores.npz'); factors=np.load(BASE/'factor_logits.npz')
    report={'utc':datetime.now(timezone.utc).isoformat(),'status':'posthoc diagnostic, not model selection','models':['baseline','ssl','conditioned'],
        'new_training':False,'gamma_search':False,'roles':{},'checks':{},'input_hashes':{}}
    paired_rows=[]; examples=[]
    for role,split in [('official_seen','test_seen'),('official_unseen','test_unseen')]:
        rows=p['roles'][role]; y=np.array([r['composition_id'] for r in rows]); dy=np.array([r['disease_id'] for r in rows]); cy=np.array([r['crop_id'] for r in rows])
        saved=read_csv(BASE/f'predictions_{split}.csv')
        assert [r['file'] for r in saved]==[r['filename'] for r in rows]
        assert np.array_equal(raw[split+'_labels'],y)
        ss=np.load(SSL/f'logits_{role}.npz'); cc=np.load(CONDITIONED/f'logits_{role}.npz')
        for bank in (ss,cc):
            assert bank['filenames'].tolist()==[r['filename'] for r in rows]
            assert np.array_equal(bank['composition_targets'],y)
            assert np.array_equal(bank['crop'],factors[split+'_crop'])
        scores={'baseline':raw[split+'_scores'].astype(np.float64),'ssl':ss['pair'].astype(np.float64),'conditioned':cc['pair'].astype(np.float64)}
        disease={'baseline':factors[split+'_disease'],'ssl':ss['disease'],'conditioned':cc['disease']}
        disease_rank={k:np.argsort(-v,axis=1,kind='stable') for k,v in disease.items()}
        true_rank={k:np.argmax(rank==dy[:,None],axis=1)+1 for k,rank in disease_rank.items()}
        preds={g:{k:(s-g*seen_mask[None,:]).argmax(1) for k,s in scores.items()} for g in (0.,6.2)}
        value={'images':len(rows),'factor_metrics':{},'pair_metrics':{},'per_composition':[]}
        for model,rank in disease_rank.items():
            value['factor_metrics'][model]={'disease_top1_pct':float(100*(rank[:,0]==dy).mean()),'disease_top5_pct':float(100*(rank[:,:5]==dy[:,None]).any(1).mean()),'median_true_disease_rank':float(np.median(true_rank[model]))}
        for gamma,by_model in preds.items():
            value['pair_metrics'][str(gamma)]={}
            bc=by_model['baseline']==y
            for model,pred in by_model.items():
                correct=pred==y
                value['pair_metrics'][str(gamma)][model]={'correct':int(correct.sum()),'top1_pct':float(100*correct.mean()),
                    'fixed_vs_baseline':int((~bc&correct).sum()),'broken_vs_baseline':int((bc&~correct).sum())}
            value['pair_metrics'][str(gamma)]['both_new_models_break_baseline_correct']=int((bc&(by_model['ssl']!=y)&(by_model['conditioned']!=y)).sum())
        crop_pred=factors[split+'_crop'].argmax(1)
        value['crop_branch_top1_pct']=float(100*(crop_pred==cy).mean())
        for cid in np.unique(y):
            take=y==cid; entry=candidates[int(cid)]; crop=entry['crop_id']; disease_id=entry['disease_id']
            same_disease=[r for r in train if r['disease_id']==disease_id]
            support=Counter(r['crop_id'] for r in same_disease)
            row={'composition':names[int(cid)],'composition_id':int(cid),'images':int(take.sum()),
                'full_train_images_same_disease':len(same_disease),'full_train_crops_same_disease':len(support),
                'same_disease_training_support':[{'crop':crop_names[c],'images':n} for c,n in support.most_common()],
                'full_train_images_same_crop':sum(r['crop_id']==crop for r in train),'metrics':{}}
            for model in scores:
                row['metrics'][model]={'raw_disease_correct':int((disease_rank[model][take,0]==disease_id).sum()),
                    'raw_disease_top5_correct':int((disease_rank[model][take,:5]==disease_id).any(1).sum()),
                    'median_true_disease_rank':float(np.median(true_rank[model][take]))}
            for gamma,by_model in preds.items():
                row[str(gamma)]={}
                bc=by_model['baseline']==y
                for model,pred in by_model.items():
                    correct=pred==y; broken=take&bc&~correct; fixed=take&~bc&correct
                    row[str(gamma)][model]={'correct':int(correct[take].sum()),'broken':int(broken.sum()),'fixed':int(fixed.sum()),
                        'wrong_predictions':counts(pred[take&~correct],names),'regression_destinations':counts(pred[broken],names)}
            value['per_composition'].append(row)
        # Retain every changed decision, not just illustrative examples.
        for gamma,by_model in preds.items():
            for model in ('ssl','conditioned'):
                before=by_model['baseline']; after=by_model[model]
                for i in np.flatnonzero(before!=after):
                    paired_rows.append({'role':role,'gamma':gamma,'model':model,'filename':rows[i]['filename'],'truth':names[int(y[i])],
                        'baseline_prediction':names[int(before[i])],'new_prediction':names[int(after[i])],
                        'baseline_correct':bool(before[i]==y[i]),'new_correct':bool(after[i]==y[i]),
                        'crop_branch_correct':bool(crop_pred[i]==cy[i]),'baseline_true_disease_rank':int(true_rank['baseline'][i]),'new_true_disease_rank':int(true_rank[model][i])})
        if role=='official_unseen':
            for name,gamma in [('pepper_bell_bacterial_spot',0.),('grape_black_rot',6.2)]:
                cid=next(c for c,n in names.items() if n==name); bp=preds[gamma]['baseline']; sp=preds[gamma]['ssl']
                pick=[i for i in np.flatnonzero((y==cid)&(bp==y)&(sp!=y))]
                pick=sorted(pick,key=lambda i:rows[i]['filename'])[:3]
                for i in pick:
                    examples.append({'composition':name,'gamma':gamma,'filename':rows[i]['filename'],'original_path':rows[i]['original_path'],'cache_path':rows[i]['path'],
                        'baseline_prediction':names[int(bp[i])],'ssl_prediction':names[int(sp[i])],
                        'baseline_true_disease_rank':int(true_rank['baseline'][i]),'ssl_true_disease_rank':int(true_rank['ssl'][i])})
        report['roles'][role]=value
    # Exact hashes after the real image resize/cache stage complement the
    # previous raw-file duplicate checks. This does not detect all near duplicates.
    groups={}
    for role,rows in p['roles'].items():
        by_hash={}
        for r in rows: by_hash.setdefault(p['cache_hashes'][r['path']],[]).append(r['filename'])
        groups[role]=by_hash
    for role in ('official_seen','official_unseen'):
        shared=groups['fit'].keys()&groups[role].keys()
        report['checks'][f'cached_input_duplicates_fit_{role}']=[{'sha256':h,'train':groups['fit'][h],'test':groups[role][h]} for h in sorted(shared)]
    report['checks']['all_role_and_crop_logit_alignments_passed']=True
    # Stage exact existing Base development references for review; create no split.
    refs=HERE/'reference_manifests'; refs.mkdir(exist_ok=True)
    frames={}
    for name in ('fit','validation_seen','validation_unseen','selection_excluded','all_train'):
        path=STUDY/f'{name}.csv'; shutil.copyfile(path,refs/path.name); frames[name]=read_csv(path); report['input_hashes'][str(path)]=sha(path)
    sets={k:{r['file'] for r in rows} for k,rows in frames.items()}
    for i,a in enumerate(('fit','validation_seen','validation_unseen','selection_excluded')):
        for b in ('fit','validation_seen','validation_unseen','selection_excluded')[i+1:]: assert not sets[a]&sets[b]
    assert set.union(*(sets[k] for k in ('fit','validation_seen','validation_unseen','selection_excluded')))==sets['all_train']
    fit_pairs={int(r['pair_id']) for r in frames['fit']}; pseudo_pairs={int(r['pair_id']) for r in frames['validation_unseen']}
    assert len(pseudo_pairs)==20 and not fit_pairs&pseudo_pairs
    assert {int(r['pair_id']) for r in frames['selection_excluded']}<=pseudo_pairs
    assert {int(r['crop_id']) for r in frames['fit']}==set(range(55)) and {int(r['disease_id']) for r in frames['fit']}==set(range(175))
    assert sets['all_train']=={r['filename'] for r in train}
    selected=json.loads((DEV.parent.parent/'selected_recipe.json').read_text())
    for name in ('crop','disease'):
        path=DEV/name/'best_epoch_15.pt'; assert path.exists(); report['input_hashes'][str(path)]=sha(path)
    dev_bank=DEV/'validation_epoch_15.npz'; assert dev_bank.exists()
    for path in (BASE/'scores.npz',BASE/'factor_logits.npz',SSL/'logits_official_seen.npz',SSL/'logits_official_unseen.npz',CONDITIONED/'logits_official_seen.npz',CONDITIONED/'logits_official_unseen.npz',dev_bank): report['input_hashes'][str(path)]=sha(path)
    proposal={'status':'PROPOSED ONLY; no data roles or training changed','requires_agreement_to_revisit_prior_no_pseudo_validation_preference':True,
        'existing_base_partition_counts':{k:len(v) for k,v in frames.items()},'pseudo_unseen_pair_count':20,
        'development_crop_checkpoint':str(DEV/'crop/best_epoch_15.pt'),'development_baseline_logits':str(dev_bank),
        'primary_objective':'validation composition Top-1 H, gamma0; earliest exact tie',
        'saved_development_baseline_metrics':selected['validation'],'baseline_retraining_required':False,
        'ssl_development_images':'fit5798 only; exclude all validation and capped held-out images',
        'final_refit_images':8000,'official_seen_images':2000,'official_unseen_images':1448,
        'official_candidates':333,'official_test_policy':'Previously inspected; descriptive final reporting only, never tuning',
        'source_manifest_hashes':{str(refs/name):sha(refs/name) for name in ('fit.csv','validation_seen.csv','validation_unseen.csv','selection_excluded.csv','all_train.csv')}}
    write(HERE/'development_protocol_proposal.json',proposal)
    write(HERE/'audit.json',report); write(HERE/'paired_decision_changes.json',paired_rows); write(HERE/'illustrative_regressions.json',examples)
    slim=report['roles']['official_unseen']
    print(json.dumps({'unseen_summary':{k:v for k,v in slim.items() if k!='per_composition'},
        'targeted_compositions':[c for c in slim['per_composition'] if c['composition'] in ('grape_black_rot','pepper_bell_bacterial_spot')],
        'cache_duplicate_checks':report['checks'],'reference_counts':proposal['existing_base_partition_counts']},indent=2))


if __name__=='__main__': main()
