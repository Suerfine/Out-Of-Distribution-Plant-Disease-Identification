"""Fit and freeze a ConvNeXt seen / CLIP unseen decision on development."""
from study import (
    HERE, PROJECT, WORKSPACE, COLUMNS, np, pd, torch, json, os,
    load_clip, text_conditions, seed_everything, build_anchor_pool,
    train_generator, fit_classifier, make_validation_split,
)


def decisions(clip, cnn, seen, unseen, blend):
    def normalize(scores):
        return (scores-scores.mean(1,keepdims=True))/scores.std(1,keepdims=True).clip(1e-9)
    sc,uc = clip[:,seen],clip[:,unseen]
    sn,un = cnn[:,seen],cnn[:,unseen]
    seen_pred = seen[(blend*normalize(sn)+(1-blend)*normalize(sc)).argmax(1)]
    unseen_pred = unseen[uc.argmax(1)]
    return seen_pred,unseen_pred,sc.max(1)-uc.max(1),sn.max(1)-un.max(1)


def accuracy(pred, target, nseen):
    hits = pred == target
    s,u = hits[:nseen].mean(),hits[nseen:].mean()
    return {'seen':float(s),'unseen':float(u),'h':float(2*s*u/(s+u) if s+u else 0),
            'seen_correct':int(hits[:nseen].sum()),'unseen_correct':int(hits[nseen:].sum())}


def main():
    torch.set_num_threads(4)
    out = HERE/'hybrid'
    out.mkdir(exist_ok=True)
    cfg = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())
    cfg['model']['pretrained'] = str(HERE/'ViT-B-16-state.pt')
    prepared = json.loads((HERE.parent/'cnn_detail_base_v1/runs/development_seed42/prepared.json').read_text())
    pairs = pd.DataFrame(prepared['pairs']).sort_values('pair_id').reset_index(drop=True)
    fit_data = pd.read_csv(WORKSPACE/'Plant_Disease/DPD_Small_train.csv',header=None,names=COLUMNS)
    fit_data['row'] = np.arange(len(fit_data))
    v = cfg['validation']
    fit,seen,unseen,held = make_validation_split(fit_data,v['n_holdout_pairs'],v['min_holdout_images'],
        v['val_fraction'],42,tuple(v['exclude_holdout_diseases']),v['max_images_per_holdout_pair'])
    assert len(seen)==862 and len(unseen)==533
    assert set(fit.file)=={r['filename'] for r in prepared['roles']['fit']}
    frame = pd.concat([seen,unseen],ignore_index=True)
    index = {r['filename']:i for i,r in enumerate(prepared['roles']['validation_seen']+prepared['roles']['validation_unseen'])}
    order = [index[name] for name in frame.file]
    targets = frame.pair_id.to_numpy()
    assert np.array_equal(targets,np.array([r['composition_id'] for r in prepared['roles']['validation_seen']+prepared['roles']['validation_unseen']])[order])
    ci,di = pairs.crop_id.to_numpy(),pairs.disease_id.to_numpy()
    with np.load(prepared['baseline_validation_logits']) as bank:
        cnn = (bank['crop'][:,ci]+bank['disease'][:,di])[order]
    official_unseen = pairs.loc[pairs.is_unseen,'pair_id'].to_numpy()
    dev_seen = np.setdiff1d(np.arange(333),np.r_[held,official_unseen])
    held = np.sort(held)
    dev_path = out/'development_scores.npz'
    if dev_path.is_file():
        with np.load(dev_path) as saved:
            assert np.array_equal(frame.file.to_numpy(),saved['filenames'])
            clip = saved['clip']
    else:
        seed_everything(42)
        net,_,tokenizer = load_clip(cfg['model'],'cuda')
        prompt,cond = [torch.as_tensor(a,device='cuda') for a in text_conditions(cfg,net,tokenizer,pairs,'cuda')]
        del net
        with np.load(HERE/'runs/seed_42/features.npz') as bank:
            assert np.array_equal(fit_data.file.to_numpy(),bank['filenames'])
            features = torch.as_tensor(bank['embeddings'],device='cuda')
        x,y = features[fit.row.to_numpy()],torch.as_tensor(fit.pair_id.to_numpy(copy=True),device='cuda')
        trained = torch.zeros(333,dtype=torch.bool,device='cuda')
        trained[sorted(set(fit.pair_id))] = True
        pool,_ = build_anchor_pool(x,fit,pairs)
        seed_everything(42)
        opt = cfg['aux_classifier']
        aux = fit_classifier(x,y,trained,prompt,opt,opt['epochs'],tag='aux')
        generate = train_generator(x,y,cond,aux,cfg['gan'],pool)
        sy = torch.as_tensor(held,device='cuda').repeat_interleave(300)
        with torch.no_grad():
            sx = generate(cond[sy],sy)
        seed_everything(42)
        allowed = torch.ones(333,dtype=torch.bool,device='cuda')
        allowed[official_unseen] = False
        head = fit_classifier(torch.cat((x,sx)),torch.cat((y,sy)),allowed,prompt,cfg['classifier'],1)
        with torch.inference_mode():
            clip = head(features[frame.row.to_numpy()]).cpu().numpy()/100
        assert np.isfinite(clip).all()
        np.savez_compressed(dev_path,clip=clip,filenames=frame.file.to_numpy(dtype=str))

    a,b,gap,cgap = decisions(clip,cnn,dev_seen,held,0)
    clip_reference = accuracy(np.where(gap>=.03,a,b),targets,len(seen))
    scale = float(np.std(cgap))
    assert np.isfinite(scale) and scale>0
    candidates = []
    for blend in (.5,.75,1.):
        a,b,gap,cgap = decisions(clip,cnn,dev_seen,held,blend)
        for weight in (0.,.0025,.005,.01,.02):
            route = gap+weight*cgap/scale
            for threshold in np.arange(-.05,.151,.0025):
                scores = accuracy(np.where(route>=threshold,a,b),targets,len(seen))
                if scores['unseen_correct'] >= clip_reference['unseen_correct']:
                    candidates.append((scores['h'],blend,weight,float(threshold),scores))
    _,blend,weight,threshold,dev = max(candidates,key=lambda r:r[0])
    selection = {'split':42,'cnn_seen_blend':blend,'cnn_gate_weight':weight,'cnn_gap_scale':scale,
                 'threshold':threshold,'hybrid_development':dev,'clip_gamma_003_development':clip_reference}
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    print('Frozen on development',selection,flush=True)

    official_seen = np.setdiff1d(np.arange(333),official_unseen)
    cnn_path = PROJECT/'runs/baseline/independent_cnn_baseline_convnext_base_seed42/scores.npz'
    result = {'hybrid':[],'clip_gamma_003':[]}
    targets = []
    with np.load(cnn_path) as saved:
        for split in ('test_seen','test_unseen'):
            with np.load(HERE/'official'/f'anchored_{split}.npz') as z:
                clip,y,names = z['scores'],z['labels'],z['filenames']
            metadata = pd.read_csv(WORKSPACE/'Plant_Disease'/f'DPD_Small_{split}.csv',header=None,names=COLUMNS)
            assert np.array_equal(names,metadata.file.to_numpy())
            assert np.array_equal(y,saved[f'{split}_labels']) and np.array_equal(y,metadata.pair_id.to_numpy())
            cnn = saved[f'{split}_scores']
            a,b,gap,cgap = decisions(clip,cnn,official_seen,official_unseen,blend)
            pred = np.where(gap+weight*cgap/scale>=threshold,a,b)
            clip_a = official_seen[clip[:,official_seen].argmax(1)]
            old = np.where(gap>=.03,clip_a,b)
            result['hybrid'].append(pred)
            result['clip_gamma_003'].append(old)
            targets.append(y)
            np.savez_compressed(out/f'{split}.npz',filenames=names,labels=y,hybrid=pred,clip_gamma_003=old)
    metrics = {arm:accuracy(np.concatenate(pred),np.concatenate(targets),2000) for arm,pred in result.items()}
    assert metrics['clip_gamma_003']['seen_correct']==1212 and metrics['clip_gamma_003']['unseen_correct']==1105
    (out/'results.json').write_text(json.dumps({'selection':selection,'official':metrics},indent=2))
    print(metrics,flush=True)


if __name__ == '__main__':
    main()
