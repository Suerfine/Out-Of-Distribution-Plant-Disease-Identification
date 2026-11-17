"""Official comparison with the classifier epoch fixed on development."""
from study import (
    HERE, PROJECT, WORKSPACE, COLUMNS, np, pd, torch, json, hashlib,
    load_clip, text_conditions, train_lora, ensure_resized, clip_transforms,
    seed_everything, build_anchor_pool, train_generator, fit_classifier,
)


def main():
    torch.set_num_threads(4)
    out = HERE/'official'
    out.mkdir(exist_ok=True)
    arms = ('original','anchored')
    dev = [json.loads((HERE/'runs'/f'seed_{s}'/'results.json').read_text()) for s in (42,43,44)]
    for run in dev:
        assert all(len(run[arm]) == 25 for arm in arms)
    # Use one common classifier epoch to match the two training budgets.
    averages = np.mean([[r[arm][i]['h'] for i in range(25)]
                        for r in dev for arm in arms],axis=0)
    epochs = int(averages.argmax())+1
    base = json.loads((PROJECT/'config/vlm/v11.3_lora_features_config.json').read_text())
    lora = json.loads((PROJECT/'config/vlm/v10.3_text_anchor_three_splits_config.json').read_text())
    weights = HERE/'ViT-B-16-state.pt'
    for cfg in (base,lora):
        cfg['model']['pretrained'] = str(weights)
    prepared = WORKSPACE/'Disease_Transfer_Experiments/cnn_detail_base_v1/runs/development_seed42/prepared.json'
    pairs = pd.DataFrame(json.loads(prepared.read_text())['pairs']).sort_values('pair_id').reset_index(drop=True)
    assert pairs.pair_id.tolist()==list(range(333))
    held = pairs.loc[pairs.is_unseen,'pair_id'].tolist()
    budget = base['synthesis']['n_per_unseen']
    (out/'selection.json').write_text(json.dumps(dict(classifier_epochs=epochs,gamma=0,
        lora_epochs=8,vectors_per_pair=budget,target_pairs=pairs.loc[held,'pair'].tolist()),indent=2))
    print(f'Locked classifier epoch {epochs} and {budget} vectors for each of {len(held)} target pairs',flush=True)
    assert len(held)==10 and budget==300
    data = WORKSPACE/'Plant_Disease'
    fit = pd.read_csv(data/'DPD_Small_train.csv',header=None,names=COLUMNS)
    fit['source'] = fit.file.str.split('_').str[0]
    assert len(fit)==8000 and fit.file.is_unique and set(held).isdisjoint(fit.pair_id)
    fit_ids = sorted(fit.pair_id.unique())
    cache = ensure_resized(fit.file,lora['input']['resize_cache'])
    tf = clip_transforms(lora['input'],lora['augmentation'])
    seed_everything(base['seed'])
    clip,_,tokenizer = load_clip(base['model'],'cuda')
    prompt,cond = [torch.as_tensor(a,device='cuda') for a in text_conditions(base,clip,tokenizer,pairs,'cuda')]
    del clip
    torch.cuda.empty_cache()
    seed_everything(lora['seed'])
    model,params = train_lora.build(lora,pairs,'cuda')
    print('Refitting the shared CLIP encoder on 8,000 training photos',flush=True)
    train_lora.fit(model,params,fit,fit_ids,8,lora,cache,tf,log=lambda r:print(r,flush=True))
    def encode(frame, directory):
        loader = train_lora.loader(frame,directory,tf[1],lora,shuffle=False)
        return torch.as_tensor(train_lora.encode(model,loader,'cuda',lora['runtime']['amp'],
            tuple(lora['evaluation'].get('tta',['identity'])))[0],device='cuda')
    x = encode(fit,cache)
    y = torch.as_tensor(fit.pair_id.to_numpy(copy=True),device='cuda')
    trained = torch.zeros(len(pairs),dtype=torch.bool,device='cuda')
    trained[fit_ids] = True
    pool,audit = build_anchor_pool(x,fit,pairs)
    fallback = [pairs.loc[k,'pair'] for k in held if not audit[int(pairs.loc[k,'crop_id'])]]
    sy = torch.as_tensor(held,device='cuda').repeat_interleave(budget)
    models = {}
    for arm,anchors in [('original',None),('anchored',pool)]:
        seed_everything(base['seed'])
        aux_cfg = base['aux_classifier']
        aux = fit_classifier(x,y,trained,prompt,aux_cfg,aux_cfg['epochs'],tag='aux')
        g = train_generator(x,y,cond,aux,base['gan'],anchors)
        with torch.no_grad():
            sx = g(cond[sy],sy)
        assert torch.isfinite(sx).all()
        seed_everything(base['seed'])
        head = fit_classifier(torch.cat((x,sx)),torch.cat((y,sy)),torch.ones_like(trained),
                              prompt,base['classifier'],epochs)
        head.eval()
        torch.save(head.state_dict(),out/f'{arm}_head.pt')
        models[arm] = head
        del g,aux,sx
    results = {'gamma':0,'classifier_epochs':epochs,'vectors_per_pair':budget,'fallback_pairs':fallback}
    for split,expected in [('test_seen',2000),('test_unseen',1448)]:
        frame = pd.read_csv(data/f'DPD_Small_{split}.csv',header=None,names=COLUMNS)
        assert len(frame)==expected and set(fit.file).isdisjoint(frame.file)
        assert (frame.pair_id.isin(held) == (split=='test_unseen')).all()
        xt = encode(frame,ensure_resized(frame.file,lora['input']['resize_cache']))
        targets = frame.pair_id.to_numpy()
        results[split] = {}
        for arm,head in models.items():
            with torch.inference_mode():
                scores = head(xt).cpu().numpy()/100
            assert np.isfinite(scores).all()
            hit = scores.argmax(axis=1)==targets
            results[split][arm] = dict(correct=int(hit.sum()),images=expected,top1=float(hit.mean()),
                per_pair={pairs.loc[k,'pair']:dict(correct=int(hit[targets==k].sum()),images=int((targets==k).sum()))
                          for k in np.unique(targets)})
            np.savez_compressed(out/f'{arm}_{split}.npz',scores=scores,labels=targets,filenames=frame.file.to_numpy(dtype=str))
    for arm in arms:
        s,u = (results[t][arm]['top1'] for t in ('test_seen','test_unseen'))
        results[arm] = dict(harmonic_mean_top1=2*s*u/(s+u) if s+u else 0.0)
    temp = out/'results.tmp'
    temp.write_text(json.dumps(results,indent=2))
    temp.replace(out/'results.json')
    print('Official comparison complete',flush=True)


if __name__ == '__main__':
    main()
