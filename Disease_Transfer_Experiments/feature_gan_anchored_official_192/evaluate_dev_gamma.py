"""Select gamma on development vectors and evaluate the frozen settings."""
from evaluate_calibrated import HERE, DETAIL, IndependentNet, transforms, torch, np, json, write_json, Images
from pathlib import Path

SEEDS = (42,43,44)
ARMS = ('baseline','gan','real_only')


def vectors(model, rows, transform):
    loader = torch.utils.data.DataLoader(Images(rows,transform),batch_size=16,num_workers=0)
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
        return torch.cat([model.backbone(x.cuda()).float() for x,_ in loader])


def score_head(model, path, x):
    model.head.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
    with torch.inference_mode():
        return model.head(x).cpu().numpy()


def margins(crop, disease, ci, di, seen, unseen):
    scores = crop[:,ci]+disease[:,di]
    a,b = scores[:,seen],scores[:,unseen]
    return seen[a.argmax(1)],unseen[b.argmax(1)],a.max(1)-b.max(1)


def accuracy(values, labels, nseen, gamma):
    a,b,gap = values
    hit = np.where(gap>=gamma,a,b)==labels
    s,u = hit[:nseen].mean(),hit[nseen:].mean()
    return [float(s),float(u),float(2*s*u/(s+u) if s+u else 0)]


def main():
    torch.set_num_threads(4)
    out = HERE/'calibrated'
    out.mkdir(exist_ok=True)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    pairs = sorted(p['pairs'],key=lambda r:r['pair_id'])
    ci,di = [[r[k] for r in pairs] for k in ('crop_id','disease_id')]
    roles = p['roles']
    rows = roles['validation_seen']+roles['validation_unseen']
    targets = np.array([r['composition_id'] for r in rows])
    pseudo = np.unique(targets[862:])
    official_unseen = np.array([r['pair_id'] for r in pairs if r['is_unseen']])
    dev_seen = np.setdiff1d(np.arange(333),np.r_[pseudo,official_unseen])
    cfg = p['baseline_config']
    model = IndependentNet(cfg,'disease').cuda().eval()
    source = next(path for path in p['reference_hashes'] if Path(path).parent.name=='disease' and Path(path).name=='best_epoch_15.pt')
    model.load_state_dict(torch.load(source,weights_only=True,map_location='cpu')['model_state'])
    _, transform = transforms(cfg)
    x = vectors(model,rows,transform)
    dev = {arm:[] for arm in ARMS}
    with np.load(p['baseline_validation_logits']) as bank:
        crop = bank['crop']
        dev['baseline'].append(margins(crop,bank['disease'],ci,di,dev_seen,pseudo))
        for seed in SEEDS:
            for arm in ('gan','real_only'):
                path = HERE.parent/'feature_gan_anchored_v1/runs'/f'seed_{seed}'/f'192_{arm}_head.pt'
                dev[arm].append(margins(crop,score_head(model,path,x),ci,di,dev_seen,pseudo))
    selection = {}
    for arm, values in dev.items():
        grid = [(round(float(g),2),np.mean([accuracy(v,targets,862,g) for v in values],axis=0).tolist())
                for g in np.arange(0,12.001,.05)]
        gamma,triple = max(grid,key=lambda item:item[1][2])
        selection[arm] = {'gamma':gamma,'dev_seen_unseen_h':triple}
    write_json(out/'selection_dev.json',selection)
    print(selection,flush=True)
    baseline = HERE.parents[1]/'Out-Of-Distribution-Plant-Disease-Identification/runs/baseline/independent_cnn_baseline_convnext_base_seed42'
    model.load_state_dict(torch.load(baseline/'disease_net.pt',weights_only=True,map_location='cpu'))
    cache = baseline.parents[1]/'cache/resized_256'
    official_seen = np.setdiff1d(np.arange(333),official_unseen)
    official = {arm:[] for arm in ARMS}
    target_splits = []
    with np.load(baseline/'factor_logits.npz') as factors:
        for split in ('test_seen','test_unseen'):
            with np.load(HERE/'runs/seed_42'/f'{split}.npz') as bank:
                names,labels = bank['filenames'],bank['labels']
            rows = [{'path':str(cache/(Path(n).stem+'.jpg')),'disease_id':di[k]} for n,k in zip(names,labels)]
            x = vectors(model,rows,transform)
            crop = factors[f'{split}_crop']
            target_splits.append(labels)
            official['baseline'].append([margins(crop,factors[f'{split}_disease'],ci,di,official_seen,official_unseen)])
            for arm in ('gan','real_only'):
                official[arm].append([margins(crop,score_head(model,HERE/'runs'/f'seed_{seed}'/f'{arm}_head.pt',x),
                                              ci,di,official_seen,official_unseen) for seed in SEEDS])
    results = {}
    targets = np.concatenate(target_splits)
    for arm,(seen,unseen) in official.items():
        gamma = selection[arm]['gamma']
        results[arm] = [accuracy(tuple(np.concatenate((a,b)) for a,b in zip(s,u)),targets,2000,gamma)
                        for s,u in zip(seen,unseen)]
    write_json(out/'development_tuned_results.json',{'selection':selection,'official_seen_unseen_h':results})
    print(results,flush=True)


if __name__=='__main__':
    main()

