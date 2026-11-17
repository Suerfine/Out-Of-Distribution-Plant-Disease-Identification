"""Compare cross-entropy and hinge losses on frozen ConvNeXt features."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from within_crop_disease import BASE, DETAIL, HERE, IndependentNet, transforms, write_json


def main(lesion_pilot=False, *, feature_experiment=None, experiment_name='seen_svm'):
    from dpd.feature_gen import CosineClassifier
    sys.path.insert(0, str(HERE.parent / 'leafgan_v1'))
    from classifier_multi import Images

    torch.set_num_threads(4)
    out = HERE / (experiment_name if feature_experiment else 'seen_lesion_crop' if lesion_pilot else 'seen_hinge_head')
    out.mkdir(exist_ok=True)
    p = json.loads((DETAIL / 'runs/development_seed42/prepared.json').read_text())
    fit, seen, unseen = (p['roles'][k] for k in ('fit', 'validation_seen', 'validation_unseen'))
    pairs = sorted(p['pairs'], key=lambda r: r['pair_id'])
    ci, di = [np.array([r[key] for r in pairs]) for key in ('crop_id', 'disease_id')]
    si, ui = [np.unique([r['composition_id'] for r in rows]) for rows in (fit, unseen)]
    y_pair = np.array([r['composition_id'] for r in seen])
    assert set(si).isdisjoint(ui) and {r['filename'] for r in fit}.isdisjoint(r['filename'] for r in seen)
    with np.load(p['baseline_validation_logits']) as ref:
        crop, disease = ref['crop'][:len(seen)], ref['disease'][:len(seen)]
    prior = HERE / 'clip_disease_only'
    gamma = json.loads((prior / 'development_results.json').read_text())['disease_plus_gan']['gamma']
    scale = json.loads((BASE / 'config/vlm/v11.3_lora_features_config.json').read_text())['classifier']['logit_scale']
    weight = torch.load(prior / 'disease_plus_gan_head.pt', map_location='cpu', weights_only=True)['weight']
    with np.load(HERE / 'runs/seed_42/features.npz') as bank:
        index = {name:i for i,name in enumerate(bank['filenames'])}
        embeddings = torch.from_numpy(bank['embeddings'][[index[r['filename']] for r in seen]])
    with torch.inference_mode():
        clip_disease = CosineClassifier(weight, scale)(embeddings).numpy()
    gap = (crop[:,ci[si]]+clip_disease[:,di[si]]).max(1) - (crop[:,ci[ui]]+clip_disease[:,di[ui]]).max(1)
    route = gap >= gamma
    y_pair, crop, disease = y_pair[route], crop[route], disease[route]
    eval_rows = [r for r, keep in zip(seen, route) if keep]
    source = next(Path(k) for k in p['reference_hashes']
                  if Path(k).parent.name == 'disease' and Path(k).name == 'best_epoch_15.pt')
    cfg = p['baseline_config']
    net = IndependentNet(cfg, 'disease')
    net.load_state_dict(torch.load(source, map_location='cpu', weights_only=True)['model_state'])
    net.cuda().eval()
    _, tf = transforms(cfg)

    def encode(rows):
        loader = torch.utils.data.DataLoader(Images(rows, tf), batch_size=16)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.cat([net.backbone(batch.cuda()).float().cpu() for batch,_ in loader])
        return x.cuda()

    print(f'Extracting frozen features for {len(fit)} training and {len(eval_rows)} routed validation images', flush=True)
    x = encode(fit)
    v = encode(eval_rows)
    y = torch.tensor([r['disease_id'] for r in fit], device='cuda')
    if lesion_pilot:
        from seen_lesion_crop import reviewed_views
        views, extra_y, donor_pairs = reviewed_views(fit, x, net, tf)
    net.backbone.cpu()
    crop_scores = crop[:,ci[si]]

    def rank(logits, disease_weight):
        scores = crop_scores + disease_weight * logits[:,di[si]]
        return si[np.argsort(-scores, axis=1, kind='stable')[:,:3]]

    def report(logits, top):
        truth = di[y_pair]
        row = np.arange(len(truth))
        alternatives = logits.copy()
        alternatives[row, truth] = -np.inf
        margin = logits[row, truth] - alternatives.max(1)
        return dict(top1=int((top[:,0]==y_pair).sum()), top3=int((top==y_pair[:,None]).any(1).sum()),
                    disease_top1=int((logits.argmax(1)==truth).sum()),
                    median_true_disease_margin=float(np.median(margin)))

    baseline = report(disease, rank(disease, 1.))
    assert len(y_pair)==652 and baseline['top1']==593
    if feature_experiment:
        result = feature_experiment(net.head, x, y, v, y_pair, rank, report)
        write_json(out / 'results.json', dict(gamma=gamma, images=len(y_pair), baseline=baseline, **result))
        print(json.dumps(result), flush=True)
        return
    if lesion_pilot:
        from seen_lesion_crop import compare
        result = compare(net.head, x, y, v, views, extra_y, donor_pairs, y_pair, rank, report)
        write_json(out / 'results.json', dict(gamma=gamma, images=len(y_pair), baseline=baseline, **result))
        print(json.dumps(result), flush=True)
        return
    rng = np.random.default_rng(42)
    fold = {k:i%5 for i,k in enumerate(rng.permutation(np.unique(y_pair)))}
    fold_id = np.array([fold[k] for k in y_pair])
    scales = (.25,.5,.75,1.,1.5,2.,3.,4.)
    results = {}
    for name in ('cross_entropy', 'hinge'):
        random = torch.Generator(device='cuda').manual_seed(42)
        head = deepcopy(net.head)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
        for _ in range(3):
            head.train()
            for idx in torch.randperm(len(y), device='cuda', generator=random).split(64):
                logits = head(x[idx])
                targets = y[idx,None]
                if name == 'hinge':
                    wrong = logits.scatter(1, targets, -torch.inf).max(1).values
                    loss = F.relu(1 + wrong - logits.gather(1, targets).squeeze(1)).mean()
                else:
                    loss = F.cross_entropy(logits, targets.squeeze(1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        head.eval()
        with torch.inference_mode():
            logits = head(v).cpu().numpy()
        assert np.isfinite(logits).all()
        top = {w:rank(logits,w) for w in scales}
        selected = np.empty_like(top[1.])
        selected_weights = []
        for k in range(5):
            test = fold_id==k
            cal = ~test
            choice = max(scales, key=lambda w: ((top[w][cal,0]==y_pair[cal]).sum(), -abs(np.log(w))))
            selected[test] = top[choice][test]
            selected_weights.append(choice)
        results[name] = dict(fixed_weight=report(logits, top[1.]), tuned_weight=report(logits, selected),
                             fold_weights=selected_weights)
        torch.save(head.state_dict(), out / f'{name}_head.pt')
        print(name, json.dumps(results[name]), flush=True)
    result = dict(gamma=gamma, images=len(y_pair), epochs=3, baseline=baseline, **results)
    write_json(out / 'results.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
