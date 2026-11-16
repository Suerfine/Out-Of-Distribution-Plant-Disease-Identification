"""Compare routing labels and expert-correctness labels on held-out development pairs."""
import json

import numpy as np
import torch
from torch.nn import functional as F

from within_crop_disease import BASE, DETAIL, HERE, write_json
from dpd.feature_gen import CosineClassifier


def fit_scores(x, target, fit):
    center = x[fit].mean(axis=0)
    scale = np.maximum(x[fit].std(axis=0), 1e-6)
    features = torch.as_tensor((x - center) / scale, dtype=torch.float64)
    y = torch.as_tensor(target[fit], dtype=torch.float64)
    head = torch.nn.Linear(x.shape[1], 1, dtype=torch.float64)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    opt = torch.optim.LBFGS(head.parameters(), max_iter=60, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(features[fit]).flatten(), y)
        loss = loss + .05 * head.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.inference_mode():
        return head(features).flatten().numpy()


def main():
    torch.set_num_threads(4)
    prepared = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    fit, vs, vu = (prepared['roles'][p] for p in ('fit', 'validation_seen', 'validation_unseen'))
    rows = vs + vu
    y = np.array([r['composition_id'] for r in rows])
    seen = np.arange(len(rows)) < len(vs)
    si = np.unique([r['composition_id'] for r in fit])
    ui = np.unique(y[~seen])
    pairs = sorted(prepared['pairs'], key=lambda r:r['pair_id'])
    ci, di = [np.array([r[k] for r in pairs]) for k in ('crop_id','disease_id')]
    prior = HERE/'clip_disease_only'
    gamma = json.loads((prior/'development_results.json').read_text())['disease_plus_gan']['gamma']
    scale = json.loads((BASE/'config/vlm/v11.3_lora_features_config.json').read_text())['classifier']['logit_scale']
    weight = torch.load(prior/'disease_plus_gan_head.pt', weights_only=True, map_location='cpu')['weight']
    with np.load(HERE/'runs/seed_42/features.npz') as b:
        index = {n:i for i,n in enumerate(b['filenames'])}
        x = torch.from_numpy(b['embeddings'][[index[r['filename']] for r in rows]])
    with torch.inference_mode():
        clip = CosineClassifier(weight,scale)(x).numpy()
    with np.load(prepared['baseline_validation_logits']) as b:
        crop, disease = b['crop'], b['disease']
    clip_seen = crop[:,ci[si]] + clip[:,di[si]]
    clip_unseen = crop[:,ci[ui]] + clip[:,di[ui]]
    cnn_seen = crop[:,ci[si]] + disease[:,di[si]]
    seen_ok = si[cnn_seen.argmax(1)] == y
    unseen_ok = ui[clip_unseen.argmax(1)] == y
    assert not np.any(seen_ok & unseen_ok)
    gap = clip_seen.max(1) - clip_unseen.max(1)

    def margin(scores):
        two = np.partition(scores, -2, axis=1)[:, -2:]
        return two[:,1]-two[:,0]

    features = np.column_stack((gap, margin(cnn_seen), margin(clip_unseen)))
    assert np.isfinite(features).all()

    def correct(scores, mask, cutoff):
        route = scores >= cutoff
        return int((mask & route & seen_ok).sum()), int((mask & ~route & unseen_ok).sum())

    def threshold(scores, mask, target):
        if not target:
            return float('-inf')
        scores = np.sort(scores[mask & unseen_ok].astype(np.float64))
        assert target <= len(scores)
        return float(np.nextafter(scores[target-1], np.inf))

    assert correct(gap, np.ones(len(y), dtype=bool), gamma) == (593,315)
    rng = np.random.default_rng(42)
    groups = {pair:i%5 for group in (y[seen], y[~seen])
              for i,pair in enumerate(rng.permutation(np.unique(group)))}
    folds = np.array([groups[pair] for pair in y])
    records = []
    for fold in range(5):
        test, cal = folds==fold, folds==(fold+1)%5
        fit_rows = ~(test | cal)
        assert set(y[test]).isdisjoint(y[~test])
        candidates = {
            'fixed_current': gap.astype(np.float64),
            'retuned_gap': gap.astype(np.float64),
            'group_label': fit_scores(features, seen, fit_rows),
            'expert_correctness': fit_scores(features, seen_ok, fit_rows & (seen_ok | unseen_ok)),
        }
        target = correct(gap,cal,gamma)[1]
        test_target = correct(gap,test,gamma)[1]
        record = {}
        for name, scores in candidates.items():
            cutoff = gamma if name=='fixed_current' else threshold(scores,cal,target)
            diagnostic = threshold(scores,test,test_target)
            record[name] = dict(locked=correct(scores,test,cutoff),
                                matched_unseen=correct(scores,test,diagnostic))
        records.append(record)
    totals = {mode:{arm:np.sum([r[arm][mode] for r in records],axis=0).tolist() for arm in records[0]}
              for mode in ('locked','matched_unseen')}
    out = HERE/'convnext_crop_clip_disease/expert_correctness_gate_development.json'
    write_json(out, dict(split=42, images=[len(vs),len(vu)], totals=totals, folds=records))
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
