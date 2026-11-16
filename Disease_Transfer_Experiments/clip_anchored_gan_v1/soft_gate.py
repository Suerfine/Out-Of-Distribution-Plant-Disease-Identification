"""Compare fixed and image-dependent score blends on seed-42 development."""
import itertools
import json

import numpy as np
import torch

from within_crop_disease import BASE, DETAIL, HERE, write_json
from dpd.feature_gen import CosineClassifier


def log_probs(scores):
    return scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)


def soft_weight(gap, center, temperature, strength, bias):
    transition = 1 / (1 + np.exp(np.clip((center - gap) / temperature, -50, 50)))
    return np.clip(.5 + bias + strength * (transition - .5), 0., 1.)[:, None]


def main():
    torch.set_num_threads(4)
    prepared = json.loads((DETAIL / 'runs/development_seed42/prepared.json').read_text())
    fit, seen_rows, unseen_rows = (prepared['roles'][name]
                                  for name in ('fit', 'validation_seen', 'validation_unseen'))
    rows = seen_rows + unseen_rows
    y = np.array([row['composition_id'] for row in rows])
    seen = np.arange(len(rows)) < len(seen_rows)
    ids = np.union1d([row['composition_id'] for row in fit], y[~seen])
    is_seen_pair = ~np.isin(ids, y[~seen])
    pairs = sorted(prepared['pairs'], key=lambda row: row['pair_id'])
    ci, di = [np.array([pairs[i][key] for i in ids]) for key in ('crop_id', 'disease_id')]
    prior = HERE / 'clip_disease_only'
    gamma = json.loads((prior / 'development_results.json').read_text())['disease_plus_gan']['gamma']
    scale = json.loads((BASE / 'config/vlm/v11.3_lora_features_config.json').read_text())['classifier']['logit_scale']
    weight = torch.load(prior / 'disease_plus_gan_head.pt', map_location='cpu', weights_only=True)['weight']
    with np.load(HERE / 'runs/seed_42/features.npz') as bank:
        index = {name: i for i, name in enumerate(bank['filenames'])}
        x = torch.from_numpy(bank['embeddings'][[index[row['filename']] for row in rows]])
    with torch.inference_mode():
        clip_disease = CosineClassifier(weight, scale)(x).numpy()
    with np.load(prepared['baseline_validation_logits']) as cnn:
        crop = cnn['crop'][:, ci]
        cnn_scores = crop + cnn['disease'][:, di]
    clip_scores = crop + clip_disease[:, di]

    def decisions(scores, seen_scores=None):
        a, b = scores[:, is_seen_pair], scores[:, ~is_seen_pair]
        seen_choice = a if seen_scores is None else seen_scores[:, is_seen_pair]
        return a.max(1) - b.max(1), ids[is_seen_pair][seen_choice.argmax(1)] == y, ids[~is_seen_pair][b.argmax(1)] == y

    hard = decisions(clip_scores, cnn_scores)
    clip_norm, cnn_norm = (log_probs(scores.astype(np.float64)) for scores in (clip_scores, cnn_scores))

    def blend(w):
        return decisions(clip_norm + w * (cnn_norm - clip_norm))

    fixed = [(dict(weight=w), blend(w)) for w in (.1, .25, .4, .5, .6, .75, .9)]
    soft = []
    for center, temperature, strength, bias in itertools.product(
        (0., gamma), (.5, 1., 2., 4.), (.5, .75, 1.), (-.15, 0., .15)
    ):
        params = dict(center=center, temperature=temperature, strength=strength, bias=bias)
        soft.append((params, blend(soft_weight(hard[0], **params))))

    def counts(candidate, mask, offset):
        gap, a, b = candidate
        route = gap >= offset
        return [int((mask & route & a & seen).sum()), int((mask & ~route & b & ~seen).sum())]

    def h(correct, mask):
        a, b = np.divide(correct, [(seen & mask).sum(), (~seen & mask).sum()])
        return float((2*a*b)/(a+b)) if a+b else 0.

    def cutoff(candidate, mask, target):
        vals = np.sort(candidate[0][mask & candidate[1] & seen])
        return float(vals[-min(target, len(vals))]) if len(vals) and target else float('inf')

    all_images = np.ones(len(y), dtype=bool)
    assert counts(hard, all_images, 0) == [753, 126]
    assert counts(hard, all_images, gamma) == [593, 315]
    rng = np.random.default_rng(42)
    fold = {pair: i % 5 for group in (y[seen], y[~seen])
            for i, pair in enumerate(rng.permutation(np.unique(group)))}
    fold_id = np.array([fold[pair] for pair in y])
    results = []
    for k in range(5):
        test = fold_id == k
        cal = ~test
        assert set(y[test]).isdisjoint(y[cal])
        floor = counts(hard, cal, gamma)[0]
        target = counts(hard, test, gamma)[0]
        row = dict(fold=k, arms={})
        for name, candidates in [('hard', [({}, hard)]), ('fixed', fixed), ('soft', soft)]:
            params, candidate = max(candidates, key=lambda item: h(counts(item[1], cal, 0), cal))
            zero = dict(params=params, counts=counts(candidate, test, 0))
            params, candidate = max(candidates, key=lambda item: sum(counts(
                item[1], cal, cutoff(item[1], cal, floor))) if
                (cal & item[1][1] & seen).sum() >= floor else -1)
            offset = cutoff(candidate, cal, floor)
            matched = cutoff(candidate, test, target)
            row['arms'][name] = dict(zero=zero, locked=dict(params=params, offset=offset,
                counts=counts(candidate, test, offset)), matched=dict(counts=counts(candidate, test, matched)))
        results.append(row)
    totals = {}
    for mode in ('zero', 'locked', 'matched'):
        totals[mode] = {}
        for name in ('hard', 'fixed', 'soft'):
            correct = np.sum([row['arms'][name][mode]['counts'] for row in results], axis=0).tolist()
            totals[mode][name] = dict(correct=correct, h=h(correct, all_images))
    selection = {}
    seen_floor = counts(hard, all_images, gamma)[0]
    for name, candidates in [('fixed', fixed), ('soft', soft)]:
        zero, _ = max(candidates, key=lambda item: h(counts(item[1], all_images, 0), all_images))
        tuned, blend_scores = max(candidates, key=lambda item: counts(
            item[1], all_images, cutoff(item[1], all_images, seen_floor))[1]
            if (item[1][1] & seen).sum() >= seen_floor else -1)
        selection[name] = dict(zero=dict(params=zero, offset=0.),
                               tuned=dict(params=tuned, offset=cutoff(blend_scores, all_images, seen_floor)))
    output = dict(split=42, images=[len(seen_rows), len(unseen_rows)], totals=totals,
                  folds=results, official_selection=selection)
    write_json(HERE / 'convnext_crop_clip_disease/soft_gate_development.json', output)
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    main()
