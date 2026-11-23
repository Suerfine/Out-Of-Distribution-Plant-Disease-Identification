"""Development ablation: the hard gate and the blends with and without GAN vectors in the disease head.

Repeats the selection of soft_gate.py on development split 42 (same images, same ConvNeXt logits, same weight grid,
same offset rule) with the disease head swapped. 'gan' reproduces the original numbers (asserted); 'real_only' uses
disease_real_only_head.pt (head fitted on real training embeddings only). Nothing existing is modified.
"""
import itertools
import json
import sys

import numpy as np
import torch

from within_crop_disease import DETAIL, HERE, write_json
from dpd.feature_gen import CosineClassifier
from soft_gate import log_probs, soft_weight

BASE = HERE.parents[1]  # repository root; within_crop_disease.BASE assumes the folder sits next to the repository

ARMS = {
    'gan': ('disease_plus_gan', 'disease_plus_gan_head.pt'),
    'real_only': ('disease_real_only', 'disease_real_only_head.pt'),
}


def run(arm):
    key, head_file = ARMS[arm]
    prepared = json.loads((DETAIL / 'runs/development_seed42/prepared.json').read_text())
    fit, seen_rows, unseen_rows = (prepared['roles'][name] for name in ('fit', 'validation_seen', 'validation_unseen'))
    rows = seen_rows + unseen_rows
    y = np.array([row['composition_id'] for row in rows])
    seen = np.arange(len(rows)) < len(seen_rows)
    ids = np.union1d([row['composition_id'] for row in fit], y[~seen])
    is_seen_pair = ~np.isin(ids, y[~seen])
    pairs = sorted(prepared['pairs'], key=lambda row: row['pair_id'])
    ci, di = [np.array([pairs[i][k] for i in ids]) for k in ('crop_id', 'disease_id')]
    prior = HERE / 'clip_disease_only'
    gamma = json.loads((prior / 'development_results.json').read_text())[key]['gamma']
    scale = json.loads((BASE / 'config/vlm/v11.3_lora_features_config.json').read_text())['classifier']['logit_scale']
    weight = torch.load(prior / head_file, map_location='cpu', weights_only=True)['weight']
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
    for center, temperature, strength, bias in itertools.product((0., gamma), (.5, 1., 2., 4.), (.5, .75, 1.), (-.15, 0., .15)):
        params = dict(center=center, temperature=temperature, strength=strength, bias=bias)
        soft.append((params, blend(soft_weight(hard[0], **params))))

    def counts(candidate, mask, offset):
        gap, a, b = candidate
        route = gap >= offset
        return [int((mask & route & a & seen).sum()), int((mask & ~route & b & ~seen).sum())]

    def h(correct, mask):
        a, b = np.divide(correct, [(seen & mask).sum(), (~seen & mask).sum()])
        return float((2 * a * b) / (a + b)) if a + b else 0.

    def cutoff(candidate, mask, target):
        vals = np.sort(candidate[0][mask & candidate[1] & seen])
        return float(vals[-min(target, len(vals))]) if len(vals) and target else float('inf')

    everything = np.ones(len(y), dtype=bool)
    if arm == 'gan':
        assert counts(hard, everything, 0) == [753, 126]
        assert counts(hard, everything, gamma) == [593, 315]
    seen_floor = counts(hard, everything, gamma)[0]
    selection, dev = {}, {'hard': {'zero': counts(hard, everything, 0), 'tuned': counts(hard, everything, gamma)}}
    for name, candidates in (('fixed', fixed), ('soft', soft)):
        zero, zc = max(candidates, key=lambda item: h(counts(item[1], everything, 0), everything))
        tuned, tc = max(candidates, key=lambda item: counts(item[1], everything, cutoff(item[1], everything, seen_floor))[1]
                        if (item[1][1] & seen).sum() >= seen_floor else -1)
        off = cutoff(tc, everything, seen_floor)
        selection[name] = dict(zero=dict(params=zero, offset=0.), tuned=dict(params=tuned, offset=off))
        dev[name] = {'zero': counts(zc, everything, 0), 'tuned': counts(tc, everything, off)}
    n_seen, n_unseen = int(seen.sum()), int((~seen).sum())
    summary = {rule: {mode: dict(correct=c, seen=c[0] / n_seen, unseen=c[1] / n_unseen, h=h(c, everything))
                      for mode, c in modes.items()} for rule, modes in dev.items()}
    out = dict(arm=arm, hard_gamma=gamma, selection=selection, development=summary, images=[n_seen, n_unseen])
    write_json(HERE / f'ablation_no_gan/dev_{arm}.json', out)
    return out


if __name__ == '__main__':
    torch.set_num_threads(4)
    (HERE / 'ablation_no_gan').mkdir(exist_ok=True)
    for arm in (sys.argv[1:] or ARMS):
        r = run(arm)
        print(f"\n== {arm}: hard gamma {r['hard_gamma']}")
        for rule, modes in r['development'].items():
            for mode, v in modes.items():
                print(f"  {rule:5s} {mode:5s} seen {v['seen']:.4f} unseen {v['unseen']:.4f} H {v['h']:.4f}")
        print('  selection', json.dumps(r['selection']))
