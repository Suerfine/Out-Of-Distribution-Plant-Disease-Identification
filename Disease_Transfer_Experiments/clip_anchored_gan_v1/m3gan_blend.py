"""Blend of the pair-based CLIP + crop-anchored GAN (v10.3 recipe + GAN) with the ConvNeXt baseline (M1), selected on development.

Uses only saved scores: development pair scores from hybrid/development_scores.npz, official pair scores from
official/<arm>_test_{seen,unseen}.npz, ConvNeXt logits from the baseline runs. The pair score is built exactly as in
clip_disease_only.py ('pair_based' arm): 100 x cosine score, log-softmax within each crop, plus the ConvNeXt crop logit.
Step 1 asserts that the hard gate reproduces the published counts (development and official). Step 2 repeats the blend
selection of soft_gate.py on development with this CLIP expert and applies it once to the official test.
Nothing existing is modified; the result goes to ablation_no_gan/m3gan_blend.json.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from dpd.data import COLUMNS  # noqa: E402
from dpd.splits import make_validation_split  # noqa: E402

DETAIL = HERE.parent / 'cnn_detail_base_v1'
DATA = REPO.parent / 'Plant_Disease'
OUT = HERE / 'ablation_no_gan'


def log_probs(s):
    return s - np.logaddexp.reduce(s, axis=1, keepdims=True)


def soft_weight(gap, center, temperature, strength, bias):
    t = 1 / (1 + np.exp(np.clip((center - gap) / temperature, -50, 50)))
    return np.clip(.5 + bias + strength * (t - .5), 0., 1.)[:, None]


def pair_scores(cos, crop, ci, ids):
    """cos: raw pair scores (n, 333); crop: ConvNeXt crop logits already indexed to ids."""
    s = 100 * cos[:, ids].astype(np.float64)
    for c in np.unique(ci[ids]):
        cols = ci[ids] == c
        s[:, cols] -= np.logaddexp.reduce(s[:, cols], axis=1)[:, None]
    return s + crop


def main(arm='anchored'):
    prepared = json.loads((DETAIL / 'runs/development_seed42/prepared.json').read_text())
    pairs = pd.DataFrame(prepared['pairs']).sort_values('pair_id').reset_index(drop=True)
    ci, di = pairs.crop_id.to_numpy(), pairs.disease_id.to_numpy()
    cfg = json.loads((REPO / 'config/vlm/v11.3_lora_features_config.json').read_text())['validation']
    train = pd.read_csv(DATA / 'DPD_Small_train.csv', header=None, names=COLUMNS)
    fit, vs, vu, held = make_validation_split(train, cfg['n_holdout_pairs'], cfg['min_holdout_images'], cfg['val_fraction'], 42,
                                              tuple(cfg['exclude_holdout_diseases']), cfg['max_images_per_holdout_pair'])
    held = np.sort(held)
    assert set(fit.file) == {r['filename'] for r in prepared['roles']['fit']}
    seen_pairs = np.array(sorted(set(fit.pair_id)))
    ids = np.sort(np.r_[seen_pairs, held])
    seen_cols = np.isin(ids, seen_pairs)

    # ---------- development ----------
    val = pd.concat((vs, vu), ignore_index=True)
    y = val.pair_id.to_numpy()
    is_seen = np.arange(len(y)) < len(vs)
    conventional = prepared['roles']['validation_seen'] + prepared['roles']['validation_unseen']
    index = {r['filename']: i for i, r in enumerate(conventional)}
    order = np.array([index[n] for n in val.file])
    with np.load(prepared['baseline_validation_logits']) as cnn, np.load(HERE / 'hybrid/development_scores.npz') as clip:
        crop, cnn_dis = cnn['crop'][order][:, ci[ids]], cnn['disease'][order][:, di[ids]]
        pos = {n: i for i, n in enumerate(clip['filenames'])}
        cos = clip['clip'][[pos[n] for n in val.file]]
    cnn_dev = crop + cnn_dis
    clip_dev = pair_scores(cos, crop, ci, ids)

    def decisions(scores, seen_scores=None):
        a, b = scores[:, seen_cols], scores[:, ~seen_cols]
        sa = a if seen_scores is None else seen_scores[:, seen_cols]
        return a.max(1) - b.max(1), ids[seen_cols][sa.argmax(1)], ids[~seen_cols][b.argmax(1)]

    def counts(cand, mask, offset):
        gap, sp, up = cand
        pred = np.where(gap >= offset, sp, up)
        hit = pred == y
        return [int((hit & is_seen & mask).sum()), int((hit & ~is_seen & mask).sum())]

    def h(c, mask):
        a, b = np.divide(c, [(is_seen & mask).sum(), (~is_seen & mask).sum()])
        return float(2 * a * b / (a + b)) if a + b else 0.

    everything = np.ones(len(y), dtype=bool)
    hard = decisions(clip_dev, cnn_dev)
    assert counts(hard, everything, 0.) == [762, 122], counts(hard, everything, 0.)
    assert counts(hard, everything, 2.9) == [613, 334], counts(hard, everything, 2.9)
    print('development hard gate reproduces the published counts [762,122] and [613,334]', flush=True)

    clip_n, cnn_n = log_probs(clip_dev), log_probs(cnn_dev)

    def blend(w):
        mix = clip_n + w * (cnn_n - clip_n)
        gap, a, b = mix[:, seen_cols].max(1) - mix[:, ~seen_cols].max(1), ids[seen_cols][mix[:, seen_cols].argmax(1)], ids[~seen_cols][mix[:, ~seen_cols].argmax(1)]
        return gap, a, b

    def cutoff(cand, mask, target):
        gap, sp, _ = cand
        vals = np.sort(gap[mask & is_seen & (sp == y)])
        return float(vals[-min(target, len(vals))]) if len(vals) and target else float('inf')

    fixed = [(dict(weight=w), blend(w)) for w in (.1, .25, .4, .5, .6, .75, .9)]
    soft = []
    for center, temp, strength, bias in itertools.product((0., 2.9), (.5, 1., 2., 4.), (.5, .75, 1.), (-.15, 0., .15)):
        params = dict(center=center, temperature=temp, strength=strength, bias=bias)
        soft.append((params, blend(soft_weight(hard[0], **params))))
    gamma = 2.9
    floor = counts(hard, everything, gamma)[0]
    selection, dev = {}, {'hard': {'zero': counts(hard, everything, 0.), 'tuned': counts(hard, everything, gamma)}}
    for name, cands in (('fixed', fixed), ('soft', soft)):
        z, zc = max(cands, key=lambda it: h(counts(it[1], everything, 0.), everything))
        t, tc = max(cands, key=lambda it: counts(it[1], everything, cutoff(it[1], everything, floor))[1]
                    if int((it[1][1] == y)[is_seen].sum()) >= floor else -1)
        off = cutoff(tc, everything, floor)
        selection[name] = dict(zero=dict(params=z, offset=0.), tuned=dict(params=t, offset=off))
        dev[name] = {'zero': counts(zc, everything, 0.), 'tuned': counts(tc, everything, off)}
    n_s, n_u = int(is_seen.sum()), int((~is_seen).sum())
    dev_summary = {r: {m: dict(correct=c, seen=c[0] / n_s, unseen=c[1] / n_u, h=h(c, everything)) for m, c in d.items()} for r, d in dev.items()}

    # ---------- official ----------
    ids_o = pairs.pair_id.to_numpy()
    seen_o = ~pairs.is_unseen.to_numpy()
    assert len(ids_o) == 333 and seen_o.sum() == 323
    frames = {s: pd.read_csv(DATA / f'DPD_Small_{s}.csv', header=None, names=COLUMNS) for s in ('test_seen', 'test_unseen')}
    labels, seen_t, crop_t, cnn_t, cos_t = [], [], [], [], []
    with np.load(REPO / 'runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz') as cnn:
        for s, frame in frames.items():
            with np.load(HERE / f'official/{arm}_{s}.npz', allow_pickle=True) as bank:
                assert np.array_equal(bank['filenames'], frame.file.to_numpy())
                cos_t.append(bank['scores'])
            labels.append(frame.pair_id.to_numpy())
            seen_t.append(np.full(len(frame), s == 'test_seen'))
            crop_t.append(cnn[f'{s}_crop'][:, ci[ids_o]])
            cnn_t.append(cnn[f'{s}_disease'][:, di[ids_o]])
    labels, seen_t = np.concatenate(labels), np.concatenate(seen_t)
    crop_t, cnn_dis_t, cos_t = np.concatenate(crop_t), np.concatenate(cnn_t), np.concatenate(cos_t)
    cnn_off = crop_t + cnn_dis_t
    clip_off = pair_scores(cos_t, crop_t, ci, ids_o)

    def official(rule, params, offset):
        if rule == 'hard':
            gap = clip_off[:, seen_o].max(1) - clip_off[:, ~seen_o].max(1)
            sp = ids_o[seen_o][cnn_off[:, seen_o].argmax(1)]
            up = ids_o[~seen_o][clip_off[:, ~seen_o].argmax(1)]
            pred = np.where(gap >= offset, sp, up)
        else:
            gap_clip = clip_off[:, seen_o].max(1) - clip_off[:, ~seen_o].max(1)
            w = params['weight'] if rule == 'fixed' else soft_weight(gap_clip, **params)
            cn, nn = log_probs(clip_off), log_probs(cnn_off)
            mix = cn + w * (nn - cn)
            gap = mix[:, seen_o].max(1) - mix[:, ~seen_o].max(1)
            pred = np.where(gap >= offset, ids_o[seen_o][mix[:, seen_o].argmax(1)], ids_o[~seen_o][mix[:, ~seen_o].argmax(1)])
        hit = pred == labels
        n, m = int((hit & seen_t).sum()), int((hit & ~seen_t).sum())
        a, b = n / seen_t.sum(), m / (~seen_t).sum()
        return dict(correct=[n, m], seen=float(a), unseen=float(b), h=float(2 * a * b / (a + b)) if a + b else 0.)

    off = {'hard': {'zero': official('hard', None, 0.), 'tuned': official('hard', None, gamma)}}
    assert off['hard']['zero']['correct'] == [1466, 589], off['hard']['zero']['correct']
    assert off['hard']['tuned']['correct'] == [1385, 1123], off['hard']['tuned']['correct']
    print('official hard gate reproduces the published counts [1466,589] and [1385,1123]', flush=True)
    for rule in ('fixed', 'soft'):
        off[rule] = {m: official(rule, selection[rule][m]['params'], selection[rule][m]['offset']) for m in ('zero', 'tuned')}
    result = dict(arm=arm, selection=selection, development=dev_summary, official=off)
    OUT.mkdir(exist_ok=True)
    (OUT / 'm3gan_blend.json').write_text(json.dumps(result, indent=2))
    for name, tab in (('DEVELOPMENT', dev_summary), ('OFFICIAL TEST', off)):
        print(f'\n== {name}')
        for rule in ('hard', 'fixed', 'soft'):
            for mode, v in tab[rule].items():
                print(f"  {rule:5s} {mode:5s} seen {v['seen']:.4f} unseen {v['unseen']:.4f} H {v['h']:.4f}  {v['correct']}")
    print('\nselection', json.dumps(selection))


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['anchored']))
