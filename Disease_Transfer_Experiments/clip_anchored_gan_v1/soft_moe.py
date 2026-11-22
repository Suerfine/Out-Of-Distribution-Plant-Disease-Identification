"""Compare hard and probability-weighted routing with disjoint expert labels."""
from itertools import product
import json

import numpy as np
import pandas as pd
import torch

from expert_correctness_gate import BASE, DATA, DETAIL, HERE, borderline_override, fit_gate, gate_score, write_json
from dpd.data import COLUMNS


def inputs(crop, cnn, clip, ci, di, si, ui, labels):
    seen = crop[:, ci[si]] + cnn[:, di[si]]
    unseen = crop[:, ci[ui]] + clip[:, di[ui]]
    clip_seen = crop[:, ci[si]] + clip[:, di[si]]
    gap = clip_seen.max(1) - unseen.max(1)

    def margin(scores):
        top = np.sort(np.partition(scores, -2, axis=1)[:, -2:], axis=1)
        return top[:, 1] - top[:, 0]

    features = np.column_stack((gap, margin(seen), margin(unseen)))
    return features, seen, unseen, si[seen.argmax(1)] == labels, ui[unseen.argmax(1)] == labels


def top_log_probability(scores, temperature):
    scaled = (scores - scores.max(1, keepdims=True)).astype(np.float64) / temperature
    return -np.logaddexp.reduce(scaled, axis=1)


def compare(data, labels, seen, baseline, train=None, test=None, gap_only=False):
    features, seen_scores, unseen_scores, seen_hit, unseen_hit = data
    route = ((features[:, 0] >= 0) | borderline_override(features, 0, baseline['to_seen']))
    route &= ~borderline_override(features, 0, baseline['to_unseen'], to_unseen=True)
    plain_gap = features[:, 0] >= 0
    gate_features = features[:, :1] if gap_only else features

    def stats(route_seen, subset):
        a = int((subset & route_seen & seen_hit).sum())
        b = int((subset & ~route_seen & unseen_hit).sum())
        s, u = a / (subset & seen).sum(), b / (subset & ~seen).sum()
        return dict(seen_correct=a, unseen_correct=b, seen=float(s), unseen=float(u),
                    h=float(2*s*u/(s+u)) if s+u else 0.)

    if train is None:
        mask = np.ones(len(labels), dtype=bool)
        train = []
        rng = np.random.default_rng(42)
        group = {pair: i%5 for ids in (labels[seen], labels[~seen])
                 for i, pair in enumerate(rng.permutation(np.unique(ids)))}
        folds = np.array([group[pair] for pair in labels])
        for fold in range(5):
            held = folds == fold
            train.append((~held, held))
        train.append((mask, mask))
        configs, reports = [], []
        temperatures = (.25, .5, 1., 2., 4., 8.)
        seen_logp = {t: top_log_probability(seen_scores, t) for t in temperatures}
        unseen_logp = {t: top_log_probability(unseen_scores, t) for t in temperatures}
        for fit, held in train:
            model = fit_gate(gate_features, seen, fit)
            probability_log_odds = gate_score(gate_features, model)
            plain = probability_log_odds >= 0
            floor = stats(route, fit)['seen_correct']
            candidates = []
            for st, ut in product(temperatures, repeat=2):
                soft = probability_log_odds + seen_logp[st] - unseen_logp[ut]
                candidates.append((stats(soft >= 0, fit), st, ut, soft >= 0))
            chosen = max(candidates, key=lambda t: (t[0]['seen_correct'] >= floor,
                          t[0]['h'] if t[0]['seen_correct'] >= floor else t[0]['seen_correct']))
            _, st, ut, soft = chosen
            configs.append(dict(model=model, seen_temperature=st, unseen_temperature=ut))
            reports.append({name: stats(r, held) for name, r in
                            [('two_way', route), ('plain_gap', plain_gap),
                             ('learned_hard', plain), ('learned_soft', soft)]})
        return configs[-1], dict(folds=reports[:-1], development_fit=reports[-1])

    model = train['model']
    odds = gate_score(gate_features, model)
    soft = odds + top_log_probability(seen_scores, train['seen_temperature'])
    soft -= top_log_probability(unseen_scores, train['unseen_temperature'])
    return {name: stats(r, test) for name, r in [('two_way', route), ('plain_gap', plain_gap), ('learned_hard', odds >= 0),
                                                ('learned_soft', soft >= 0)]}


def lora_official_scores(rows, pairs, scale, directory):
    from dpd.clip_utils import clip_transforms
    from dpd.feature_gen import CosineClassifier
    from dpd.resize_cache import ensure_resized
    from train import train_lora

    cfg = json.loads((BASE/'config/vlm/v10.3_text_anchor_three_splits_config.json').read_text())
    cfg['model']['pretrained'] = str(HERE/'ViT-B-16-state.pt')
    cfg['runtime']['num_workers'] = 0
    model, _ = train_lora.build(cfg, pd.DataFrame(pairs), 'cuda')
    model.clip.visual.load_state_dict(torch.load(directory/'visual_encoder.pt', map_location='cpu', weights_only=True))
    state = torch.load(directory/'disease_plus_gan_head.pt', map_location='cpu', weights_only=True)
    head = CosineClassifier(state['weight'], scale).cuda().eval()
    _, tf = clip_transforms(cfg['input'], cfg['augmentation'])
    scores = []
    for frame in rows:
        cache = ensure_resized(frame.file, cfg['input']['resize_cache'])
        loader = train_lora.loader(frame, cache, tf, cfg, shuffle=False)
        x, _ = train_lora.encode(model, loader, 'cuda', cfg['runtime']['amp'],
                                 tuple(cfg['evaluation'].get('tta', ['identity'])))
        with torch.inference_mode():
            scores.append(head(torch.as_tensor(x, device='cuda')).cpu().numpy())
    return np.concatenate(scores)


def main(gap_only=False, lora=False):
    torch.set_num_threads(4)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    pairs = sorted(p['pairs'], key=lambda row: row['pair_id'])
    ci, di = [np.array([r[k] for r in pairs]) for k in ('crop_id', 'disease_id')]
    fit, vs, vu = [p['roles'][k] for k in ('fit','validation_seen','validation_unseen')]
    ids = lambda rows: np.array([r['composition_id'] for r in rows])
    ys = ids(vs+vu)
    development_seen = np.arange(len(ys)) < len(vs)
    out = HERE/'siglip_gate_pilot'
    if lora:
        from dpd.feature_gen import CosineClassifier

        prior = HERE/'clip_disease_only'
        artifact = HERE/'convnext_crop_clip_disease/two_way_override_development.json'
        reference = json.loads(artifact.read_text())['0']['two_way']
        scale = json.loads((BASE/'config/vlm/v11.3_lora_features_config.json').read_text())['classifier']['logit_scale']
        weights = torch.load(prior/'disease_plus_gan_head.pt', map_location='cpu', weights_only=True)['weight']
        with np.load(HERE/'runs/seed_42/features.npz') as bank:
            order = {name:i for i,name in enumerate(bank['filenames'])}
            x = torch.from_numpy(bank['embeddings'][[order[row['filename']] for row in vs+vu]])
        with torch.inference_mode():
            dev_scores = CosineClassifier(weights, scale)(x).numpy()
        official_dir = prior/'official'
    else:
        reference = json.loads((out/'development_results.json').read_text())['results']['clip']['zero_offset']['two_way']
        dev_scores = np.load(out/'clip_development_scores.npy')
        official_dir = out/'official_frozen_clip'
    with np.load(p['baseline_validation_logits']) as bank:
        dev = inputs(bank['crop'], bank['disease'], dev_scores,
                     ci, di, np.unique(ids(fit)), np.unique(ids(vu)), ys)
    selection, diagnostics = compare(dev, ys, development_seen, reference, gap_only=gap_only)
    print('Locked development settings', json.dumps(selection), flush=True)
    splits = ('test_seen','test_unseen')
    rows = [pd.read_csv(DATA/f'DPD_Small_{split}.csv', header=None, names=COLUMNS) for split in splits]
    for split, frame in zip(splits, rows):
        with np.load(HERE/'clip_disease_only/official'/f'{split}.npz') as bank:
            assert np.array_equal(bank['filenames'], frame.file.to_numpy())
    labels = np.concatenate([frame.pair_id.to_numpy() for frame in rows])
    seen = np.arange(len(labels)) < len(rows[0])
    path = BASE/'runs/baseline/independent_cnn_baseline_convnext_base_seed42/factor_logits.npz'
    with np.load(path) as bank:
        crop, cnn = [np.concatenate([bank[f'{split}_{kind}'] for split in splits]) for kind in ('crop','disease')]
    unseen_cols = np.array([r['is_unseen'] for r in pairs])
    score_path = official_dir/'clip_scores.npy'
    if lora and not score_path.exists():
        clip_scores = lora_official_scores(rows, pairs, scale, official_dir)
        np.save(score_path, clip_scores)
    else:
        clip_scores = np.load(score_path)
    data = inputs(crop, cnn, clip_scores, ci, di,
                  np.flatnonzero(~unseen_cols), np.flatnonzero(unseen_cols), labels)
    official = compare(data, labels, seen, reference, train=selection,
                       test=np.ones(len(labels),dtype=bool), gap_only=gap_only)
    expected = [1447,863] if lora else [1339,1095]
    assert [official['two_way'][k] for k in ('seen_correct','unseen_correct')] == expected
    report = dict(selection=selection, development=diagnostics, official=official)
    filename = 'gap_soft_moe_comparison.json' if gap_only else 'soft_moe_comparison.json'
    write_json(official_dir/filename, report)
    print(json.dumps(official), flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gap-only', action='store_true')
    parser.add_argument('--lora', action='store_true')
    args = parser.parse_args()
    main(args.gap_only, args.lora)
