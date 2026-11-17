"""Compare same-crop hard-negative fine-tuning at fixed development routing."""
import json
from pathlib import Path

import numpy as np
import torch

from within_crop_disease import (
    BASE, DETAIL, HERE, IndependentNet, fit_model, predict, sha256, transforms, write_json,
)


def main(*, anchored=False):
    from dpd.feature_gen import CosineClassifier

    torch.set_num_threads(4)
    out = HERE / ('seen_backbone_anchor' if anchored else 'seen_hard_negative')
    out.mkdir(exist_ok=True)
    p = json.loads((DETAIL / 'runs/development_seed42/prepared.json').read_text())
    cfg = p['baseline_config']
    fit, seen, unseen = (p['roles'][name] for name in ('fit', 'validation_seen', 'validation_unseen'))
    seen_ids = np.unique([r['composition_id'] for r in fit])
    unseen_ids = np.unique([r['composition_id'] for r in unseen])
    pairs = sorted(p['pairs'], key=lambda r: r['pair_id'])
    ci, di = [np.array([r[k] for r in pairs]) for k in ('crop_id', 'disease_id')]
    source = next(Path(k) for k in p['reference_hashes']
                  if Path(k).parent.name == 'disease' and Path(k).name == 'best_epoch_15.pt')
    source_hash = sha256(source)
    assert source_hash == p['reference_hashes'][str(source)]
    disease_run = HERE / 'clip_disease_only'
    gamma = json.loads((disease_run / 'development_results.json').read_text())['disease_plus_gan']['gamma']
    weights = torch.load(disease_run / 'disease_plus_gan_head.pt', map_location='cpu', weights_only=True)['weight']
    scale = json.loads((BASE / 'config/vlm/v11.3_lora_features_config.json').read_text())['classifier']['logit_scale']
    head = CosineClassifier(weights, scale)
    with np.load(HERE / 'runs/seed_42/features.npz') as bank:
        order = {name: i for i, name in enumerate(bank['filenames'])}
        x = torch.from_numpy(bank['embeddings'][[order[r['filename']] for r in seen]])
    with torch.inference_mode():
        clip_disease = head(x).numpy()
    with np.load(p['baseline_validation_logits']) as bank:
        crop, cnn_disease = bank['crop'][:len(seen)], bank['disease'][:len(seen)]
    gate_seen = crop[:, ci[seen_ids]] + clip_disease[:, di[seen_ids]]
    gate_unseen = crop[:, ci[unseen_ids]] + clip_disease[:, di[unseen_ids]]
    routed = gate_seen.max(1) - gate_unseen.max(1) >= gamma
    rows = [row for row, include in zip(seen, routed) if include]
    target = np.array([r['composition_id'] for r in rows])
    crop_scores = crop[routed][:, ci[seen_ids]]
    sources = np.array([r['filename'].split('_', 1)[0] for r in rows])
    _, tf = transforms(cfg)

    def evaluate_logits(logits):
        scores = crop_scores + logits[:, di[seen_ids]]
        top3 = seen_ids[np.argsort(-scores, axis=1, kind='stable')[:, :3]]
        hit = top3 == target[:, None]
        result = dict(correct=int(hit[:, 0].sum()), top3=int(hit.any(1).sum()), images=len(rows))
        if anchored:
            result['by_source'] = {
                key: dict(images=int((sources == key).sum()), correct=int(hit[sources == key, 0].sum()))
                for key in np.unique(sources)
            }
        return result

    def evaluate(model, folder):
        return evaluate_logits(predict(model, rows, tf, folder))

    baseline = evaluate_logits(cnn_disease[routed])
    assert baseline['correct'] == 593
    control_path = HERE / 'within_crop_disease/weight_0.0'
    recipe = json.loads((control_path / 'recipe.json').read_text())
    assert recipe['weight'] == 0 and recipe['fit_images'] == len(fit) and recipe['source_sha256'] == source_hash
    pointer = json.loads((control_path / 'checkpoint_pointer.json').read_text())
    control = []
    for key in ('previous', 'active'):
        ref = pointer[key]
        if anchored and ref['epoch'] != 2:
            continue
        path = control_path / ref['file']
        assert sha256(path) == ref['sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = IndependentNet(cfg, 'disease')
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        control.append(dict(epoch=ref['epoch'], **evaluate(model.cuda(), out)))
        del model
        torch.cuda.empty_cache()
    write_json(out / 'baseline_and_control.json', dict(gamma=gamma, baseline=baseline, control=control))
    print('Fixed-gate baseline and ordinary fine-tuning', baseline['correct'],
          [(r['epoch'], r['correct']) for r in control], flush=True)
    if anchored:
        assert [row['epoch'] for row in control] == [2]
        model, history = fit_model(fit, cfg, source, 0., 2, out / 'anchor', evaluate, backbone_anchor=.1)
    else:
        model, history = fit_model(fit, cfg, source, .25, 3, out / 'margin', evaluate, hard_margin=1.)
    if model is None:
        return
    result = dict(gamma=gamma, baseline=baseline, control=control)
    result['backbone_anchor' if anchored else 'hard_negative'] = history
    write_json(out / 'results.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone-anchor', action='store_true')
    main(anchored=parser.parse_args().backbone_anchor)
