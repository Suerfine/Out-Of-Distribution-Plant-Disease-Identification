"""Compare capped oversampling during end-to-end ConvNeXt fine-tuning."""
from collections import Counter
import json

import numpy as np
import torch

from within_crop_disease import (
    BASELINE, DETAIL, HERE, fit_model, official_rows, predict, transforms, write_json,
)


def main():
    torch.set_num_threads(4)
    out = HERE / 'seen_oversampling_full'
    out.mkdir(exist_ok=True)
    fit, test = official_rows('train'), official_rows('test_seen')
    assert len(fit) == 8000 and len(test) == 2000
    assert {r['filename'] for r in fit}.isdisjoint(r['filename'] for r in test)
    prepared = json.loads((DETAIL / 'runs/development_seed42/prepared.json').read_text())
    pairs = sorted(prepared['pairs'], key=lambda r: r['pair_id'])
    ci, di = [np.array([r[key] for r in pairs]) for key in ('crop_id', 'disease_id')]
    composition = np.array([r['composition_id'] for r in fit])
    seen_ids = np.unique(composition)
    healthy_crops = {r['crop_id'] for r in pairs if r['disease'] == 'healthy' and r['pair_id'] in seen_ids}
    targets = {key: min(16, count) for key, count in sorted(Counter(composition).items())
               if 8 <= count <= 25 and ci[key] in healthy_crops}
    rng = np.random.default_rng(42)
    targeted = np.concatenate([rng.choice(np.flatnonzero(composition == key), size=count, replace=False)
                               for key, count in targets.items()])
    control = rng.choice(len(fit), size=len(targeted), replace=False)
    selection = dict(epochs=1, training_images=len(fit), extra_images=len(targeted),
                     max_repeat_per_image=1, target_pairs={pairs[k]['pair']: v for k, v in targets.items()})
    prior = json.loads((HERE / 'seen_oversampling/official_results.json').read_text())
    assert selection['target_pairs'] == prior['selection']['target_pairs']
    write_json(out / 'selection.json', selection)
    cfg = json.loads((BASELINE / 'config.json').read_text())
    source = BASELINE / 'disease_net.pt'
    _, tf = transforms(cfg)

    with np.load(HERE / 'clip_disease_only/official/test_seen.npz') as bank:
        assert np.array_equal(bank['filenames'], [r['filename'] for r in test])
        labels, gap, seen_pred = bank['labels'], bank['gap'], bank['seen_pred']
    assert np.array_equal(labels, [r['composition_id'] for r in test])
    with np.load(BASELINE / 'factor_logits.npz') as bank:
        crop = bank['test_seen_crop'][:, ci[seen_ids]]
        reference = crop + bank['test_seen_disease'][:, di[seen_ids]]
    assert np.array_equal(seen_ids[reference.argmax(1)], seen_pred)
    eligible = gap >= 0
    rows = [row for row, include in zip(test, eligible) if include]
    truth = labels[eligible]
    target_test = np.isin(truth, list(targets))
    original = json.loads((HERE / 'clip_disease_only/official/results.json').read_text())['official']

    def evaluate(scores):
        top = seen_ids[np.argsort(-scores, axis=1, kind='stable')[:, :3]]
        correct = top[:, 0] == truth
        result = {}
        for gamma in (0, 3):
            mask = gap[eligible] >= gamma
            n = int((mask & correct).sum())
            a, b = n / len(test), original[str(gamma)]['unseen']
            result[str(gamma)] = dict(top1=n, top3=int((mask & (top == truth[:, None]).any(1)).sum()),
                                      target_correct=int((mask & correct & target_test).sum()),
                                      target_routed=int((mask & target_test).sum()),
                                      seen=a, unseen=b, h=2*a*b/(a+b))
        return result

    results = {'current': evaluate(reference[eligible])}
    assert results['current']['3']['top1'] == original['3']['seen_correct']
    for name, indices in (('real_control', control), ('oversampling', targeted)):
        print(f'Training the full disease network: {name}', flush=True)
        model, _ = fit_model(fit + [fit[i] for i in indices], cfg, source, 0., selection['epochs'],
                             out / name, scope=f'full-data seen oversampling {name}')
        if model is None:
            return
        logits = predict(model, rows, tf, out / name)
        results[name] = evaluate(crop[eligible] + logits[:, di[seen_ids]])
        write_json(out / f'{name}_official.json', results[name])
        print(name, json.dumps(results[name]), flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(out / 'official_results.json', dict(selection=selection, results=results))


if __name__ == '__main__':
    main()
