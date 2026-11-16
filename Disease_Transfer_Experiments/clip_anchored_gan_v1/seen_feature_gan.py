"""Augment low-count seen pairs with the saved crop-anchored ConvNeXt generator."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json

import numpy as np
import torch
from torch.nn import functional as F

from seen_hinge_head import DETAIL, HERE, main


def compare(initial, x, y, val, validation_pairs, rank, report):
    prepared = json.loads((DETAIL / 'runs/development_seed42/prepared.json').read_text())
    fit = prepared['roles']['fit']
    compositions = np.array([row['composition_id'] for row in fit])
    crops = np.array([row['crop_id'] for row in fit])
    diseases = np.array([row['disease_id'] for row in fit])
    healthy = np.array([row['disease_name'] == 'healthy' for row in fit])
    assert np.array_equal(diseases, y.cpu().numpy()) and len(fit) == len(x)
    eligible_crops = set(crops[healthy])
    counts = Counter(compositions)
    pairs = {row['pair_id']: row for row in prepared['pairs']}
    targets = [key for key, count in sorted(counts.items())
               if 8 <= count <= 25 and pairs[key]['crop_id'] in eligible_crops]
    per_pair = 16
    selected = np.repeat(targets, per_pair)
    assert len(selected) and all(not pairs[key]['is_unseen'] for key in targets)
    rng = np.random.default_rng(42)
    crop_pool = {c: np.flatnonzero(healthy & (crops == c)) for c in eligible_crops}
    real_pool = {key: np.flatnonzero(compositions == key) for key in targets}
    selected_crop = np.array([pairs[key]['crop_id'] for key in selected])
    selected_disease = np.array([pairs[key]['disease_id'] for key in selected])
    anchor_ids = np.array([rng.choice(crop_pool[c]) for c in selected_crop])
    repeated_ids = np.array([rng.choice(real_pool[key]) for key in selected])
    general_ids = rng.integers(len(x), size=len(selected))

    gan = HERE.parent / 'feature_gan_anchored_v1'
    spec = importlib.util.spec_from_file_location('anchored_convnext_gan', gan / 'train.py')
    generator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator_module)
    checkpoint = gan / 'runs/seed_42/generator.pt'
    saved = torch.load(checkpoint, map_location=x.device, weights_only=True)
    # The generator was trained on this same seed-42 feature bank and scale.
    assert torch.allclose(x.mean(0), saved['mean'], atol=.002, rtol=.01)
    assert torch.allclose(x.std(0).clamp_min(.01), saved['scale'], atol=.002, rtol=.01)
    generator = generator_module.mlp(1318, 1024).to(x.device).eval()
    generator.load_state_dict(saved['generator'])
    extra_y = torch.as_tensor(selected_disease, device=x.device)
    assert torch.equal(y[repeated_ids], extra_y)
    conditions = generator_module.condition(torch.as_tensor(selected_crop, device=x.device), extra_y)
    random = torch.Generator(device=x.device).manual_seed(42)
    noise = torch.randn((len(selected), 64), device=x.device, generator=random)
    with torch.no_grad():
        anchor = (x[anchor_ids] - saved['mean']) / saved['scale']
        synth = (anchor + generator(torch.cat((anchor, noise, conditions), dim=1))) * saved['scale'] + saved['mean']
    assert torch.isfinite(synth).all()
    print(f'Generated {len(synth)} vectors for {len(targets)} seen pairs', flush=True)
    target_val = np.isin(validation_pairs, targets)
    result = {}
    arms = {'real_only': (x[general_ids], y[general_ids]),
            'repeat_targeted': (x[repeated_ids], extra_y), 'gan_targeted': (synth, extra_y)}
    for name, (extra, labels) in arms.items():
        head = deepcopy(initial)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-4)
        train_x, train_y = torch.cat((x, extra)), torch.cat((y, labels))
        random.manual_seed(43)
        for _ in range(3):
            for batch in torch.randperm(len(train_y), device=x.device, generator=random).split(64):
                loss = F.cross_entropy(head(train_x[batch]), train_y[batch])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        head.eval()
        with torch.inference_mode():
            logits = head(val).cpu().numpy()
        assert np.isfinite(logits).all()
        top = rank(logits, 1.)
        correct = top[:,0] == validation_pairs
        result[name] = dict(**report(logits, top), targeted_top1=int((correct & target_val).sum()),
                            other_top1=int((correct & ~target_val).sum()))
        torch.save(head.state_dict(), HERE / 'seen_feature_gan' / f'{name}_head.pt')
    return dict(epochs=3, extra_vectors_per_pair=per_pair, total_extra_vectors=len(selected),
                targets=[dict(pair=pairs[key]['pair'], fit_images=counts[key]) for key in targets],
                targeted_validation_images=int(target_val.sum()), arms=result)


if __name__ == '__main__':
    main(feature_experiment=compare, experiment_name='seen_feature_gan')
