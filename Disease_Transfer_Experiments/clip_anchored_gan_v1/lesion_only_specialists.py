"""Compare full-leaf and lesion-only specialists on held-out donors."""
import json
import sys

import numpy as np
import torch

from seen_lesion_crop import HERE, reviewed_views
from within_crop_disease import DETAIL, IndependentNet, transforms, write_json


def main():
    sys.path.insert(0, str(HERE.parent / 'leafgan_v1'))
    from classifier_multi import Images

    torch.set_num_threads(4)
    prepared = json.loads((DETAIL / 'runs/development_seed42/prepared.json').read_text())
    donors = {}
    for version in (1, 2, 3):
        manifest = HERE.parent / f'lesion_copy_paste_base_v{version}/composites.json'
        for row in json.loads(manifest.read_text()):
            donors.setdefault(row['donor'], None)
    fit = {row['filename']: row for row in prepared['roles']['fit']}
    rows = [fit[name] for name in donors]
    cfg = prepared['baseline_config']
    model = IndependentNet(cfg, 'disease', pretrained=True).cuda().eval()
    _, transform = transforms(cfg)
    batches = torch.utils.data.DataLoader(Images(rows, transform), batch_size=16)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        full = torch.cat([model.backbone(images.cuda()).float().cpu() for images, _ in batches])
    views, targets, _ = reviewed_views(rows, full, model, transform)
    labels = np.array([row['disease_id'] for row in rows])
    assert np.array_equal(targets.cpu().numpy(), np.repeat(labels, 4))
    crops = np.array([row['crop_name'] for row in rows])
    features = {'full_leaf': full.numpy(), **{
        name: views[name].cpu().numpy().reshape(len(rows), 4, -1).mean(axis=1)
        for name in ('corner', 'reviewed_lesion')
    }}
    results = {}
    for crop in np.unique(crops):
        idx = np.flatnonzero(crops == crop)
        classes, counts = np.unique(labels[idx], return_counts=True)
        if len(classes) < 2 or counts.min() < 2:
            continue
        result = dict(donors=len(idx), disease_counts={
            row['disease_name']: int((labels[idx] == row['disease_id']).sum())
            for row in rows if row['crop_name'] == crop
        }, filenames=[rows[i]['filename'] for i in idx], truth=labels[idx].tolist(), arms={})
        for name, vectors in features.items():
            x = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            predictions = []
            for i in idx:
                train = idx[idx != i]
                # A regularized linear head in dual form, fitted on the other donors.
                basis = x[train]
                one_hot = labels[train, None] == classes[None, :]
                kernel = basis @ basis.T + .1 * np.eye(len(train))
                head = np.linalg.solve(kernel, one_hot)
                predictions.append(int(classes[(x[i] @ basis.T @ head).argmax()]))
            result['arms'][name] = dict(correct=int((np.array(predictions) == labels[idx]).sum()),
                                        predictions=predictions)
        results[str(crop)] = result
    assert results
    out = HERE / 'seen_lesion_crop/lesion_only_specialists.json'
    write_json(out, dict(reviewed_donors=len(rows), backbone='ImageNet-pretrained ConvNeXt-Base',
                         protocol='leave one source leaf out', results=results))
    for crop, result in results.items():
        print(crop, result['disease_counts'], {name: value['correct'] for name, value in result['arms'].items()}, flush=True)


if __name__ == '__main__':
    main()
