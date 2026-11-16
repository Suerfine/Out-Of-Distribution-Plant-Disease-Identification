"""Evaluate capped seen-pair oversampling with frozen ConvNeXt features."""
from collections import Counter
from copy import deepcopy
import json
import sys

import numpy as np
import torch
from torch.nn import functional as F

from within_crop_disease import BASELINE, DETAIL, HERE, IndependentNet, official_rows, transforms, write_json


def main():
    sys.path.insert(0, str(HERE.parent / 'leafgan_v1'))
    from classifier_multi import Images

    torch.set_num_threads(4)
    out = HERE / 'seen_oversampling'
    out.mkdir(exist_ok=True)
    fit, test = official_rows('train'), official_rows('test_seen')
    assert len(fit) == 8000 and len(test) == 2000
    assert {r['filename'] for r in fit}.isdisjoint(r['filename'] for r in test)
    p = json.loads((DETAIL / 'runs/development_seed42/prepared.json').read_text())
    pairs = sorted(p['pairs'], key=lambda r:r['pair_id'])
    ci, di = [np.array([r[k] for r in pairs]) for k in ('crop_id', 'disease_id')]
    composition = np.array([r['composition_id'] for r in fit])
    seen_ids = np.unique(composition)
    assert len(seen_ids) == 323
    healthy_crops = {r['crop_id'] for r in pairs if r['disease'] == 'healthy' and r['pair_id'] in seen_ids}
    targets = {key: min(16, count) for key,count in sorted(Counter(composition).items())
               if 8 <= count <= 25 and ci[key] in healthy_crops}
    rng = np.random.default_rng(42)
    targeted = np.concatenate([rng.choice(np.flatnonzero(composition == key), size=count, replace=False)
                               for key,count in targets.items()])
    random = rng.choice(len(fit), size=len(targeted), replace=False)
    configuration = dict(epochs=3, max_extra_per_pair=16, max_repeat_per_image=1, training_images=len(fit),
                         extra_images_per_epoch=len(targeted), target_pairs={pairs[k]['pair']: v for k,v in targets.items()})
    write_json(out / 'selection.json', configuration)

    cfg = json.loads((BASELINE / 'config.json').read_text())
    net = IndependentNet(cfg, 'disease')
    net.load_state_dict(torch.load(BASELINE/'disease_net.pt', map_location='cpu', weights_only=True))
    net.cuda().eval()
    _, tf = transforms(cfg)
    with np.load(HERE/'clip_disease_only/official/test_seen.npz') as bank:
        assert np.array_equal(bank['filenames'], [r['filename'] for r in test])
        labels, gap, seen_pred = bank['labels'], bank['gap'], bank['seen_pred']
    assert np.array_equal(labels, [r['composition_id'] for r in test])
    with np.load(BASELINE/'factor_logits.npz') as bank:
        crop = bank['test_seen_crop'][:,ci[seen_ids]]
        reference = crop + bank['test_seen_disease'][:,di[seen_ids]]
    assert np.array_equal(seen_ids[reference.argmax(1)], seen_pred)
    eligible = gap >= 0
    routed_test = [row for row, keep in zip(test, eligible) if keep]

    def encode(rows):
        loader = torch.utils.data.DataLoader(Images(rows, tf), batch_size=16)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.cat([net.backbone(images.cuda()).float().cpu() for images, _ in loader])
        return x.cuda()

    print(f'Encoding training images and {len(routed_test)} official seen images', flush=True)
    x, val = encode(fit), encode(routed_test)
    net.backbone.cpu()
    y = torch.as_tensor(di[composition], device=x.device)
    original = json.loads((HERE/'clip_disease_only/official/results.json').read_text())['official']
    truth = labels[eligible]
    targeted_test = np.isin(truth, list(targets))

    def evaluate(scores):
        top = seen_ids[np.argsort(-scores, axis=1, kind='stable')[:,:3]]
        result = {}
        for gamma in (0, 3):
            use = gap[eligible] >= gamma
            correct = top[:,0] == truth
            seen = float((use & correct).sum()/len(test))
            unseen = original[str(gamma)]['unseen']
            result[str(gamma)] = dict(top1=int((use & correct).sum()),
                                      top3=int((use & (top == truth[:,None]).any(1)).sum()),
                                      target_correct=int((use & correct & targeted_test).sum()),
                                      target_routed=int((use & targeted_test).sum()),
                                      seen=seen, unseen=unseen, h=2*seen*unseen/(seen+unseen))
        return result

    results = {'current': evaluate(reference[eligible])}
    assert results['current']['3']['top1'] == original['3']['seen_correct']
    for name, indices in [('real_control', random), ('oversampling', targeted)]:
        head = deepcopy(net.head)
        optim = torch.optim.AdamW(head.parameters(), lr=1e-4)
        train_x, train_y = torch.cat((x, x[indices])), torch.cat((y, y[indices]))
        rng = torch.Generator(device=x.device).manual_seed(43)
        for _ in range(configuration['epochs']):
            for idx in torch.randperm(len(train_y), device=x.device, generator=rng).split(64):
                loss = F.cross_entropy(head(train_x[idx]), train_y[idx])
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
        head.eval()
        with torch.inference_mode():
            logits = head(val).cpu().numpy()
        assert np.isfinite(logits).all()
        results[name] = evaluate(crop[eligible] + logits[:,di[seen_ids]])
        torch.save(head.state_dict(), out/f'{name}_head.pt')
    write_json(out/'official_results.json', dict(selection=configuration, results=results))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
