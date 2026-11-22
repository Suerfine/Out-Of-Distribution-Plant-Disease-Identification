"""Run a frozen-backbone pilot against the saved development model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
DETAIL = HERE.parent/'cnn_detail_base_v1'
sys.path.insert(0, str(DETAIL))
from protocol import seed_all, status, write_json, sha256
from evaluation import composed
import pandas as pd
from independent_cnn_baseline.models import IndependentNet, transforms


class Images(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row.get('legacy_cache_path', row['path'])) as im:
            image = self.transform(im.convert('RGB'))
        return image, int(row['disease_id'])


@torch.inference_mode()
def features(model, rows, transform, run, split):
    vectors, labels = [], []
    batches = DataLoader(Images(rows, transform), batch_size=16, num_workers=2)
    for step, (x, y) in enumerate(batches):
        with torch.autocast('cuda', dtype=torch.float16):
            vectors.append(model.backbone(x.cuda()).float().cpu())
        labels.append(y)
        if step % 50 == 0:
            status(run, 'extracting_features', split=split, images=min(step*16+len(y), len(rows)))
    return torch.cat(vectors), torch.cat(labels)


def main(arm):
    run = HERE/'multi_pair'/arm
    run.mkdir(exist_ok=True)
    seed_all(42)
    torch.set_num_threads(4)
    p = json.loads((DETAIL/'runs/development_seed42/prepared.json').read_text())
    model_path = next(path for path in p['reference_hashes'] if path.replace(chr(92), '/').endswith('/disease/best_epoch_15.pt'))
    for path in (model_path, p['baseline_validation_logits']):
        if sha256(Path(path)) != p['reference_hashes'][path]:
            raise RuntimeError(f'Saved reference changed: {path}')
    base = p['baseline_config']
    model = IndependentNet(base, 'disease')
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu')['model_state'])
    model = model.cuda().eval()
    roles = p['roles']
    synthetic = json.loads((HERE/'multi_pair/composites.json').read_text())
    fit_names = {row['filename'] for row in roles['fit']}
    if any(row['recipient'] not in fit_names for row in synthetic):
        raise RuntimeError('Composite uses an image outside fitting')
    generated_pairs = {(r['crop_id'],r['disease_id']) for r in synthetic}
    comparison = synthetic
    if arm == 'real_only':
        rng = np.random.default_rng(42)
        pools = {pair: [r for r in roles['fit'] if (r['crop_id'],r['disease_id']) == pair]
                 for pair in generated_pairs}
        comparison = [rng.choice(pools[(r['crop_id'],r['disease_id'])]) for r in synthetic]
    _, transform = transforms(base)
    x, y = features(model, roles['fit']+comparison, transform, run, 'training')
    vx, _ = features(model, roles['validation_seen']+roles['validation_unseen'], transform, run, 'validation')
    full_rows = roles['validation_seen'] + roles['validation_unseen']
    target_pairs = [(r['crop_id'],r['disease_id']) for r in full_rows]
    augmented = np.array([pair in generated_pairs for pair in target_pairs])
    transferred = np.array([pair not in generated_pairs and pair[1] in {d for _,d in generated_pairs}
                            for pair in target_pairs])
    x,y,vx = x.cuda(),y.cuda(),vx.cuda()
    model.backbone.cpu()
    model.head.train()
    opt = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=.1)
    history = []
    for epoch in range(1,4):
        order = torch.randperm(len(y), device='cuda')
        for indices in order.split(64):
            opt.zero_grad(set_to_none=True)
            loss = criterion(model.head(x[indices]), y[indices])
            loss.backward()
            opt.step()
        model.head.eval()
        with torch.no_grad():
            scores = model.head(vx).cpu().numpy()
        with np.load(p['baseline_validation_logits']) as bank:
            pairs = pd.DataFrame(p['pairs'])
            pred = composed(bank['crop'], scores, pairs).argmax(1)
            reference = composed(bank['crop'], bank['disease'], pairs).argmax(1)
        rows = roles['validation_seen'] + roles['validation_unseen']
        targets = np.array([r['composition_id'] for r in rows])
        seen = np.arange(len(rows)) < len(roles['validation_seen'])

        metrics = {}
        for name, mask in [('seen', seen), ('pseudo_unseen', ~seen),
                           ('generated_pairs', augmented), ('same_disease_other_crop', transferred), ('transfer_heldout', transferred & ~seen)]:
            metrics[name] = {'images': int(mask.sum()), 'correct': int((pred[mask] == targets[mask]).sum()),
                             'baseline_correct': int((reference[mask] == targets[mask]).sum())}
        history.append({'epoch':epoch, **metrics})
        np.save(run/f'epoch_{epoch}_disease_logits.npy',scores)
        torch.save(model.head.state_dict(),run/f'head_epoch_{epoch}.pt')
        write_json(run/'history.json',history)
        status(run,'epoch_complete', epoch=epoch, metrics=metrics)
        model.head.train()
    status(run,'complete', epochs=3, composites=len(synthetic))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['leafgan','real_only'], required=True)
    main(parser.parse_args().arm)



