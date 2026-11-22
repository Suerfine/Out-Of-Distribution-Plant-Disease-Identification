"""Fine-tune the full saved development model with lesion copy-paste."""
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
from independent_cnn_baseline.models import IndependentNet, transforms, optimizer
from storage import CheckpointStore
import hashlib


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


def main():
    run = HERE/'full_run'
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
    synthetic = json.loads((HERE/'composites.json').read_text())
    fit_names = {row['filename'] for row in roles['fit']}
    if any(row['donor'] not in fit_names or row['recipient'] not in fit_names for row in synthetic):
        raise RuntimeError('Composite uses an image outside fitting')
    train_transform, transform = transforms(base)
    pair_by_factors = {(r['crop_id'], r['disease_id']): r['pair_id'] for r in p['pairs']}
    synthetic_ids = {pair_by_factors[(r['crop_id'], r['disease_id'])] for r in synthetic}
    holdout_ids = {r['composition_id'] for r in roles['validation_unseen']}
    augmented_holdout = synthetic_ids & holdout_ids
    assert len(synthetic_ids) == 9 and len(augmented_holdout) == 3
    fit_rows = roles['fit'] + synthetic
    fit = DataLoader(Images(fit_rows, train_transform), batch_size=4, shuffle=True, num_workers=2)
    val = DataLoader(Images(roles['validation_seen']+roles['validation_unseen'], transform),
                     batch_size=8, num_workers=2)
    model.backbone.set_grad_checkpointing(True)
    opt = optimizer(model, base)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=.1)
    manifest = HERE/'composites.json'
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()+manifest.read_bytes()+
                                 bytes.fromhex(p['reference_hashes'][model_path])).hexdigest()
    store = CheckpointStore(run, fingerprint, 6*1024**3)
    saved = store.load()
    history = []
    if saved is not None:
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        history = saved['history']
    for epoch in range(len(history)+1,16):
        seed_all(42+epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        status(run,'training',epoch=epoch,step=0,total_steps=len(fit))
        for step,(images,labels) in enumerate(fit,1):
            start = ((step-1)//8)*32
            samples = min(32,len(fit_rows)-start)
            with torch.autocast('cuda',dtype=torch.float16):
                loss = criterion(model(images.cuda()),labels.cuda())*len(labels)/samples
            scaler.scale(loss).backward()
            if step%8 == 0 or step == len(fit):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if step%100 == 0:
                status(run,'training',epoch=epoch,step=step,total_steps=len(fit))
        model.eval()
        status(run,'validation',epoch=epoch)
        parts = []
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
            for images,_ in val:
                parts.append(model(images.cuda()).float().cpu().numpy())
        scores = np.concatenate(parts)
        with np.load(p['baseline_validation_logits']) as bank:
            pairs = pd.DataFrame(p['pairs'])
            pred = composed(bank['crop'], scores, pairs).argmax(1)
            reference = composed(bank['crop'], bank['disease'], pairs).argmax(1)
        rows = roles['validation_seen'] + roles['validation_unseen']
        targets = np.array([r['composition_id'] for r in rows])
        seen = np.arange(len(rows)) < len(roles['validation_seen'])
        augmented = ~seen & np.isin(targets, list(augmented_holdout))
        metrics = {}
        for name, mask in [('seen', seen), ('pseudo_unseen', ~seen),
                           ('augmented_holdout', augmented), ('untouched_holdout', ~seen & ~augmented)]:
            metrics[name] = {'images': int(mask.sum()), 'correct': int((pred[mask] == targets[mask]).sum()),
                             'baseline_correct': int((reference[mask] == targets[mask]).sum())}
        for pair_id in sorted(synthetic_ids):
            mask = targets == pair_id
            metrics[f'pair_{pair_id}'] = {'images': int(mask.sum()),
                                         'correct': int((pred[mask] == targets[mask]).sum()),
                                         'baseline_correct': int((reference[mask] == targets[mask]).sum())}
        history.append({'epoch':epoch, **metrics})
        np.save(run/f'epoch_{epoch}_disease_logits.npy',scores)
        s = metrics['seen']['correct']/metrics['seen']['images']
        u = metrics['pseudo_unseen']['correct']/metrics['pseudo_unseen']['images']
        history[-1]['harmonic_mean'] = 2*s*u/(s+u) if s+u else 0
        if history[-1]['harmonic_mean'] > max((r['harmonic_mean'] for r in history[:-1]),default=-1):
            from storage import checked_save
            checked_save(run/'best.pt',{'model':model.state_dict(),'epoch':epoch},6*1024**3,run/'checkpoint_events.jsonl')
        store.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
                    'history':history,'epoch':epoch,'cursor':0,'stage':'finetune','fingerprint':fingerprint})
        write_json(run/'history.json',history)
        status(run,'epoch_complete', epoch=epoch, metrics=metrics)
        if (run/"STOP.request").exists():
            status(run,"paused",epoch=epoch)
            return
    status(run,'complete', epochs=15, composites=len(synthetic))


if __name__ == '__main__':
    main()




