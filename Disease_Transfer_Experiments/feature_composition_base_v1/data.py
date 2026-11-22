"""Deterministic real batches and fitting-only donors for feature composition."""
from collections import defaultdict
import math

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from protocol import worker_seed, transforms_for
from storage import IntegrityError


def synthesis_plan(rows, pairs):
    by_crop, by_disease = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        by_crop[row['crop_id']].append(i); by_disease[row['disease_id']].append(i)
    fitting = {r['composition_id'] for r in rows}
    novel, familiar = [], []
    for pair in pairs:
        if pair['is_unseen']: continue  # The ten official unseen classes are outside development.
        c, d = pair['crop_id'], pair['disease_id']
        if any(rows[i]['disease_id'] != d for i in by_crop[c]) and any(rows[i]['crop_id'] != c for i in by_disease[d]):
            (familiar if pair['pair_id'] in fitting else novel).append(pair['pair_id'])
    if not novel or not familiar: raise IntegrityError('No eligible novel/familiar feature compositions')
    return {'novel_pair_ids': novel, 'familiar_pair_ids': familiar,
            'fitting_pair_ids': sorted(fitting), 'donor_images': 'fitting partition only',
            'class_metadata': 'known candidate identities, without validation image features'}


def balanced_stream(ids, count, rng):
    return np.concatenate([rng.permutation(ids) for _ in range(math.ceil(count / len(ids)))])[:count]


def epoch_schedule(rows, pairs, cfg, epoch):
    plan = synthesis_plan(rows, pairs)
    size, synthetic, novel_count = cfg['finetune_batch'], cfg['synthetic_batch'], cfg['novel_synthetic_batch']
    if not 0 < novel_count < synthetic: raise IntegrityError('Require novel and familiar synthetic targets')
    primary = np.random.default_rng(cfg['seed'] + 2000 + epoch).permutation(len(rows))
    steps = math.ceil(len(rows) / size)
    rng = np.random.default_rng(cfg['seed'] + 470000 + epoch)
    novel = balanced_stream(plan['novel_pair_ids'], steps * novel_count, rng).reshape(steps, novel_count)
    familiar = balanced_stream(plan['familiar_pair_ids'], steps * (synthetic - novel_count), rng).reshape(steps, synthetic - novel_count)
    by_crop, by_disease = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        by_crop[row['crop_id']].append(i); by_disease[row['disease_id']].append(i)
    pairs_by_id = {p['pair_id']: p for p in pairs}
    crop_pools, disease_pools = {}, {}
    for pid in plan['novel_pair_ids'] + plan['familiar_pair_ids']:
        c, d = pairs_by_id[pid]['crop_id'], pairs_by_id[pid]['disease_id']
        crop_pools[pid] = [i for i in by_crop[c] if rows[i]['disease_id'] != d]
        disease_pools[pid] = [i for i in by_disease[d] if rows[i]['crop_id'] != c]
    schedule = []
    for step in range(steps):
        real = primary[step * size:(step + 1) * size]
        targets = np.concatenate([novel[step], familiar[step]])
        crop_source = np.array([rng.choice(crop_pools[int(pid)]) for pid in targets], dtype=np.int64)
        disease_source = np.array([rng.choice(disease_pools[int(pid)]) for pid in targets], dtype=np.int64)
        unique = list(real); position = {int(i): j for j, i in enumerate(unique)}
        for i in np.concatenate([crop_source, disease_source]):
            if int(i) not in position: position[int(i)] = len(unique); unique.append(int(i))
        unique = np.asarray(unique, dtype=np.int64)
        seeds = [cfg['seed'] + 10000000 + epoch * 100000 + int(i) if j < len(real)
                 else cfg['seed'] + 20000000 + epoch * 10000000 + step * 10000 + int(i)
                 for j, i in enumerate(unique)]
        schedule.append({'primary_indices': real, 'unique_indices': unique,
            'keys': list(zip(unique.tolist(), seeds)), 'crop_donor_indices': crop_source,
            'disease_donor_indices': disease_source,
            'crop_positions': np.array([position[int(i)] for i in crop_source]),
            'disease_positions': np.array([position[int(i)] for i in disease_source]),
            'target_pair': targets, 'target_crop': np.array([pairs_by_id[int(i)]['crop_id'] for i in targets]),
            'target_disease': np.array([pairs_by_id[int(i)]['disease_id'] for i in targets]),
            'novel': np.array([True] * novel_count + [False] * (synthetic - novel_count)),
            'primary_crop': np.array([rows[int(i)]['crop_id'] for i in real]),
            'primary_disease': np.array([rows[int(i)]['disease_id'] for i in real])})
    return schedule


class FittingImages(Dataset):
    def __init__(self, rows): self.rows, self.transform = rows, transforms_for(None, 224)[0]
    def __len__(self): return len(self.rows)
    def __getitem__(self, key):
        index, seed = key
        with Image.open(self.rows[index]['path']) as im: image = im.convert('RGB')
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed); image = self.transform(image)
        return image, index


class EvaluationImages(Dataset):
    def __init__(self, rows): self.rows, self.transform = rows, transforms_for(None, 224)[1]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        with Image.open(self.rows[index]['path']) as im: return self.transform(im.convert('RGB'))


def training_loader(rows, schedule, cfg, epoch, cursor):
    generator = torch.Generator().manual_seed(cfg['seed'] + 50000 + epoch)
    return DataLoader(FittingImages(rows), batch_sampler=[s['keys'] for s in schedule[cursor:]],
        num_workers=cfg['workers'], pin_memory=True, worker_init_fn=worker_seed,
        generator=generator, persistent_workers=False)


def batch_metadata(step, device):
    keys = ('crop_positions', 'disease_positions', 'target_crop', 'target_disease', 'primary_crop', 'primary_disease')
    return {**{key: torch.as_tensor(step[key], dtype=torch.long, device=device) for key in keys},
            'primary_count': len(step['primary_indices'])}


def exposure_arrays(schedule, count, candidates):
    primary = np.concatenate([s['primary_indices'] for s in schedule])
    encoded = np.concatenate([s['unique_indices'] for s in schedule])
    target = np.stack([s['target_pair'] for s in schedule])
    return {'primary_indices': primary, 'primary_counts': np.bincount(primary, minlength=count),
        'encoded_counts': np.bincount(encoded, minlength=count),
        'crop_donor_indices': np.stack([s['crop_donor_indices'] for s in schedule]),
        'disease_donor_indices': np.stack([s['disease_donor_indices'] for s in schedule]),
        'synthetic_pair_ids': target, 'synthetic_pair_counts': np.bincount(target.ravel(), minlength=candidates),
        'novel': np.stack([s['novel'] for s in schedule])}
