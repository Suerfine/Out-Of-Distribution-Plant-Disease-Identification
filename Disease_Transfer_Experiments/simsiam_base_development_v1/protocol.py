"""Exact existing Base development roles; official images are not loaded."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_PROJECT = ROOT / 'Out-Of-Distribution-Plant-Disease-Identification'
sys.path.insert(0, str(BASE_PROJECT))
from independent_cnn_baseline.models import transforms as base_transforms
from storage import atomic_json, sha as sha256, IntegrityError

ARMS = ('ssl', 'finetune')


def stamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


write_json = atomic_json
_progress_times = {}


def status(run, phase, **details):
    value = {'updated_utc': stamp(), 'phase': phase, 'pid': os.getpid(), **details}
    write_json(run / 'status.json', value)
    print(json.dumps(value), flush=True)
    now = time.monotonic()
    if now - _progress_times.get(str(run), 0) >= 900 or phase in ('complete', 'failed', 'scientific_failure', 'integrity_failure'):
        summaries = {}
        for arm in ARMS:
            path = run / arm / 'summary.json'
            if path.exists(): summaries[arm] = json.loads(path.read_text())
        snapshot = {**value, 'models': summaries}
        write_json(run / 'progress_15min.json', snapshot)
        with (run / 'progress_15min.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(snapshot) + '\n')
        _progress_times[str(run)] = now


class StudyLock:
    def __init__(self, path): self.path, self.stream = Path(path), None
    def __enter__(self):
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.stream.write(b'0'); self.stream.flush()
        self.stream.seek(0)
        try: msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.stream.close(); self.stream = None
            raise RuntimeError('Another worker holds this study lock')
        return self
    def __exit__(self, *args):
        if self.stream:
            import msvcrt
            self.stream.seek(0); msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for key in ('data_dir', 'baseline_run', 'development_study', 'development_trial', 'inventory_run'):
        cfg[key] = str((HERE / cfg[key]).resolve())
    return cfg


def transforms_for(model, size):
    cfg = json.loads((BASE_PROJECT / 'runs/baseline/independent_cnn_baseline_convnext_base_seed42/config.json').read_text())
    a, b = base_transforms(cfg)
    return a, b, cfg['input']


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed); np.random.seed(seed); torch.set_num_threads(1)


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), *state[2:]),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def source_hashes():
    paths = sorted(HERE.glob('*.py')) + [HERE / 'config.json', HERE / 'PLAN.md']
    paths += [BASE_PROJECT / name for name in (
        'independent_cnn_baseline/models.py', 'independent_cnn_baseline/engine.py',
        'independent_cnn_baseline/uncalibrated_objective.py', 'dpd/cnn.py', 'dpd/evaluate.py')]
    return {str(f): sha256(f) for f in paths}


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def validate_partition(frames):
    roles = ('fit', 'validation_seen', 'validation_unseen', 'selection_excluded')
    counts = {name: len(frames[name]) for name in (*roles, 'all_train')}
    if counts != dict(fit=5798, validation_seen=862, validation_unseen=533, selection_excluded=807, all_train=8000):
        raise IntegrityError('Unexpected original Base partition sizes: ' + str(counts))
    sets = {k: {r['file'] for r in rows} for k, rows in frames.items()}
    for i, a in enumerate(roles):
        if len(sets[a]) != len(frames[a]): raise IntegrityError('Duplicate manifest image')
        for b in roles[i + 1:]:
            if sets[a] & sets[b]: raise IntegrityError('Development role overlap')
    if set.union(*(sets[k] for k in roles)) != sets['all_train']:
        raise IntegrityError('Development roles do not partition all training images')
    pseudo = {int(r['pair_id']) for r in frames['validation_unseen']}
    fitting = {int(r['pair_id']) for r in frames['fit']}
    if len(pseudo) != 20 or pseudo & fitting: raise IntegrityError('Held-out combination leaked into fitting')
    if {int(r['pair_id']) for r in frames['validation_seen']} - fitting:
        raise IntegrityError('Seen validation includes a nonfitting combination')
    if {int(r['pair_id']) for r in frames['selection_excluded']} - pseudo:
        raise IntegrityError('Unexpected capped-out combinations')
    if {int(r['crop_id']) for r in frames['fit']} != set(range(55)):
        raise IntegrityError('Missing fitting crop')
    if {int(r['disease_id']) for r in frames['fit']} != set(range(175)):
        raise IntegrityError('Missing fitting disease')
    return counts


def prepare(cfg, run):
    saved = run / 'prepared.json'
    if saved.exists():
        p = json.loads(saved.read_text())
        if p['config'] != cfg or p['source_hashes'] != source_hashes():
            raise IntegrityError('Prepared source/config has changed')
        for f, digest in p['reference_hashes'].items():
            if sha256(f) != digest: raise IntegrityError('Changed development input: ' + f)
        return p
    status(run, 'auditing_development_inputs')
    study, trial = Path(cfg['development_study']), Path(cfg['development_trial'])
    frames = {k: read_rows(study / f'{k}.csv') for k in ('fit', 'validation_seen', 'validation_unseen', 'selection_excluded', 'all_train')}
    counts = validate_partition(frames)
    inventory_path = Path(cfg['inventory_run']) / 'prepared.json'
    inventory = json.loads(inventory_path.read_text())
    by_file = {r['filename']: r for r in inventory['roles']['fit']}
    if set(by_file) != {r['file'] for r in frames['all_train']}:
        raise IntegrityError('Training inventory membership changed')
    mapped = {}
    for role in ('fit', 'validation_seen', 'validation_unseen'):
        rows = []
        for r in frames[role]:
            item = dict(by_file[r['file']])
            if tuple(int(r[k]) for k in ('crop_id', 'disease_id', 'pair_id')) != tuple(item[k] for k in ('crop_id', 'disease_id', 'composition_id')):
                raise IntegrityError('Training inventory label mismatch')
            rows.append(item)
        mapped[role] = rows
    official_names = {r['filename'] for role in ('official_seen', 'official_unseen') for r in inventory['roles'][role]}
    if official_names & set(by_file): raise IntegrityError('Official image appears in development training inventory')
    pairs = json.loads((study / 'pairs.json').read_text())
    if [r['pair_id'] for r in pairs] != list(range(333)) or sum(r['is_unseen'] for r in pairs) != 10:
        raise IntegrityError('Unexpected original candidate ontology')
    official_pairs = {r['pair_id'] for r in pairs if r['is_unseen']}
    if official_pairs & {int(r['pair_id']) for r in frames['all_train']}:
        raise IntegrityError('Official unseen combination occurs in original training manifest')
    candidates = [{'composition_id': r['pair_id'], 'composition_name': r['pair'],
                   'crop_id': r['crop_id'], 'disease_id': r['disease_id']} for r in pairs]
    base = json.loads((trial / 'config.json').read_text())
    base['model']['pretrained_file'] = str((BASE_PROJECT / base['model']['pretrained_file']).resolve())
    pretrained = Path(base['model']['pretrained_file'])
    if sha256(pretrained) != base['model']['pretrained_sha256']:
        raise IntegrityError('Pinned ImageNet weights changed')
    from concurrent.futures import ThreadPoolExecutor
    paths = [r['path'] for rows in mapped.values() for r in rows]
    with ThreadPoolExecutor(max_workers=4) as pool:
        cache_hashes = dict(zip(paths, pool.map(sha256, paths)))
    if any(inventory['cache_hashes'][path] != digest for path, digest in cache_hashes.items()):
        raise IntegrityError('A cached development image changed')
    grouped_hashes = {role: {cache_hashes[r['path']] for r in rows} for role, rows in mapped.items()}
    for role in ('validation_seen', 'validation_unseen'):
        if grouped_hashes['fit'] & grouped_hashes[role]:
            raise IntegrityError('Exact cached input overlap between fit and validation')
    references = [study / f'{k}.csv' for k in frames] + [study / 'pairs.json', inventory_path,
        trial / 'config.json', trial / 'identity.json', trial / 'crop/best_epoch_15.pt',
        trial / 'disease/best_epoch_15.pt', trial / 'validation_epoch_15.npz',
        trial.parent.parent / 'selected_recipe.json', pretrained]
    for role in frames:
        shutil.copyfile(study / f'{role}.csv', run / f'{role}.csv')
    p = {'prepared_utc': stamp(), 'config': cfg, 'source_hashes': source_hashes(),
        'reference_hashes': {str(f): sha256(f) for f in references}, 'cache_hashes': cache_hashes,
        'roles': mapped, 'role_summary': {k: {'images': len(v)} for k, v in mapped.items()},
        'original_partition_counts': counts, 'pairs': pairs,
        'ontology': {'crop': list(range(55)), 'disease': list(range(175)), 'candidates': candidates},
        'baseline_config': base, 'pretrained_path': str(pretrained), 'pretrained_sha256': sha256(pretrained),
        'crop_checkpoint': str(trial / 'crop/best_epoch_15.pt'),
        'baseline_validation_logits': str(trial / 'validation_epoch_15.npz'),
        'official_test_images_loaded': False, 'all_training_exposure': 'fit5798 only, including SSL',
        'selection': 'maximum gamma0 validation composition Top1 H; earliest exact tie; patience8; max30 supervised'}
    p['fingerprint'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    from evaluation import validation_metrics
    with np.load(p['baseline_validation_logits']) as bank:
        reference = validation_metrics(bank['disease'], p, bank['crop'])
    expected = json.loads((trial.parent.parent / 'selected_recipe.json').read_text())['validation']
    for key in ('seen_top1', 'unseen_top1', 'harmonic_mean_top1'):
        if abs(reference['objective'][key] - expected[key]) > 1e-12:
            raise IntegrityError('Saved development objective did not reproduce: ' + key)
    write_json(run / 'baseline_validation_reference.json', reference)
    (run / 'source').mkdir(exist_ok=True)
    for f in sorted(HERE.glob('*.py')) + [HERE / 'config.json', HERE / 'PLAN.md']:
        shutil.copyfile(f, run / 'source' / f.name)
    write_json(saved, p); write_json(run / 'role_summary.json', p['role_summary'])
    status(run, 'prepared', roles=p['role_summary'], baseline_retraining=False)
    return p
