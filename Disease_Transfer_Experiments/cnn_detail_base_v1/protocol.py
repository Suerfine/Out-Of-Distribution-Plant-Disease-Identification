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

ARMS = ('resolution384', 'global_local')


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
    for key in ('data_dir', 'baseline_run', 'source_development_run'):
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


def arm_fingerprint(p, arm):
    if arm not in ARMS: raise IntegrityError('Unrecognized CNN arm')
    return hashlib.sha256((p['fingerprint'] + '|' + arm).encode()).hexdigest()


def prepare(cfg, root):
    saved = root / 'prepared.json'
    if saved.exists():
        p = json.loads(saved.read_text())
        if p['config'] != cfg or p['source_hashes'] != source_hashes(): raise IntegrityError('Prepared source/config changed')
        for path,digest in p['reference_hashes'].items():
            if sha256(path) != digest: raise IntegrityError('Changed reference: ' + path)
        return p
    status(root, 'auditing_original_inputs')
    source_path = Path(cfg['source_development_run']) / 'prepared.json'
    source = json.loads(source_path.read_text())
    study = BASE_PROJECT / 'runs/baseline/independent_cnn_baseline_study_seed42'
    frames = {k:read_rows(study / f'{k}.csv') for k in ('fit','validation_seen','validation_unseen','selection_excluded','all_train')}
    counts = validate_partition(frames)
    roles = {role:[dict(row) for row in source['roles'][role]] for role in ('fit','validation_seen','validation_unseen')}
    for role,rows in roles.items():
        if [r['filename'] for r in rows] != [r['file'] for r in frames[role]]: raise IntegrityError('Manifest order differs')
        for row,original in zip(rows,frames[role]):
            if tuple(row[k] for k in ('crop_id','disease_id','composition_id')) != tuple(int(original[k]) for k in ('crop_id','disease_id','pair_id')):
                raise IntegrityError('Manifest label differs')
    original_inventory = HERE.parent / 'simsiam_base_v1/runs/overnight_seed42/prepared.json'
    inventory = json.loads(original_inventory.read_text())
    rows = [r for values in roles.values() for r in values]
    if len({Path(r['filename']).stem for r in rows}) != len(rows): raise IntegrityError('Cache filename collision')
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data import cache_original
    cache = root / 'inputs_512'; cache.mkdir(exist_ok=True)
    cache_hashes, image_hashes = {}, {}
    def build(row):
        original = Path(row['original_path']); digest = sha256(original)
        if digest != inventory['image_hashes'][str(original)]: raise IntegrityError('Original image changed: ' + str(original))
        destination = cache / (Path(row['filename']).stem + '.jpg')
        metadata = cache_original(original,destination,cfg['cache_short_side'])
        return row,str(destination),sha256(destination),digest,metadata
    cache_records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(build,row) for row in rows]
        for count,future in enumerate(as_completed(futures),1):
            row,destination,digest,original_digest,metadata = future.result()
            row['legacy_cache_path'], row['path'] = row['path'], destination
            cache_hashes[destination] = digest; image_hashes[row['original_path']] = original_digest
            cache_records.append({'filename':row['filename'],'cache_path':destination,'sha256':digest,
                                  'original_sha256':original_digest,**metadata})
            if count % 128 == 0 or count == len(rows): status(root,'building_512_cache',images=count,total_images=len(rows))
    for role in ('validation_seen','validation_unseen'):
        fitting={cache_hashes[r['path']] for r in roles['fit']}
        if fitting & {cache_hashes[r['path']] for r in roles[role]}: raise IntegrityError('Exact cached fit/validation content overlap')
    base = source['baseline_config']
    pretrained = Path(base['model']['pretrained_file'])
    if base['model']['backbone'] != cfg['model'] or sha256(pretrained) != base['model']['pretrained_sha256']:
        raise IntegrityError('Pinned Base initialization differs')
    references = dict(source['reference_hashes'])
    references[str(source_path)] = sha256(source_path)
    references[str(original_inventory)] = sha256(original_inventory)
    for name in frames:
        path=study/f'{name}.csv';shutil.copyfile(path,root/path.name);references[str(path)]=sha256(path)
    for path,digest in references.items():
        if sha256(path)!=digest: raise IntegrityError('Original reference changed: '+path)
    write_json(root/'cache_manifest.json',sorted(cache_records,key=lambda r:r['filename']))
    p={'prepared_utc':stamp(),'config':cfg,'source_hashes':source_hashes(),'reference_hashes':references,
       'image_hashes':image_hashes,'cache_hashes':cache_hashes,'roles':roles,'pairs':source['pairs'],
       'baseline_config':base,'baseline_validation_logits':source['baseline_validation_logits'],
       'crop_checkpoint':source['crop_checkpoint'],'role_summary':source['role_summary'],
       'original_partition_counts':counts,'initialization':'same pinned ImageNet Base; fresh disease head; no plant SSL or warm start',
       'pretrained_path':str(pretrained),'pretrained_sha256':sha256(pretrained),'official_test_images_loaded':False,
       'detail_input':{'cache_short_side':512,'global_size':384,'eval_center_canvas':448,'local_patch_size':224,'local_patches':4},
       'selection':'gamma0 validation composition Top1 H; earliest exact tie; max30/patience8 independently per arm'}
    from importlib.metadata import version
    p['environment']={'python':sys.version,**{name:version(name) for name in ('torch','torchvision','timm','Pillow')}}
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    from evaluation import validation_metrics
    with np.load(p['baseline_validation_logits']) as bank: reference=validation_metrics(bank['disease'],p)
    if abs(reference['objective']['harmonic_mean_top1']-.0857505367497348)>1e-12: raise IntegrityError('Baseline metric reproduction differs')
    write_json(root/'baseline_validation_reference.json',reference)
    for name in cfg['arms']:
        (root/name).mkdir(exist_ok=True)
        write_json(root/name/'queued.json',{'queued_utc':stamp(),'arm':name,'study_fingerprint':p['fingerprint']})
    (root/'source').mkdir(exist_ok=True)
    for f in sorted(HERE.glob('*.py'))+[HERE/'config.json',HERE/'PLAN.md']:shutil.copyfile(f,root/'source'/f.name)
    write_json(saved,p);write_json(root/'role_summary.json',p['role_summary'])
    status(root,'prepared',roles=p['role_summary'],arms=cfg['arms'],original_image_cache=True)
    return p


