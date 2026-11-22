"""Immutable data roles, deterministic batches, metrics and resumable state."""
from __future__ import annotations

import hashlib
import csv
import time
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "Plant_Disease"))
from baseline.data import (audit, read_records, resolve_images, save_manifest,
                           sha256, validation_split)
from baseline.engine import transforms_for


ARMS = ("compositional",)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


_progress_times={}

def status(run, phase, **details):
    value = {"updated_utc": stamp(), "phase": phase, "pid": os.getpid(), **details}
    write_json(run / "status.json", value)
    print(json.dumps(value), flush=True)
    key=str(run)
    previous=_progress_times.get(key,0)
    now=time.monotonic()
    if now-previous >= 900 or phase in ('complete','failed'):
        summaries={}
        for arm in ARMS:
            path=run/arm/'summary.json'
            if path.exists(): summaries[arm]=json.loads(path.read_text())
        snapshot={**value,'models':summaries}
        write_json(run/'progress_15min.json',snapshot)
        with (run/'progress_15min.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(snapshot)+'\n')
        _progress_times[key]=now


class StudyLock:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.stream.close()
            self.stream = None
            raise RuntimeError("Another controller holds the study lock")
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for key in ("data_dir", "pretrained_weights", "baseline_run"):
        cfg[key] = str((HERE / cfg[key]).resolve())
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def source_hashes():
    files = sorted(HERE.glob("*.py")) + [HERE / "config.json", HERE / "PLAN.md"]
    files += sorted((ROOT / "Plant_Disease" / "baseline").glob("*.py"))
    return {str(p): sha256(p) for p in files}


def historical_hashes():
    base = ROOT / "Plant_Disease"
    paths = [base / "BASELINE_NOTES.md", base / "config.json"]
    for directory in ("independent_convnext_tiny_22k_seed42", "convnext_tiny_22k_tuning_seed42/evaluation"):
        for item in ("crop/best.pt", "disease/best.pt", "results.md", "metrics.json"):
            paths.append(base / "runs" / directory / item)
    original=base/'runs/independent_convnext_tiny_22k_seed42'
    paths += [original/'predictions_seen.csv',original/'predictions_unseen.csv']
    return {str(p): sha256(p) for p in paths}


def prepare(cfg, run):
    saved_path=run/'prepared.json'
    if saved_path.exists():
        saved=json.loads(saved_path.read_text())
        if saved['config']!=cfg or saved['source_hashes']!=source_hashes():
            raise ValueError('Source/config changed since preparation')
        for path,h in {**saved['csv_hashes'],**saved['manifest_hashes']}.items():
            if sha256(path)!=h: raise ValueError('Metadata changed: '+path)
        return saved
    status(run,'auditing_images')
    data=Path(cfg['data_dir']); baseline=Path(cfg['baseline_run'])
    csvs={k:data/f'DPD_Small_{v}.csv' for k,v in (('train','train'),('seen','test_seen'),('unseen','test_unseen'))}
    splits={k:resolve_images(read_records(p),data/'DPD_Small_dataset') for k,p in csvs.items()}
    audit_report,hashes=audit(splits,verify_images=True,hash_images=True)
    by_name={r.filename:r for r in splits['train']}
    manifests={role:baseline/name for role,name in (('fit','fit.csv'),('seen_validation','validation.csv'))}
    roles={}
    for role,path in manifests.items():
        with path.open(encoding='utf-8-sig',newline='') as f: items=list(csv.DictReader(f))
        rows=[]
        for item in items:
            r=by_name[item['filename']]
            for key in ('crop_id','disease_id','composition_id'):
                if getattr(r,key)!=int(item[key]): raise ValueError('Manifest label mismatch')
            rows.append(r)
        roles[role]=rows
    if len(roles['fit'])!=6746 or len(roles['seen_validation'])!=1254: raise ValueError('Original split sizes changed')
    a={r.filename for r in roles['fit']}; b={r.filename for r in roles['seen_validation']}
    if a&b or a|b!=set(by_name): raise ValueError('Original manifest partition mismatch')
    if {hashes[r.path] for r in roles['fit']} & {hashes[r.path] for r in roles['seen_validation']}:
        raise ValueError('Duplicate-content leakage')
    roles.update(official_seen=splits['seen'],official_unseen=splits['unseen'])
    candidates={r.composition_id:{k:getattr(r,k) for k in ('composition_id','composition_name','crop_id','disease_id')} for rows in splits.values() for r in rows}
    ontology={'crop':sorted({r.crop_id for r in splits['train']}),'disease':sorted({r.disease_id for r in splits['train']}),
              'candidates':[candidates[k] for k in sorted(candidates)],
              'healthy_disease_id':next(r.disease_id for r in splits['train'] if r.disease_name=='healthy')}
    if len(candidates)!=333: raise ValueError('Ontology changed')
    summary={k:{'images':len(v),'compositions':len({r.composition_id for r in v})} for k,v in roles.items()}
    for k,v in roles.items(): save_manifest(run/f'{k}.csv',v)
    crop_path=baseline/'crop/best.pt'
    value={'prepared_utc':stamp(),'config':cfg,'source_hashes':source_hashes(),
        'csv_hashes':{str(p):sha256(p) for p in csvs.values()},'manifest_hashes':{str(p):sha256(p) for p in manifests.values()},
        'pretrained_sha256':sha256(cfg['pretrained_weights']),'historical_hashes':historical_hashes(),
        'crop_checkpoint':str(crop_path),'crop_checkpoint_sha256':sha256(crop_path),'audit_report':audit_report,
        'ontology':ontology,'roles':{k:[asdict(r) for r in v] for k,v in roles.items()},'image_hashes':hashes,'role_summary':summary}
    value['fingerprint']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    for path in list(HERE.glob('*.py'))+[HERE/'config.json',HERE/'PLAN.md']:
        (run/'source').mkdir(exist_ok=True); shutil.copy2(path,run/'source'/path.name)
    write_json(saved_path,value); write_json(run/'role_summary.json',summary)
    status(run,'prepared',roles=summary)
    return value

class ImageRows(Dataset):
    def __init__(self, rows, transform, ontology):
        self.rows, self.transform = rows, transform
        self.crop_map = {v: i for i, v in enumerate(ontology["crop"])}
        self.disease_map = {v: i for i, v in enumerate(ontology["disease"])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = self.transform(image.convert("RGB"))
        return image, torch.tensor([self.crop_map[row["crop_id"]],
                                    self.disease_map[row["disease_id"]], row["composition_id"]])


class EpochBatches(Sampler):
    def __init__(self,rows,cfg,grouped=False):
        if grouped: raise ValueError('Only the original full-coverage shuffle policy is allowed')
        self.rows,self.cfg,self.epoch=rows,cfg,0
    def __len__(self): return math.ceil(len(self.rows)/self.cfg['batch_size'])
    def indices(self):
        return np.random.default_rng(self.cfg['seed']+1000+self.epoch).permutation(len(self.rows)).tolist()
    def __iter__(self):
        indices=self.indices(); size=self.cfg['batch_size']
        return iter([indices[i:i+size] for i in range(0,len(indices),size)])


def worker_seed(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def loader(rows, cfg, ontology, transform, sampler=None, epoch=0):
    generator = torch.Generator().manual_seed(cfg["seed"] + 50000 + epoch)
    kw = {"num_workers": cfg["workers"], "pin_memory": True, "worker_init_fn": worker_seed,
          "generator": generator, "persistent_workers": False}
    data = ImageRows(rows, transform, ontology)
    if sampler is not None:
        return DataLoader(data, batch_sampler=sampler, **kw)
    return DataLoader(data, batch_size=cfg["batch_size"], shuffle=False, **kw)


def harmonic(s, u):
    return 2*s*u/(s+u) if s+u else 0.0


def classification_metrics(scores, targets):
    ranking = scores.argsort(dim=1, descending=True, stable=True)
    hits = ranking.eq(targets[:, None])
    top1, top5 = hits[:, 0], hits[:, :min(5, scores.shape[1])].any(1)
    return {"top1": float(top1.double().mean()), "top5": float(top5.double().mean()),
            "macro_top1": float(torch.stack([top1[targets == c].double().mean() for c in targets.unique()]).mean())}


def metrics(crop, disease, rows, ontology, pair):
    cm = {v: i for i, v in enumerate(ontology["crop"])}
    dm = {v: i for i, v in enumerate(ontology["disease"])}
    pm = {c["composition_id"]: i for i, c in enumerate(ontology["candidates"])}
    cy = torch.tensor([cm[r["crop_id"]] for r in rows])
    dy = torch.tensor([dm[r["disease_id"]] for r in rows])
    py = torch.tensor([pm[r["composition_id"]] for r in rows])
    scores = pair.double()
    ranking = scores.argsort(dim=1, descending=True, stable=True)
    correct = ranking[:, 0].eq(py)
    per_comp = {str(c): {"images": int(sum(r["composition_id"] == c for r in rows)),
                         "correct": int(correct[torch.tensor([r["composition_id"] == c for r in rows])].sum())}
                for c in sorted({r["composition_id"] for r in rows})}
    return {"images": len(rows), "crop": classification_metrics(crop, cy),
            "disease": classification_metrics(disease, dy),
            "composition": classification_metrics(scores, py), "per_composition": per_comp}


def paired_bootstrap(a, b, rows, cfg, cluster):
    rng = np.random.default_rng(cfg["split_seed"] + 100 + 37*len(rows))
    group_ids = np.array([r["composition_id"] for r in rows]) if cluster else np.arange(len(rows))
    groups = np.unique(group_ids)
    sums_a = np.array([a[group_ids == g].sum() for g in groups])
    sums_b = np.array([b[group_ids == g].sum() for g in groups])
    counts = np.array([(group_ids == g).sum() for g in groups])
    indices = rng.integers(0, len(groups), size=(cfg["bootstrap_resamples"], len(groups)))
    return sums_a[indices].sum(1)/counts[indices].sum(1), sums_b[indices].sum(1)/counts[indices].sum(1)
