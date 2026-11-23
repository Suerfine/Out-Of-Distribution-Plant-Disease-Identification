"""Write a vlm2 config that combines several single-factor changes on top of s0_base (stage 3 pairs, stage 4 final).

Usage:
  python -m train.vlm2_combine T1 C1                     # stage 3, validation-only: config/vlm2/s3_T1_C1_config.json
  python -m train.vlm2_combine T1 I1 C1 --final          # stage 4: s4_final + s4_final_r2 (full runs: refit + test)
  python -m train.vlm2_combine T1 C1 --name s3_custom    # any name

Each factor id (T1, I1, C1, C2, G1, ...) is looked up in config/vlm2/ as the config whose `_vlm2.factors` is exactly
[id]. Its changes relative to s0_base (every key that differs, including removed keys) are merged; two factors that set
the same key to different values are a conflict and stop the script. Documentation keys (_note, ...), names and the
validation.only / save_scores / train_seed_offset settings are not factor changes.

G1 (the GAN classifier) replaces the prompt classifier, so it cannot be merged into a LoRA config. With G1 among the
factors, the other factors become a full LoRA run (<name>_lora, or s0_base when G1 is alone) and the G1 config points
its features.run at that run; queue the LoRA run first.
"""

import argparse
import copy
import json
from pathlib import Path

from dpd.data import PROJECT_ROOT

CFG = PROJECT_ROOT / "config" / "vlm2"
IGNORED = {("validation", "only"), ("validation", "save_scores"), ("validation", "train_seed_offset")}
TOP_IGNORED = {"version", "title", "output_dir"}
MISSING = object()


def load(name):
    return json.loads((CFG / f"{name}_config.json").read_text(encoding="utf-8"))


def changes(cfg, base, path=()):
    """(path, value) for every leaf where cfg differs from base; value MISSING = key removed in cfg."""
    out = []
    for k in set(cfg) | set(base):
        if k.startswith("_") or (not path and k in TOP_IGNORED) or path + (k,) in IGNORED:
            continue
        a, b = cfg.get(k, MISSING), base.get(k, MISSING)
        if isinstance(a, dict) and isinstance(b, dict):
            out += changes(a, b, path + (k,))
        elif a != b:
            out.append((path + (k,), a))
    return out


def apply(cfg, path, value):
    d = cfg
    for k in path[:-1]:
        d = d.setdefault(k, {})
    if value is MISSING:
        d.pop(path[-1], None)
    else:
        d[path[-1]] = copy.deepcopy(value)


def factor_config(fid):
    hits = [p for p in CFG.glob("*_config.json") if json.loads(p.read_text(encoding="utf-8")).get("_vlm2", {}).get("factors") == [fid]]
    if len(hits) != 1:
        raise SystemExit(f"factor {fid}: expected one config with _vlm2.factors == ['{fid}'], found {[p.name for p in hits]}")
    return json.loads(hits[0].read_text(encoding="utf-8")), hits[0].name


def write(name, cfg, stage, factors, note, full, seed_offset=0):
    cfg = copy.deepcopy(cfg)
    cfg["validation"]["save_scores"] = True
    cfg["validation"].pop("only", None)
    if not full:
        cfg["validation"]["only"] = True
    if seed_offset:
        cfg["validation"]["train_seed_offset"] = seed_offset
    cfg.update(version=name, title=name, output_dir=f"runs/vlm2/{name}")
    head = {"_note": note, "_result": "", "_vlm2": {"stage": stage, "group": "combo", "factors": factors, "base": "s0_base"},
            "version": name, "title": name}
    body = {k: v for k, v in cfg.items() if k not in ("_note", "_result", "_vlm2", "version", "title")}
    path = CFG / f"{name}_config.json"
    path.write_text(json.dumps({**head, **body}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(PROJECT_ROOT)} ({'full run' if full else 'validation-only'})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("factors", nargs="+")
    parser.add_argument("--final", action="store_true", help="stage 4: full runs s4_final and its replicate s4_final_r2")
    parser.add_argument("--name")
    args = parser.parse_args()
    base = load("s0_base")
    lora_factors = [f for f in args.factors if f != "G1"]
    merged, owner = copy.deepcopy(base), {}
    for fid in lora_factors:
        cfg, fname = factor_config(fid)
        if cfg["method"] != base["method"]:
            raise SystemExit(f"{fid} ({fname}) is a {cfg['method']} config; only G1 may change the method")
        for path, value in changes(cfg, base):
            if path in owner and owner[path][1] != value:
                raise SystemExit(f"conflict on {'.'.join(path)}: {owner[path][0]} vs {fid}")
            owner[path] = (fid, value)
            apply(merged, path, value)

    stage = 4 if args.final else 3
    name = args.name or ("s4_final" if args.final else "s3_" + "_".join(args.factors))
    desc = " + ".join(args.factors)
    use_gan = "G1" in args.factors
    if lora_factors:
        lora_name = f"{name}_lora" if use_gan else name
        note = (f"{lora_name}: s0_base + {' + '.join(lora_factors)}, merged by train/vlm2_combine.py from the single-factor configs "
                f"({len(owner)} changed keys). " + ("Stage 4 final model." if args.final else "Stage 3 interaction run.") +
                (" Full run so the G1 config can use its adapters." if use_gan else ""))
        write(lora_name, merged, stage, lora_factors, note, full=args.final or use_gan)
        if args.final:
            write(f"{lora_name}_r2", merged, stage, lora_factors, note + " Replicate: training seeds + 100.", full=True, seed_offset=100)
    else:
        lora_name = "s0_base"
    if use_gan:
        gan, _ = factor_config("G1")
        gan["features"]["run"] = f"runs/vlm2/{lora_name}"
        gan["features"]["_note"] = f"Image features from {lora_name}'s LoRA encoder (retrained per validation split)."
        write(name, gan, stage, args.factors, f"{name}: G1 (feature-GAN classifier) on {lora_name}'s features ({desc}).", full=True)


if __name__ == "__main__":
    main()
