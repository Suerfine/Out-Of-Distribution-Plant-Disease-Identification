"""Build a config that combines every single-change ablation that improved on its base, judged on VALIDATION only.

Usage:  python -m train.combine_ablations --base v9 --variants v10 v12 v13 v14 --ensemble v11 --out v15 [--ensemble-out v16]

A variant counts as an improvement when its validation HM beats the base's by more than --margin (default 0.01, about
run-to-run noise). Test numbers are never read. Each variant config differs from the base only in its own settings,
so those differing settings are copied into the combined config; the combined config's _note records every variant
as included or excluded with its validation HM.
If the ensemble variant improved and --ensemble-out is given, an ensemble config (combined run + the ensemble's
other member) is written as well. Exits with code 2 if nothing improved (no config written).
"""

import argparse
import copy
import json
import shutil
import sys

from dpd.data import PROJECT_ROOT

CFG = PROJECT_ROOT / "config" / "vlm"
RUNS = PROJECT_ROOT / "runs" / "vlm"


def flat(d, prefix=()):
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            out.update(flat(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def set_path(d, path, value):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def val_hm(version):
    """Validation HM recorded by the run, or None if it did not finish."""
    m = RUNS / version / "metrics.json"
    if not m.exists():
        return None
    val = json.loads(m.read_text())["val"]
    return val.get("val_hm_mean", val.get("val_hm"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--ensemble")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ensemble-out")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--alias-if-identical", action="store_true",
                    help="if only one variant qualified, copy its finished run folder instead of retraining the same model")
    ap.add_argument("--force-ensemble", action="store_true",
                    help="write the ensemble config on top of --out even if the ensemble variant did not clear the margin")
    args = ap.parse_args()

    load = lambda v: json.loads((CFG / f"{v}_config.json").read_text(encoding="utf-8"))
    base = load(args.base)
    base_hm = val_hm(args.base)
    fb = flat(base)
    combined = copy.deepcopy(base)
    lines, included = [], []
    for v in args.variants:
        hm = val_hm(v)
        cfg = load(v)
        changed = {p: val for p, val in flat(cfg).items() if fb.get(p) != val and p not in (("version",), ("output_dir",))}
        desc = ", ".join(f"{'.'.join(p)}={val}" for p, val in changed.items())
        if hm is None:
            lines.append(f"{v} EXCLUDED (run did not finish) [{desc}]")
        elif hm > base_hm + args.margin:
            included.append(v)
            for p, val in changed.items():
                set_path(combined, p, val)
            for section in {p[0] for p in changed}:  # carry the variant's explanation for each section it changed
                if isinstance(cfg.get(section), dict) and "_note" in cfg[section]:
                    combined[section]["_note"] = f"From {v}: " + cfg[section]["_note"]
            lines.append(f"{v} INCLUDED: val HM {hm:.2%} vs {args.base} {base_hm:.2%} (+{(hm - base_hm) * 100:.1f} pts) [{desc}]")
        else:
            lines.append(f"{v} EXCLUDED: val HM {hm:.2%} vs {args.base} {base_hm:.2%} ({(hm - base_hm) * 100:+.1f} pts, "
                         f"needs > +{args.margin * 100:.0f}) [{desc}]")

    ens_hm = val_hm(args.ensemble) if args.ensemble else None
    ens_improved = ens_hm is not None and ens_hm > base_hm + args.margin
    if args.ensemble:
        lines.append(f"{args.ensemble} (ensemble, applied as a separate version on top of {args.out}): "
                     + ("not finished" if ens_hm is None else f"val HM {ens_hm:.2%} ({(ens_hm - base_hm) * 100:+.1f} pts) -> "
                        + ("improved" if ens_improved else "no improvement")))
    report = "\n".join(lines)
    print(report)
    if not included:
        print(f"\nNo training change beat {args.base} on validation by more than {args.margin * 100:.0f} point(s); "
              f"{args.out} not created.")
        sys.exit(2)

    combined["version"], combined["output_dir"], combined["_result"] = args.out, f"runs/vlm/{args.out}", "Pending."
    combined["_note"] = (f"{args.out}: {args.base} + every single-change ablation that improved validation HM by more than "
                         f"{args.margin * 100:.0f} point (decided on validation only, by train/combine_ablations.py). "
                         f"Included: {', '.join(included)}. Decision log: " + " | ".join(lines)
                         + " Changes may not add up: compare this run's validation HM with the best single variant.")
    if args.alias_if_identical and len(included) == 1:
        # Only one variant qualified, so the combined config IS that variant's (apart from version, output_dir and
        # notes). Copy its finished run instead of training an identical model again.
        twin = included[0]
        combined["_note"] += (f" Identical settings to {twin}, so {twin}'s finished run was copied here instead of retraining "
                              f"(identical configs reproduce bit-for-bit in this project: v12's training matched v9's every epoch).")
        dst = RUNS / args.out
        shutil.copytree(RUNS / twin, dst)
        (dst / "config.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        m = json.loads((dst / "metrics.json").read_text())
        m["version"], m["copied_from"] = args.out, twin
        (dst / "metrics.json").write_text(json.dumps(m, indent=2))
        print(f"\n{args.out} has the same settings as {twin}: copied runs/vlm/{twin} to runs/vlm/{args.out} (no retraining)")
    (CFG / f"{args.out}_config.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote config/vlm/{args.out}_config.json with {', '.join(included)}")

    if args.ensemble_out and ens_hm is not None and (ens_improved or args.force_ensemble):
        ens = load(args.ensemble)
        ens["members"][0] = {"run": f"runs/vlm/{args.out}"}
        ens["version"], ens["output_dir"], ens["_result"] = args.ensemble_out, f"runs/vlm/{args.ensemble_out}", "Pending."
        ens["_note"] = (f"{args.ensemble_out}: the {args.ensemble} ensemble recipe applied to {args.out} instead of {args.base}: "
                        f"{args.out} + {ens['members'][1]['run'].split('/')[-1]}, weight and gamma chosen on validation. "
                        + (f"Created because {args.ensemble} improved validation HM over {args.base} ({ens_hm:.2%} vs {base_hm:.2%})."
                           if ens_improved else
                           f"Created on request (--force-ensemble): {args.ensemble}'s validation HM {ens_hm:.2%} is only "
                           f"{(ens_hm - base_hm) * 100:+.1f} pts over {args.base}'s recorded {base_hm:.2%}, under the "
                           f"{args.margin * 100:.0f}-point margin, but against {args.base} retrained on the same split inside "
                           f"{args.ensemble} it improved clearly (see {args.ensemble}'s _result)."))
        (CFG / f"{args.ensemble_out}_config.json").write_text(json.dumps(ens, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote config/vlm/{args.ensemble_out}_config.json (ensemble of {args.out})")


if __name__ == "__main__":
    main()
