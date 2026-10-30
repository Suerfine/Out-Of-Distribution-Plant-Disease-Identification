"""Write an ensemble config with a new member only if that member beat the one it replaces on VALIDATION.

Usage:  python -m train.ensemble_if_better --template v16 --replace runs/baseline/aniq_baseline_independent_cnn \
            --candidate runs/baseline/aniq_baseline_independent_cnn_v2 --out v17 [--margin 0.02]

Compares the candidate's and the replaced member's validation HM (val_hm_mean in metrics.json; they must use the same
validation splits). If the candidate is better by more than --margin, copies config/vlm/<template>_config.json, swaps
the member and writes config/vlm/<out>_config.json; otherwise exits with code 2. Test numbers are never read.
"""

import argparse
import json
import sys

from dpd.data import PROJECT_ROOT

CFG = PROJECT_ROOT / "config" / "vlm"


def val_of(run: str) -> dict | None:
    m = PROJECT_ROOT / run / "metrics.json"
    return json.loads(m.read_text())["val"] if m.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--replace", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.02)
    args = ap.parse_args()

    old, new = val_of(args.replace), val_of(args.candidate)
    if new is None:
        print(f"{args.candidate} did not finish; {args.out} not created.")
        sys.exit(2)
    old_hm, new_hm = old["val_hm_mean"], new["val_hm_mean"]
    seeds_note = f"(validation splits: {len(old.get('val_hm_per_split', [0]))} vs {len(new.get('val_hm_per_split', [0]))})"
    verdict = (f"{args.candidate.split('/')[-1]} val HM {new_hm:.2%} vs {args.replace.split('/')[-1]} {old_hm:.2%}: "
               f"{(new_hm - old_hm) * 100:+.1f} pts, needs > +{args.margin * 100:.0f} {seeds_note}")
    if "plain_scoring" in new:  # the candidate's networks scored the old way: shows how much the scoring search added
        verdict += f"; candidate with plain scoring: {new['plain_scoring']['val_hm_mean']:.2%}"
    print(verdict)
    if new_hm <= old_hm + args.margin:
        print(f"Not better by the margin; {args.out} not created.")
        sys.exit(2)

    template_path = CFG / f"{args.template}_config.json"
    cfg = json.loads(template_path.read_text(encoding="utf-8"))
    idx = [m["run"] for m in cfg["members"]].index(args.replace)
    cfg["members"][idx] = {"run": args.candidate}
    cfg["version"], cfg["output_dir"], cfg["_result"] = args.out, f"runs/vlm/{args.out}", "Pending."
    first = cfg["members"][0]["run"].split("/")[-1]
    cfg["_note"] = (f"{args.out}: {args.template}'s ensemble recipe with the CNN member replaced: {first} + "
                    f"{args.candidate.split('/')[-1]} (instead of {args.replace.split('/')[-1]}), weight and gamma chosen on "
                    f"validation. Created by train/ensemble_if_better.py because the new CNN beat the old one on validation: "
                    f"{verdict}. Compare with {args.template} on validation to see whether the stronger CNN also makes a "
                    f"stronger ensemble (a stronger member can add less if its errors overlap more with {first}'s).")
    (CFG / f"{args.out}_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote config/vlm/{args.out}_config.json")


if __name__ == "__main__":
    main()
