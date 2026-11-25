"""Compact companion to the progression figure: how much each step changed seen and unseen Top-1 (test, official split).

Run from the repo root:  .venv/Scripts/python docs/vlm/diagrams/make_seen_unseen_delta_figure.py
Writes docs/vlm/diagrams/results/fig_seen_unseen_delta.{svg,drawio,png}. Reuses the steps and the Canvas of
make_progression_gamma0_figure.py, so the numbers are the same as in that figure and in the report's results table.
F is measured against the matched re-run of the un-anchored GAN, as in the report; every other step against the step before.
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_progression_gamma0_figure import EDGE, OUT, Canvas, steps  # noqa: E402

BLACK = "#000000"
SEEN, UNSEEN = "#9DC3E6", "#F4B183"
LABELS = {  # report section letter and short name, keyed by the progression figure's step names
    "Clean names + disease descriptions": "B  Disease descriptions",
    "CSP prompt learning": "C  CSP prompt learning",
    "LoRA image encoder": "D  LoRA image encoder",
    "f-CLSWGAN": "E  f-CLSWGAN",
    "+ crop-anchored GAN": "F  Crop-anchored GAN *",
    "+ disease-only head": "G  Disease-only head",
    "+ ConvNeXt seen handoff": "H  ConvNeXt seen handoff",
    "+ calibrated stacking (γ)": "I  Calibrated stacking (γ)",
}


def deltas(rows):
    out, prev, ref = [], None, None
    for r in rows:
        if r["group"] == "reference":
            ref = r
            continue
        if prev is not None:
            base = ref if r["name"].startswith("+ crop-anchored") else prev
            # differences of the table's rounded values, so every label matches the numbers quoted in the report text
            d = lambda k: round(r[k] * 100, 1) - round(base[k] * 100, 1)  # noqa: E731
            out.append((LABELS[r["name"]], d("seen"), d("unseen")))
        prev = r
    return out


def draw(d) -> Canvas:
    W, row_h, top = 900, 34, 92
    H = top + row_h * len(d) + 70
    c = Canvas(W, H)
    t = c.text
    c.rect(0, 0, W, H, "#FFFFFF", "#FFFFFF", sw=0)
    t(20, 30, "Where each step's gain comes from", 16, "start", weight="bold")
    t(20, 52, "Change in seen and unseen Top-1 (percentage points, official test set), each step against the one before.", 12, "start")
    x0, x1, vmin, vmax = 250, W - 40, -10, 50  # plot area and value range
    sx = (x1 - x0) / (vmax - vmin)
    X = lambda v: x0 + (v - vmin) * sx  # noqa: E731
    y_end = top + row_h * len(d)
    for v in range(vmin, vmax + 1, 10):
        c.line([(X(v), top - 6), (X(v), y_end)], "#E6E6E6" if v else BLACK, 1.0 if v else 1.2)
        t(X(v), top - 12, f"{v:+d}".replace("-", "−") if v else "0", 10)
    for i, (label, ds, du) in enumerate(d):
        y = top + i * row_h
        t(20, y + row_h / 2 + 4, label, 12, "start", weight="bold" if label.startswith("I") else "normal")
        for k, (val, fill) in enumerate(((ds, SEEN), (du, UNSEEN))):
            by, bh = y + 5 + k * 12, 11
            a, b = sorted((X(0), X(val)))
            c.rect(a, by, max(b - a, 0.5), bh, fill, BLACK, sw=0.8)
            txt = f"{val:+.1f}".replace("-", "−")
            if val >= 0:
                t(b + 4, by + 9, txt, 10, "start")
            else:
                t(a - 4, by + 9, txt, 10, "end")
        if i:
            c.line([(20, y), (x1, y)], "#F0F0F0", 0.8)
    ly = y_end + 26
    for fill, lab, lx in ((SEEN, "seen Top-1", 250), (UNSEEN, "unseen Top-1", 370)):
        c.rect(lx, ly - 10, 22, 12, fill, BLACK, sw=0.8)
        t(lx + 28, ly, lab, 11, "start")
    t(20, ly + 24, "* F against the matched re-run of the un-anchored GAN (Hein's pipeline), not against E. "
                   "B–H at γ = 0; I adds calibrated stacking to H.", 10, "start")
    return c


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = deltas(steps())
    for label, ds, du in d:
        print(f"{label:<32} seen {ds:+6.1f}  unseen {du:+6.1f}")
    c = draw(d)
    svg = OUT / "fig_seen_unseen_delta.svg"
    svg.write_text(c.svg(), encoding="utf-8")
    (OUT / "fig_seen_unseen_delta.drawio").write_text(c.drawio("seen_unseen_delta"), encoding="utf-8")
    if EDGE.exists():
        png = OUT / "fig_seen_unseen_delta.png"
        png.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                            "--force-device-scale-factor=2", f"--window-size={c.w},{c.h}", f"--user-data-dir={tmp}",
                            f"--screenshot={png}", svg.as_uri()], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            for _ in range(40):
                if png.exists() and png.stat().st_size > 0:
                    break
                time.sleep(0.5)
        if not png.exists():
            raise SystemExit(f"Edge wrote no PNG: {png}")
        print("PNG:", png)


if __name__ == "__main__":
    main()
