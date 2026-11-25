"""Progression figure: the agreed VLM flow, from zero-shot CLIP to the final CLIP + GAN + ConvNeXt pipeline, at gamma 0.

Run from the repo root:  .venv/Scripts/python docs/vlm/diagrams/make_progression_gamma0_figure.py
Writes docs/vlm/diagrams/results/fig_progression_gamma0.svg, an editable fig_progression_gamma0.drawio and, with
Microsoft Edge installed, a PNG of the SVG (headless
Edge, local file only).

Sources (official test set, single run per step):
  v1-v11.3      computed here from runs/vlm/<v>/scores.npz at gamma 0; calibrated HM from runs/vlm/<v>/metrics.json.
  Hein's steps  Disease_Transfer_Experiments/clip_anchored_gan_v1:
                official/results.json (original vs crop-anchored GAN, gamma 0),
                official/fixed_gamma_results.json (same two at the shared gamma 0.03),
                clip_disease_only/official/seen_handoff_ablation.json (disease-only head without / with the ConvNeXt
                seen handoff, at gamma 0 and at the selected routing offset 3.0).
"""
import html
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.evaluate import evaluate  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
HEIN = ROOT / "Disease_Transfer_Experiments/clip_anchored_gan_v1"
INK, GREY, MUTED = "#1B1F24", "#868E96", "#495057"
GROUPS = {  # fill, stroke
    "zeroshot": ("#F1F3F5", "#868E96"),
    "trained": ("#E7F0FF", "#2F6FD0"),
    "pipeline": ("#FFF4D6", "#C99A2E"),
}


def ours(version: str, pairs) -> dict:
    z = np.load(ROOT / f"runs/vlm/{version}/scores.npz")
    e = evaluate(z["test_seen_scores"], z["test_seen_labels"], z["test_unseen_scores"], z["test_unseen_labels"], pairs, 0.0, (1,))
    m = json.loads((ROOT / f"runs/vlm/{version}/metrics.json").read_text())
    cal = m["harmonic_mean_top1"] if m.get("gamma", 0) else None
    return {"seen": e["test_seen"]["top1"], "unseen": e["test_unseen"]["top1"], "hm": e["harmonic_mean_top1"],
            "cal": cal, "cal_note": f"\u03b3 {m['gamma']}" if cal else ""}


def steps() -> list[dict]:
    pairs = pair_table(read_metadata())
    g0 = json.loads((HEIN / "official/results.json").read_text())
    g3 = json.loads((HEIN / "official/fixed_gamma_results.json").read_text())["arms"]
    ho = json.loads((HEIN / "clip_disease_only/official/seen_handoff_ablation.json").read_text())

    def hein(seen, unseen, cal, note):
        return {"seen": seen, "unseen": unseen, "hm": 2 * seen * unseen / (seen + unseen), "cal": cal, "cal_note": note}

    rows = [
        ("CLIP zero-shot", "v1", "zeroshot", ours("v1", pairs)),
        # v2 (class-name cleanup + template averaging, HM 5.4 -> 5.6) is merged into this bar: it mainly prepares the names
        # that v3's descriptions and v4's learned vectors start from.
        ("Clean names + disease descriptions", "v2 + v3", "zeroshot", ours("v3", pairs)),
        # v4.2, not v4: v4.2 (v4 with the fixed validation split and longer, steadier training) is the run LoRA was added to.
        ("CSP prompt learning", "v4.2", "trained", ours("v4.2", pairs)),
        # v5.5 (= v5.3: LoRA rank 16 on the last 8 blocks), not v5 (rank 8, last 4): it is the encoder recipe the rest of
        # the chain uses (v10.3 = v5.5 + text anchor; v11.3's GAN runs on v10.3's features).
        ("LoRA image encoder", "v5.5 (rank 16, 8 blocks)", "trained", ours("v5.5", pairs)),
        ("f-CLSWGAN", "v11.3", "trained", ours("v11.3", pairs)),
        ("Un-anchored GAN, re-run", "reference", "reference",
         hein(g0["test_seen"]["original"]["top1"], g0["test_unseen"]["original"]["top1"], g3["original"]["harmonic_mean"], "shared \u03b3 0.03")),
        ("+ crop-anchored GAN", "Hein", "pipeline",
         hein(g0["test_seen"]["anchored"]["top1"], g0["test_unseen"]["anchored"]["top1"], g3["anchored"]["harmonic_mean"], "shared \u03b3 0.03")),
        ("+ disease-only head", "Hein", "pipeline",
         hein(ho["0"]["calibration_only"]["seen"], ho["0"]["calibration_only"]["unseen"], ho["3"]["calibration_only"]["h"], "offset 3.0")),
        ("+ ConvNeXt seen handoff", "Hein, final", "pipeline",
         hein(ho["0"]["two_experts"]["seen"], ho["0"]["two_experts"]["unseen"], ho["3"]["two_experts"]["h"], "offset 3.0")),
        # Final step: the same pipeline with calibrated stacking switched on (routing offset 3.0, chosen by Hein on the
        # seed-42 validation split; routing with an offset = subtracting it from every seen score).
        ("+ calibrated stacking (γ)", "offset 3.0", "calibration",
         hein(ho["3"]["two_experts"]["seen"], ho["3"]["two_experts"]["unseen"], None, "")),
    ]
    return [{"name": n, "tag": t, "group": g, **d} for n, t, g, d in rows]


class Canvas:
    """Records drawing primitives once and renders them as SVG (for the PNG) or as an editable draw.io diagram."""

    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill, stroke, sw=1.0, rx=0, dashed=False, hatch=False):
        self.items.append(("rect", dict(x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, rx=rx, dashed=dashed, hatch=hatch)))

    def text(self, x, y, t, size=12, anchor="middle", color=INK, weight="normal", rotate=False):
        """(x, y) is the text baseline anchor point, as in SVG."""
        self.items.append(("text", dict(x=x, y=y, t=t, size=size, anchor=anchor, color=color, weight=weight, rotate=rotate)))

    def line(self, points, color, sw=1.0):
        self.items.append(("line", dict(points=points, color=color, sw=sw)))

    # ---- SVG ----
    def svg(self) -> str:
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
               f'font-family="Arial, Helvetica, sans-serif" fill="#000000">',
               '<defs><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
               f'<line x1="0" y1="0" x2="0" y2="7" stroke="#000000" stroke-width="1"/></pattern></defs>',
               f'<rect width="{self.w}" height="{self.h}" fill="white"/>']
        for kind, a in self.items:
            if kind == "rect":
                fill = "url(#hatch)" if a["hatch"] else a["fill"]
                dash = ' stroke-dasharray="5 3"' if a["dashed"] else ""
                out.append(f'<rect x="{a["x"]:.1f}" y="{a["y"]:.1f}" width="{a["w"]:.1f}" height="{a["h"]:.1f}" rx="{a["rx"]}" '
                           f'fill="{fill}" stroke="{a["stroke"]}" stroke-width="{a["sw"]}"{dash}/>')
            elif kind == "text":
                rot = f' transform="rotate(-90 {a["x"]:.1f} {a["y"]:.1f})"' if a["rotate"] else ""
                out.append(f'<text x="{a["x"]:.1f}" y="{a["y"]:.1f}" text-anchor="{a["anchor"]}" font-size="{a["size"]}" '
                           f'fill="{a["color"]}" font-weight="{a["weight"]}"{rot}>{html.escape(a["t"])}</text>')
            else:
                pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in a["points"])
                out.append(f'<polyline points="{pts}" fill="none" stroke="{a["color"]}" stroke-width="{a["sw"]}"/>')
        out.append("</svg>")
        return "\n".join(out)

    # ---- draw.io ----
    def drawio(self, name: str) -> str:
        cells, n = [], 0

        def cell(value, style, x, y, w, h, vertex=True, extra=""):
            nonlocal n
            n += 1
            kind = 'vertex="1"' if vertex else 'edge="1"'
            cells.append(f'<mxCell id="c{n}" value="{html.escape(value)}" style="{style}" {kind} parent="1">'
                         f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"{">" + extra + "</mxGeometry>" if extra else "/>"}'
                         f'</mxCell>')

        for kind, a in self.items:
            if kind == "rect":
                st = (f"rounded={1 if a['rx'] else 0};arcSize={a['rx'] * 2 if a['rx'] else 0};absoluteArcSize=1;html=1;"
                      f"fillColor={a['fill']};strokeColor={a['stroke']};strokeWidth={a['sw']};")
                if a["hatch"]:
                    st += "fillColor=#000000;fillStyle=hatch;"
                if a["dashed"]:
                    st += "dashed=1;dashPattern=5 3;"
                cell("", st, a["x"], a["y"], a["w"], a["h"])
            elif kind == "text":
                size, t = a["size"], a["t"]
                w = max(40, len(t) * size * 0.58 + 8)
                h = size + 8
                align = {"middle": "center", "start": "left", "end": "right"}[a["anchor"]]
                cx = {"middle": a["x"] - w / 2, "start": a["x"], "end": a["x"] - w}[a["anchor"]]
                cy = a["y"] - size * 0.35 - h / 2
                st = (f"text;html=1;whiteSpace=nowrap;align={align};verticalAlign=middle;spacing=0;fontFamily=Arial;"
                      f"fontSize={size};fontColor={a['color']};fontStyle={1 if a['weight'] == 'bold' else 0};")
                if a["rotate"]:
                    st += "rotation=-90;"
                    cx, cy = a["x"] - w / 2, a["y"] - h / 2
                cell(t, st, cx, cy, w, h)
            else:
                (x1, y1), (x2, y2) = a["points"][0], a["points"][-1]
                mid = "".join(f'<mxPoint x="{x:.1f}" y="{y:.1f}"/>' for x, y in a["points"][1:-1])
                pts = (f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/><mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/>'
                       + (f'<Array as="points">{mid}</Array>' if mid else ""))
                st = f"html=1;endArrow=none;rounded=0;strokeColor={a['color']};strokeWidth={a['sw']};"
                n += 1
                cells.append(f'<mxCell id="c{n}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')
        model = ('<mxGraphModel grid="0" page="0" background="#FFFFFF"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                 + "".join(cells) + "</root></mxGraphModel>")
        return f'<mxfile><diagram name="{html.escape(name)}">{model}</diagram></mxfile>'


BAR = {  # academic style: square corners, black outlines, one soft fill per stage
    "zeroshot": "#E7E6E6",
    "trained": "#BDD7EE",
    "pipeline": "#F8CBAD",
    "calibration": "#C5E0B4",
}
BLACK = "#000000"


def draw(rows: list[dict]) -> Canvas:
    W, H = 1500, 820
    c = Canvas(W, H)
    text = c.text
    left, base, scale = 70, 520, 4.5  # bar baseline y, pixels per HM point (axis 0-80)
    slot = (W - left - 40) / len(rows)
    bw = slot * 0.56
    c.rect(0, 0, W, H, "#FFFFFF", "#FFFFFF", sw=0)
    text(30, 36, "VLM progression: from zero-shot CLIP to the final CLIP + GAN + ConvNeXt pipeline, then calibrated stacking",
         18, "start", weight="bold")
    text(30, 60, "Harmonic mean of seen and unseen Top-1 on the official test set (2,000 seen / 1,448 unseen images), one run per step. "
                 "Every step at γ = 0 except the last, which adds calibrated stacking.", 13, "start")

    # stage brackets above the bars (no colour: a line with end ticks and a label)
    spans = {}
    for i, r in enumerate(rows):
        g = "pipeline" if r["group"] == "reference" else r["group"]
        spans.setdefault(g, [i, i])[1] = i
    labels = {"zeroshot": "Zero-shot CLIP (no training)", "trained": "Trained CLIP (this project)",
              "pipeline": "CLIP + GAN + ConvNeXt pipeline (Hein)", "calibration": "Calibration"}
    for g, (a, b) in spans.items():
        x0, x1 = left + a * slot + 10, left + (b + 1) * slot - 10
        c.line([(x0, 112), (x0, 104), (x1, 104), (x1, 112)], BLACK, 1.2)
        text((x0 + x1) / 2, 96, labels[g], 13, weight="bold")

    # axis: y axis line with ticks, light reference lines
    for v in range(0, 81, 10):
        y = base - v * scale
        if v:
            c.line([(left, y), (W - 40, y)], "#E6E6E6")
        c.line([(left - 6, y), (left, y)], BLACK)
        text(left - 10, y + 4, f"{v}", 11, "end")
    c.line([(left, base), (left, base - 80 * scale)], BLACK, 1.2)
    text(24, base - 40 * scale, "HM Top-1 (%)", 12, rotate=True)
    c.line([(left, base), (W - 40, base)], BLACK, 1.2)

    prev, ref, ref_cx = None, None, None
    for i, r in enumerate(rows):
        cx = left + (i + 0.5) * slot
        h = r["hm"] * 100 * scale
        if r["group"] == "reference":
            c.rect(cx - bw / 2, base - h, bw, h, "#FFFFFF", BLACK, sw=1.2, dashed=True, hatch=True)
        else:
            c.rect(cx - bw / 2, base - h, bw, h, BAR[r["group"]], BLACK, sw=1.2)
        text(cx, base - h - 24, f"{r['hm'] * 100:.1f}", 14, weight="bold")
        if prev is not None and r["group"] != "reference" and not (prev["group"] == "trained" and r["group"] == "pipeline"):
            d = (r["hm"] - prev["hm"]) * 100
            text(cx, base - h - 9, f"{d:+.1f}".replace("-", "−"), 11, color="#2E7D32" if d > 0 else "#C62828", weight="bold")
        words = r["name"].split(" ")  # two lines, split near the middle, so neighbouring labels never touch
        cut = min(range(1, len(words) + 1), key=lambda k: abs(len(" ".join(words[:k])) - len(r["name"]) / 2))
        for k, line in enumerate(filter(None, (" ".join(words[:cut]), " ".join(words[cut:])))):
            text(cx, base + 20 + 15 * k, line, 12, weight="bold")
        text(cx, base + 51, r["tag"], 11)
        text(cx, base + 71, f"seen {r['seen'] * 100:.1f}", 11)
        text(cx, base + 86, f"unseen {r['unseen'] * 100:.1f}", 11)
        if r["group"] == "calibration":
            text(cx, base + 106, "calibrated", 11)
            text(cx, base + 120, "(this bar)", 10)
            continue
        cal = f"{r['cal'] * 100:.1f}" if r["cal"] else "= γ 0"
        text(cx, base + 106, f"calibrated {cal}", 11)
        if r["cal_note"]:
            text(cx, base + 120, f"({r['cal_note']})", 10)
        if r["group"] == "reference":
            ref, ref_cx = r, cx
        else:
            prev = r

    # bracket: the matched gain of crop anchoring is measured against the re-run of the un-anchored GAN
    i_anch = next(i for i, r in enumerate(rows) if r["name"].startswith("+ crop-anchored"))
    ca, x_anc = rows[i_anch], left + (i_anch + 0.5) * slot
    ytop = base - max(ref["hm"], ca["hm"]) * 100 * scale - 46
    c.line([(ref_cx, ytop + 10), (ref_cx, ytop), (x_anc, ytop), (x_anc, ytop + 10)], BLACK, 1.0)
    text((ref_cx + x_anc) / 2, ytop - 6, f"matched gain {(ca['hm'] - ref['hm']) * 100:+.1f}", 11, weight="bold")

    # legend (as in the architecture figure: swatch + label in one row)
    ly, lx = 668, 30
    for fill, hatch, dashed, label in ((BAR["zeroshot"], False, False, "zero-shot CLIP"), (BAR["trained"], False, False, "trained CLIP"),
                                       (BAR["pipeline"], False, False, "CLIP + GAN + ConvNeXt pipeline"),
                                       (BAR["calibration"], False, False, "calibrated stacking (γ)"),
                                       ("#FFFFFF", True, True, "matched baseline (re-run)")):
        c.rect(lx, ly - 11, 28, 14, fill, BLACK, sw=1.0, dashed=dashed, hatch=hatch)
        text(lx + 36, ly, label, 12, "start")
        lx += 36 + len(label) * 7 + 34
    text(lx, ly, "+x / −x: change from the previous step", 12, "start")

    notes = [
        "Last bar: the final pipeline with calibrated stacking (routing offset γ = 3.0, chosen on validation), which subtracts γ from "
        "every seen score: seen 72.9 → 66.8, unseen 48.3 → 82.8.",
        "Hatched bar: Hein's re-run of the same un-anchored GAN (v11.3 settings) in his pipeline, the matched baseline for "
        "crop anchoring. v11.3 → re-run (44.0 → 46.3) is run-to-run difference, not a method change.",
        "“+ disease-only head” also brings in the ConvNeXt crop scores (measured alone: 49.0 → 50.9) "
        "and two-expert routing; the final step hands seen predictions to the ConvNeXt (+104 net correct seen images).",
        "Read with care: at γ = 0, HM mostly tracks how seen-biased a model is (r = 1.00 with the share of unseen images "
        "routed to unseen pairs, 25 LoRA runs), not how well it recognises.",
        "Example: training CSP longer (v4, 3 epochs → v4.2, 13 epochs) lowers HM at γ = 0 from 34.9 to 26.0 with AUSUC "
        "unchanged (38.4); LoRA then adds +17.5 AUSUC. Calibrated HM is the protocol's headline number.",
    ]
    for k, t in enumerate(notes):
        text(30, 704 + k * 20, t, 12, "start")
    return c


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # step names contain "γ"; the Windows console default (cp1252) cannot print it
    OUT.mkdir(exist_ok=True)
    rows = steps()
    canvas = draw(rows)
    path = OUT / "fig_progression_gamma0.svg"
    path.write_text(canvas.svg(), encoding="utf-8")
    (OUT / "fig_progression_gamma0.drawio").write_text(canvas.drawio("progression_gamma0"), encoding="utf-8")
    for r in rows:
        print(f"{r['name']:<28} {r['tag']:<12} seen {r['seen'] * 100:5.1f} unseen {r['unseen'] * 100:5.1f} "
              f"HM {r['hm'] * 100:5.2f} cal {'' if r['cal'] is None else round(r['cal'] * 100, 2)}")
    if EDGE.exists():
        png = OUT / "fig_progression_gamma0.png"
        png.unlink(missing_ok=True)  # so a failed render can never leave a stale PNG behind
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # own profile: a running Edge cannot swallow the call
            subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                            "--force-device-scale-factor=2", f"--window-size={canvas.w},{canvas.h}", f"--user-data-dir={tmp}",
                            f"--screenshot={png}", path.as_uri()], check=False,
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
