"""Figure for results section B: how disease descriptions change the embedding space of zero-shot CLIP (v1 -> v3).

Run from the repo root:  .venv/Scripts/python docs/vlm/diagrams/make_b_embedding_figure.py
Writes docs/vlm/diagrams/results/fig_b_embedding.{svg,drawio,png} (PNG with Microsoft Edge, headless, local file only).

Panels (a) and (b) are schematic; their cosine values and every number in panel (c) are computed here from the frozen CLIP
encoders: the v1 / v2 / v3 prompts are re-encoded and scored against the cached embeddings of all 3,448 test images.
  crop signal     mean score of the image's own-crop prompts minus the other crops' prompts
  disease signal  score of the right-disease prompt minus the mean of the wrong-disease prompts of the same crop
  within-crop win share of images whose correct prompt beats every other prompt of its crop
Style: square corners, black outlines, one soft fill per version (as fig_progression_gamma0).
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
from dpd.clip_utils import build_prompts, encode_images, encode_prompts, load_clip  # noqa: E402
from dpd.data import pair_table, read_metadata  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
BLACK, GREY = "#000000", "#595959"
FILL = {"v1": "#FFFFFF", "v2": "#E7E6E6", "v3": "#BDD7EE"}


def measure() -> dict:
    df = read_metadata()
    p = pair_table(df)
    crop, dis = p.crop_id.to_numpy(), p.disease_id.to_numpy()
    c1 = json.loads((ROOT / "runs/vlm/v1/config.json").read_text(encoding="utf-8"))
    model, pre, tok = load_clip(c1["model"], "cuda")
    I = np.concatenate([encode_images(model, pre, df.loc[df.split == s, "file"], c1["model"], c1["runtime"], cache_name=s)
                        for s in ("test_seen", "test_unseen")])
    I /= np.linalg.norm(I, axis=1, keepdims=True)
    y = np.concatenate([df.loc[df.split == s, "pair_id"].to_numpy() for s in ("test_seen", "test_unseen")])
    same_crop = [np.flatnonzero(crop == crop[t]) for t in y]
    out = {}
    for v in ("v1", "v2", "v3"):
        cfg = json.loads((ROOT / f"runs/vlm/{v}/config.json").read_text(encoding="utf-8"))
        T = encode_prompts(model, tok, build_prompts(p, cfg["names"], cfg["prompts"]), "cuda")
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        S = I @ T.T
        iu = np.triu_indices(len(p), 1)
        C = (T @ T.T)[iu]
        sc, sd = (crop[:, None] == crop[None])[iu], (dis[:, None] == dis[None])[iu]
        others = [g[g != t] for g, t in zip(same_crop, y)]
        out[v] = {
            "cos_same_crop": float(C[sc & ~sd].mean()),
            "crop_signal": float(np.mean([S[i, crop == crop[t]].mean() - S[i, crop != crop[t]].mean() for i, t in enumerate(y)])),
            "disease_signal": float(np.mean([S[i, t] - S[i, o].mean() for i, (t, o) in enumerate(zip(y, others)) if len(o)])),
            "within_crop_win": float(np.mean([S[i, t] > S[i, o].max() for i, (t, o) in enumerate(zip(y, others)) if len(o)])),
        }
    return out


class Canvas:
    """Primitives recorded once, rendered as SVG (for the PNG) or as an editable draw.io diagram."""

    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill="#FFFFFF", stroke=BLACK, sw=1.0, dashed=False):
        self.items.append(("rect", dict(x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, dashed=dashed)))

    def circle(self, cx, cy, r, fill="#FFFFFF", stroke=BLACK, sw=1.0, dashed=False):
        self.items.append(("ellipse", dict(x=cx - r, y=cy - r, w=2 * r, h=2 * r, fill=fill, stroke=stroke, sw=sw, dashed=dashed)))

    def text(self, x, y, t, size=12, anchor="middle", color=BLACK, weight="normal", italic=False):
        self.items.append(("text", dict(x=x, y=y, t=t, size=size, anchor=anchor, color=color, weight=weight, italic=italic)))

    def line(self, points, color=BLACK, sw=1.0, arrow=False, dashed=False):
        self.items.append(("line", dict(points=points, color=color, sw=sw, arrow=arrow, dashed=dashed)))

    def svg(self) -> str:
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
             f'font-family="Arial, Helvetica, sans-serif" fill="{BLACK}">',
             '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{BLACK}"/></marker></defs>',
             f'<rect width="{self.w}" height="{self.h}" fill="white"/>']
        for kind, a in self.items:
            dash = ' stroke-dasharray="5 3"' if a.get("dashed") else ""
            if kind == "rect":
                o.append(f'<rect x="{a["x"]:.1f}" y="{a["y"]:.1f}" width="{a["w"]:.1f}" height="{a["h"]:.1f}" fill="{a["fill"]}" '
                         f'stroke="{a["stroke"]}" stroke-width="{a["sw"]}"{dash}/>')
            elif kind == "ellipse":
                o.append(f'<ellipse cx="{a["x"] + a["w"] / 2:.1f}" cy="{a["y"] + a["h"] / 2:.1f}" rx="{a["w"] / 2:.1f}" ry="{a["h"] / 2:.1f}" '
                         f'fill="{a["fill"]}" stroke="{a["stroke"]}" stroke-width="{a["sw"]}"{dash}/>')
            elif kind == "text":
                st = ' font-style="italic"' if a["italic"] else ""
                o.append(f'<text x="{a["x"]:.1f}" y="{a["y"]:.1f}" text-anchor="{a["anchor"]}" font-size="{a["size"]}" '
                         f'fill="{a["color"]}" font-weight="{a["weight"]}"{st}>{html.escape(a["t"])}</text>')
            else:
                pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in a["points"])
                mk = ' marker-end="url(#a)"' if a["arrow"] else ""
                o.append(f'<polyline points="{pts}" fill="none" stroke="{a["color"]}" stroke-width="{a["sw"]}"{dash}{mk}/>')
        o.append("</svg>")
        return "\n".join(o)

    def drawio(self, name: str) -> str:
        cells, n = [], 0
        for kind, a in self.items:
            n += 1
            dash = "dashed=1;dashPattern=5 3;" if a.get("dashed") else ""
            if kind in ("rect", "ellipse"):
                shape = "ellipse;" if kind == "ellipse" else "rounded=0;"
                st = f"{shape}html=1;fillColor={a['fill']};strokeColor={a['stroke']};strokeWidth={a['sw']};{dash}"
                cells.append(f'<mxCell id="c{n}" value="" style="{st}" vertex="1" parent="1">'
                             f'<mxGeometry x="{a["x"]:.1f}" y="{a["y"]:.1f}" width="{a["w"]:.1f}" height="{a["h"]:.1f}" as="geometry"/></mxCell>')
            elif kind == "text":
                size, t = a["size"], a["t"]
                w, h = max(30, len(t) * size * 0.56 + 8), size + 8
                align = {"middle": "center", "start": "left", "end": "right"}[a["anchor"]]
                x = {"middle": a["x"] - w / 2, "start": a["x"], "end": a["x"] - w}[a["anchor"]]
                style = (1 if a["weight"] == "bold" else 0) + (2 if a["italic"] else 0)
                st = (f"text;html=1;whiteSpace=nowrap;align={align};verticalAlign=middle;spacing=0;fontFamily=Arial;"
                      f"fontSize={size};fontColor={a['color']};fontStyle={style};")
                cells.append(f'<mxCell id="c{n}" value="{html.escape(t)}" style="{st}" vertex="1" parent="1">'
                             f'<mxGeometry x="{x:.1f}" y="{a["y"] - size * 0.35 - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')
            else:
                (x1, y1), (x2, y2) = a["points"][0], a["points"][-1]
                mid = "".join(f'<mxPoint x="{x:.1f}" y="{y:.1f}"/>' for x, y in a["points"][1:-1])
                st = (f"html=1;rounded=0;endArrow={'block;endFill=1' if a['arrow'] else 'none'};"
                      f"strokeColor={a['color']};strokeWidth={a['sw']};{dash}")
                cells.append(f'<mxCell id="c{n}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                             f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/><mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/>'
                             + (f'<Array as="points">{mid}</Array>' if mid else "") + "</mxGeometry></mxCell>")
        model = ('<mxGraphModel grid="0" page="0" background="#FFFFFF"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                 + "".join(cells) + "</root></mxGraphModel>")
        return f'<mxfile><diagram name="{html.escape(name)}">{model}</diagram></mxfile>'


def draw(m: dict) -> Canvas:
    W, H = 1320, 530
    c = Canvas(W, H)
    t = c.text
    t(30, 34, "How disease descriptions change the embedding space (zero-shot CLIP, v1 \u2192 v3)", 18, "start", weight="bold")
    t(30, 56, "Panels (a) and (b) are schematic; all numbers are measured on the 3,448 test images. Both encoders are frozen.",
      12, "start", color=GREY)

    def prompt_dot(x, y, label, lx, ly, anchor="start"):
        c.circle(x, y, 5, fill=BLACK)
        t(lx, ly, label, 11, anchor)

    def image_dot(x, y):
        c.rect(x - 6, y - 6, 12, 12, fill=FILL["v3"], sw=1.2)

    # ---- (a) v1 ----
    ax, ay, aw, ah = 30, 80, 400, 380
    c.rect(ax, ay, aw, ah)
    t(ax + 12, ay + 24, "(a) v1: name-only prompts", 13, "start", weight="bold")
    t(ax + 12, ay + 44, "\u201ca photo of tomato leaves with {disease}.\u201d", 11, "start", italic=True)
    cx, cy = ax + 200, ay + 210
    c.circle(cx, cy, 52, dashed=True)
    for (dx, dy), lab in zip(((-18, -16), (16, -20), (-14, 18), (18, 14)), ("early blight", "late blight", "leaf blight", "leaf mold")):
        prompt_dot(cx + dx, cy + dy, "", 0, 0)
    t(cx, cy + 72, "18 tomato prompts: early / late / leaf blight,", 11)
    t(cx, cy + 87, "leaf mold \u2026 differ by one or two words", 11)
    t(ax + 12, ay + 66, f"Same-crop prompts nearly identical: cos {m['v1']['cos_same_crop']:.3f}", 11, "start", weight="bold")
    ix, iy = cx - 120, cy - 70
    image_dot(ix, iy)
    t(ix, iy - 14, "image: tomato early blight", 11)
    for dx, dy in ((-18, -16), (16, -20), (-14, 18), (18, 14)):
        c.line([(ix + 6, iy + 4), (cx + dx - 4, cy + dy - 3)], GREY, 0.8, dashed=True)
    t(ax + 12, ay + ah - 40, "The image is close to the crop, but every disease prompt", 11, "start")
    t(ax + 12, ay + ah - 24, "is almost equally close: the disease is decided by noise.", 11, "start")

    # ---- (b) v3 ----
    bx = ax + aw + 30
    c.rect(bx, ay, aw, ah)
    t(bx + 12, ay + 24, "(b) v3: name + symptom description", 13, "start", weight="bold")
    t(bx + 12, ay + 44, "\u201c\u2026 with {disease}: {visual symptoms}.\u201d", 11, "start", italic=True)
    ox, oy = bx + 105, ay + 215
    c.circle(ox, oy, 6, fill="#FFFFFF", sw=1.2)
    t(ox - 10, oy + 22, "crop component", 11, "end")
    t(ox - 10, oy + 36, "(\u201ctomato\u201d)", 11, "end")
    targets = (((ox + 95, oy - 95), "concentric rings, yellow halo", "early blight", "start"),
               ((ox + 135, oy - 5), "dark water-soaked patches", "late blight", "start"),
               ((ox + 75, oy + 80), "olive-green velvety mould", "leaf mold", "start"))
    for (px, py), sym, dz, anc in targets:
        c.line([(ox + 5, oy), (px - 6, py + (3 if py < oy else -3))], BLACK, 1.0, arrow=True)
        prompt_dot(px, py, dz, px + 10, py + 4, anc)
        t(px + 10, py + 18, sym, 10, anc, color=GREY, italic=True)
    t(ox + 70, oy - 58, "symptom", 10, "start", color=GREY)
    t(ox + 70, oy - 46, "component", 10, "start", color=GREY)
    jx, jy = ox + 112, oy - 124
    image_dot(jx, jy)
    t(jx + 12, jy + 4, "image: tomato early blight", 11, "start")
    c.line([(jx - 1, jy + 7), (ox + 96, oy - 102)], BLACK, 1.2, arrow=True)
    t(bx + 12, ay + 66, f"Same-crop prompts spread apart: cos {m['v3']['cos_same_crop']:.3f}", 11, "start", weight="bold")
    t(bx + 12, ay + ah - 40, "Symptom words point where leaf images actually lie, so the", 11, "start")
    t(bx + 12, ay + ah - 24, "right disease prompt is now measurably closer to the image.", 11, "start")

    # ---- (c) measured ----
    cx0 = bx + aw + 30
    cw = W - cx0 - 30
    c.rect(cx0, ay, cw, ah)
    t(cx0 + 12, ay + 24, "(c) Measured on the test images", 13, "start", weight="bold")
    groups = [("Crop signal (\u00d710\u207b\u00b3)", "crop_signal", 1000, 50, "{:.1f}"),
              ("Disease signal (\u00d710\u207b\u00b3)", "disease_signal", 1000, 12, "{:.1f}"),
              ("Within-crop wins (%)", "within_crop_win", 100, 30, "{:.1f}")]
    base, top = ay + ah - 70, ay + 80
    gw = (cw - 40) / len(groups)
    bw = gw / 4.4
    for gi, (label, key, mult, vmax, fmt) in enumerate(groups):
        gx = cx0 + 20 + gi * gw
        for vi, v in enumerate(("v1", "v2", "v3")):
            val = m[v][key] * mult
            h = (base - top) * val / vmax
            x = gx + 8 + vi * (bw + 3)
            c.rect(x, base - h, bw, h, fill=FILL[v], sw=1.0)
            t(x + bw / 2, base - h - 5, fmt.format(val), 9)
        c.line([(gx + 2, base), (gx + gw - 6, base)], BLACK, 1.0)
        words = label.split(" ")
        t(gx + gw / 2 - 2, base + 16, " ".join(words[:2]), 10, weight="bold")
        t(gx + gw / 2 - 2, base + 29, " ".join(words[2:]), 10, weight="bold")
    lx = cx0 + 14
    for v, lab in (("v1", "v1 raw prompt"), ("v2", "v2 clean names"), ("v3", "v3 + descriptions")):
        c.rect(lx, ay + 44, 12, 12, fill=FILL[v])
        t(lx + 17, ay + 54, lab, 10, "start")
        lx += 17 + len(lab) * 5.6 + 12

    # ---- definitions ----
    defs = [
        "Crop signal: score of the image's own-crop prompts minus the other crops' prompts.   "
        "Disease signal: score of the right-disease prompt minus the wrong-disease prompts of the same crop.",
        "Within-crop wins: share of images whose correct prompt beats every other prompt of its crop.   "
        f"Disease signal v1 \u2192 v3: +{(m['v3']['disease_signal'] / m['v1']['disease_signal'] - 1) * 100:.0f}%; "
        f"crop signal: +{(m['v3']['crop_signal'] / m['v1']['crop_signal'] - 1) * 100:.0f}%.",
    ]
    for k, s in enumerate(defs):
        t(30, ay + ah + 30 + k * 19, s, 11, "start", color=GREY)
    return c


def main():
    OUT.mkdir(exist_ok=True)
    m = measure()
    print(json.dumps(m, indent=1))
    c = draw(m)
    svg = OUT / "fig_b_embedding.svg"
    svg.write_text(c.svg(), encoding="utf-8")
    (OUT / "fig_b_embedding.drawio").write_text(c.drawio("b_embedding"), encoding="utf-8")
    if EDGE.exists():
        png = OUT / "fig_b_embedding.png"
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
