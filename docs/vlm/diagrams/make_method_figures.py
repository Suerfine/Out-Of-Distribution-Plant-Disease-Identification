"""Method-section figures for the report, in the style of the symptom-description / CSP / LoRA figures:
white boxes, black outlines, bold title line, dashed outline = learned / trainable, solid = fixed / frozen.

Run from the repo root:  .venv/Scripts/python docs/vlm/diagrams/make_method_figures.py
Writes docs/vlm/diagrams/results/method_<name>.{svg,drawio,png} for: vlm, gan (E), anchor (F), disease_head (G),
handoff (H), calibration (I).
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_b_embedding_figure import EDGE, OUT, Canvas  # noqa: E402

BLACK = "#000000"
W = 1090


class Fig:
    def __init__(self, h):
        self.c = Canvas(W, h)
        self.c.rect(0, 0, W, h, "#FFFFFF", "#FFFFFF", sw=0)

    def box(self, x, y, w, h, title, sub="", dashed=False, size=14):
        """Box with a bold first line and optional normal lines below (sub may contain '\n')."""
        self.c.rect(x, y, w, h, "#FFFFFF", BLACK, 1.5, dashed=dashed)
        lines = [(title, "bold")] + [(s, "normal") for s in sub.split("\n") if s]
        n = len(lines)
        for k, (s, wt) in enumerate(lines):
            self.c.text(x + w / 2, y + h / 2 + 5 + (k - (n - 1) / 2) * (size + 3), s, size if wt == "bold" else size - 1,
                        weight=wt)
        return (x, y, w, h)

    def arrow(self, pts, label="", lx=0, ly=0, anchor="middle"):
        self.c.line(pts, BLACK, 1.5, arrow=True)
        if label:
            self.c.text(lx, ly, label, 12, anchor)

    def text(self, x, y, s, size=12, anchor="start", weight="normal", italic=False):
        self.c.text(x, y, s, size, anchor, weight=weight, italic=italic)

    def legend(self, y, frozen_label="fixed / frozen", learned=True):
        x = 10
        if learned:
            self.c.rect(x, y - 10, 30, 14, "#FFFFFF", BLACK, 1.2, dashed=True)
            self.text(x + 36, y + 2, "learned / trainable", 12)
            x += 210
        self.c.rect(x, y - 10, 30, 14, "#FFFFFF", BLACK, 1.2)
        self.text(x + 36, y + 2, frozen_label, 12)


def mid_r(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def mid_l(b):
    x, y, w, h = b
    return (x, y + h / 2)


# ---------------------------------------------------------------- VLM (zero-shot CLIP)
def fig_vlm():
    f = Fig(250)
    img = f.box(10, 10, 190, 56, "Leaf image")
    ienc = f.box(240, 10, 190, 56, "Image encoder", "ViT-B/16")
    iemb = f.box(470, 10, 180, 56, "Image embedding v", "512-d")
    prm = f.box(10, 130, 190, 56, "333 class prompts", "one per crop–disease pair")
    tenc = f.box(240, 130, 190, 56, "Text encoder", "Transformer")
    temb = f.box(470, 130, 180, 56, "333 text embeddings t", "512-d each")
    cos = f.box(700, 70, 200, 56, "Cosine similarity", "v · t → 333 scores")
    pred = f.box(940, 70, 140, 56, "Prediction", "Top-1 / Top-5")
    for a, b in ((img, ienc), (ienc, iemb), (prm, tenc), (tenc, temb), (cos, pred)):
        f.arrow([mid_r(a), mid_l(b)])
    f.arrow([mid_r(iemb), (675, 38), (675, 88), (700, 88)])
    f.arrow([mid_r(temb), (675, 158), (675, 108), (700, 108)])
    f.text(10, 214, "e.g. “a photo of tomato leaves with early blight”. Seen and unseen pairs are scored the same way: "
                    "an unseen pair needs only a prompt, not training images.", 12, italic=True)
    f.legend(240, "pretrained CLIP, frozen (zero-shot)", learned=False)
    return f.c


# ---------------------------------------------------------------- E: f-CLSWGAN
def fig_gan():
    f = Fig(304)
    f.text(10, 18, "1  Train on seen pairs", 13, weight="bold")
    t1 = f.box(10, 30, 180, 56, "Pair text embedding", "+ random noise z")
    g1 = f.box(230, 30, 150, 56, "Generator G", dashed=True)
    s1 = f.box(420, 30, 170, 56, "Synthetic feature", "x̃")
    d1 = f.box(630, 30, 210, 56, "Critic D", "real or synthetic?", dashed=True)
    r1 = f.box(880, 30, 200, 56, "Real image feature x", "seen pairs only")
    for a, b in ((t1, g1), (g1, s1), (s1, d1)):
        f.arrow([mid_r(a), mid_l(b)])
    f.arrow([mid_l(r1), mid_r(d1)])
    f.text(10, 130, "2  Synthesise unseen pairs, train the classifier", 13, weight="bold")
    t2 = f.box(10, 142, 180, 56, "Unseen pair text", "+ random noise z")
    g2 = f.box(230, 142, 150, 56, "Generator G", "(trained in 1)")
    s2 = f.box(420, 142, 170, 56, "300 synthetic features", "per unseen pair")
    c2 = f.box(630, 142, 210, 56, "Classifier", "333 pairs", dashed=True)
    p2 = f.box(880, 142, 200, 56, "Prediction")
    for a, b in ((t2, g2), (g2, s2), (s2, c2), (c2, p2)):
        f.arrow([mid_r(a), mid_l(b)])
    rs = f.box(630, 222, 210, 40, "Real seen features")
    f.arrow([(735, 222), (735, 198)])
    f.legend(292)
    return f.c


# ---------------------------------------------------------------- F: crop-anchored GAN
def fig_anchor():
    f = Fig(230)
    anc = f.box(10, 10, 240, 56, "Anchor a", "real healthy feature of the target crop")
    txt = f.box(10, 120, 240, 56, "Pair text embedding", "+ random noise z")
    gen = f.box(300, 120, 230, 56, "Generator G", "outputs disease offset Δ", dashed=True)
    plus = f.box(580, 65, 50, 46, "+")
    syn = f.box(680, 60, 220, 56, "Synthetic feature", "x̃ = normalise(a + Δ)")
    cls = f.box(940, 60, 140, 56, "Classifier", "as in E", dashed=True)
    f.arrow([mid_r(txt), mid_l(gen)])
    f.arrow([mid_r(anc), (605, 38), (605, 65)])
    f.arrow([mid_r(gen), (605, 148), (605, 111)])
    f.arrow([mid_r(plus), mid_l(syn)])
    f.arrow([mid_r(syn), mid_l(cls)])
    f.text(10, 200, "e.g. tomato early blight: a = a real healthy tomato leaf, Δ = the change early blight makes. "
                    "The crop and leaf come from a real image; G only models the disease.", 12, italic=True)
    f.legend(222)
    return f.c


# ---------------------------------------------------------------- G: disease-only head
def fig_disease_head():
    f = Fig(250)
    img = f.box(10, 70, 150, 56, "Leaf image")
    cnn = f.box(210, 10, 230, 56, "ConvNeXt-Base", "crop classifier (CNN baseline)")
    clip = f.box(210, 130, 230, 56, "CLIP image encoder", "+ LoRA (frozen)")
    cs = f.box(480, 10, 190, 56, "Crop score", "55 crops")
    dh = f.box(480, 130, 190, 56, "Disease head", "175 diseases", dashed=True)
    plus = f.box(710, 75, 50, 46, "+")
    ps = f.box(800, 70, 280, 56, "Pair score", "crop score + disease score (333 pairs)")
    f.arrow([mid_r(img), (185, 98), (185, 38), (210, 38)])
    f.arrow([(185, 98), (185, 158), (210, 158)])
    f.arrow([mid_r(cnn), mid_l(cs)])
    f.arrow([mid_r(clip), mid_l(dh)])
    f.arrow([mid_r(cs), (735, 38), (735, 75)])
    f.arrow([mid_r(dh), (735, 158), (735, 121)])
    f.arrow([mid_r(plus), mid_l(ps)])
    f.text(480, 206, "trained on real features + crop-anchored synthetic features (F)", 12, italic=True)
    f.legend(240)
    return f.c


# ---------------------------------------------------------------- H: ConvNeXt seen handoff
def fig_handoff():
    f = Fig(230)
    sc = f.box(10, 70, 170, 56, "Pair scores", "from G")
    rt = f.box(220, 70, 240, 56, "Router", "best seen − best unseen ≥ γ ?")
    se = f.box(530, 10, 300, 56, "Seen expert: ConvNeXt", "crop + disease, 323 seen pairs")
    ue = f.box(530, 130, 300, 56, "Unseen expert: G's pair score", "10 unseen pairs")
    pr = f.box(890, 70, 190, 56, "Prediction")
    f.arrow([mid_r(sc), mid_l(rt)])
    f.arrow([mid_r(rt), (495, 98), (495, 38), (530, 38)], "yes: seen", 488, 30, "end")
    f.arrow([(495, 98), (495, 158), (530, 158)], "no: unseen", 488, 176, "end")
    f.arrow([mid_r(se), (860, 38), (860, 88), (890, 88)])
    f.arrow([mid_r(ue), (860, 158), (860, 108), (890, 108)])
    f.text(10, 214, "No new training: G's routing and unseen predictions are kept; only the seen side is relabelled by the ConvNeXt.",
           12, italic=True)
    return f.c


# ---------------------------------------------------------------- I: calibrated stacking
def fig_calibration():
    f = Fig(230)
    sc = f.box(10, 10, 210, 56, "333 pair scores", "s(x, c)")
    sub = f.box(260, 10, 260, 56, "Subtract γ", "from the 323 seen pairs only")
    am = f.box(560, 10, 160, 56, "argmax")
    pr = f.box(760, 10, 160, 56, "Prediction")
    val = f.box(260, 120, 260, 56, "Validation split", "held-out seen pairs act as unseen")
    for a, b in ((sc, sub), (sub, am), (am, pr)):
        f.arrow([mid_r(a), mid_l(b)])
    f.arrow([(390, 120), (390, 66)], "choose γ that maximises HM", 400, 98, "start")
    f.text(560, 150, "ŷ = argmax c [ s(x, c) − γ · 1(c is seen) ]", 14, weight="bold")
    f.text(10, 214, "Unseen-pair scores are unchanged. γ is set on validation, never on the test set.", 12, italic=True)
    return f.c


FIGS = {"vlm": fig_vlm, "gan": fig_gan, "anchor": fig_anchor, "disease_head": fig_disease_head,
        "handoff": fig_handoff, "calibration": fig_calibration}


def render(c, name):
    svg = OUT / f"method_{name}.svg"
    svg.write_text(c.svg(), encoding="utf-8")
    (OUT / f"method_{name}.drawio").write_text(c.drawio(name), encoding="utf-8")
    if not EDGE.exists():
        return
    png = OUT / f"method_{name}.png"
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


def main():
    only = sys.argv[1:]
    for name, make in FIGS.items():
        if not only or name in only:
            render(make(), name)


if __name__ == "__main__":
    main()
