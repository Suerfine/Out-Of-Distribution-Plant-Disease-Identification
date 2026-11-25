"""One compact figure: the mechanism behind each step of the VLM progression (report sections B-I), one mini-schematic each.

Run from the repo root:  .venv/Scripts/python docs/vlm/diagrams/make_mechanisms_figure.py
Writes docs/vlm/diagrams/results/fig_mechanisms.{svg,drawio,png}. Schematic only (no measured positions); the numbers
behind each panel are in the progression figure and the results table.
Style as the other report figures: square corners, black outlines, soft fills. Encodings used in every panel:
  square = image feature, filled dot = text prompt / class, hollow dot = synthetic (generated) feature,
  grey = pretrained CLIP (frozen), blue = trained in this project, orange = ConvNeXt, green = calibration.
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_b_embedding_figure import EDGE, OUT, Canvas  # noqa: E402

BLACK, GREY = "#000000", "#595959"
FROZEN, TRAINED, CNN, CAL, IMG = "#E7E6E6", "#BDD7EE", "#F8CBAD", "#C5E0B4", "#FFFFFF"
PW, PH, GAP, M, TOP = 320, 250, 14, 20, 70


class Panel:
    """Drawing helpers in panel-local coordinates."""

    def __init__(self, c: Canvas, col: int, row: int, title: str, caption: list[str]):
        self.c, self.x, self.y = c, M + col * (PW + GAP), TOP + row * (PH + GAP)
        c.rect(self.x, self.y, PW, PH)
        c.text(self.x + 10, self.y + 20, title, 13, "start", weight="bold")
        for k, line in enumerate(caption):
            c.text(self.x + 10, self.y + PH - 26 + 14 * k, line, 10, "start", color=GREY)

    def box(self, x, y, w, h, label, fill=IMG, size=10, bold=False):
        self.c.rect(self.x + x, self.y + y, w, h, fill)
        lines = label.split("\n")
        for k, ln in enumerate(lines):
            self.c.text(self.x + x + w / 2, self.y + y + h / 2 + 4 + (k - (len(lines) - 1) / 2) * 12, ln, size,
                        weight="bold" if bold else "normal")

    def img(self, x, y, fill=IMG, dashed=False):
        self.c.rect(self.x + x - 5, self.y + y - 5, 10, 10, fill, BLACK, 1.0, dashed=dashed)

    def dot(self, x, y, fill=BLACK, r=4.5, dashed=False):
        self.c.circle(self.x + x, self.y + y, r, fill, BLACK, 1.0, dashed=dashed)

    def arrow(self, pts, dashed=False, color=BLACK, sw=1.0):
        self.c.line([(self.x + a, self.y + b) for a, b in pts], color, sw, arrow=True, dashed=dashed)

    def line(self, pts, dashed=False, color=BLACK, sw=1.0):
        self.c.line([(self.x + a, self.y + b) for a, b in pts], color, sw, dashed=dashed)

    def t(self, x, y, s, size=10, anchor="middle", **kw):
        self.c.text(self.x + x, self.y + y, s, size, anchor, **kw)


def draw() -> Canvas:
    W = 2 * M + 4 * PW + 3 * GAP
    H = TOP + 2 * PH + GAP + 56
    c = Canvas(W, H)
    c.rect(0, 0, W, H, "#FFFFFF", "#FFFFFF", sw=0)
    c.text(M, 32, "Mechanism of each step in the VLM progression", 17, "start", weight="bold")
    c.text(M, 54, "Schematic. □ image feature   ● prompt / class   ○ synthetic feature.", 12, "start", color=GREY)

    # ---- B: disease descriptions ----
    p = Panel(c, 0, 0, "B  Disease descriptions", ["Unique symptom words pull same-crop prompts apart,", "towards visual concepts CLIP already knows."])
    p.t(80, 50, "name only", 10, weight="bold")
    p.c.circle(p.x + 77, p.y + 111, 22, "none", BLACK, 0.8, dashed=True)
    p.dot(70, 110); p.dot(84, 104); p.dot(78, 120)
    p.t(77, 152, "early / late / leaf blight", 9)
    p.t(77, 164, "cos 0.875", 9, color=GREY)
    p.arrow([(115, 111), (150, 111)])
    p.t(232, 50, "+ symptom sentence", 10, weight="bold")
    for (x, y, lab) in ((205, 78, "rings, yellow halo"), (262, 115, "water-soaked patches"), (200, 150, "olive mould")):
        p.dot(x, y)
        p.t(x, y - 9, lab, 9, color=GREY)
    p.t(232, 178, "cos 0.842", 9, color=GREY)

    # ---- C: CSP ----
    p = Panel(c, 1, 0, "C  CSP prompt learning", ["Crop and disease vectors move to where seen images lie;", "unseen pairs reuse them, but seen scores now dominate."])
    p.box(14, 40, 176, 22, "“a photo of [crop] leaves with [disease]”", TRAINED, 9)
    for x, y in ((210, 105), (222, 96), (230, 112), (216, 118), (238, 100)):
        p.img(x, y)
    p.t(226, 140, "seen-pair images", 9, color=GREY)
    p.dot(110, 125, "#FFFFFF")
    p.t(110, 145, "name init", 9, color=GREY)
    p.arrow([(117, 122), (213, 106)])
    p.dot(222, 106, TRAINED)
    p.t(165, 104, "training", 9, color=GREY)
    p.dot(250, 165, TRAINED, dashed=True)
    p.t(250, 185, "unseen pair = learned", 9, color=GREY)
    p.t(250, 196, "crop + disease vectors", 9, color=GREY)

    # ---- D: LoRA ----
    p = Panel(c, 2, 0, "D  LoRA image encoder", ["Small adapters re-shape CLIP's image features for leaves:", "each image moves to its own prompt, away from others."])
    p.box(14, 44, 120, 40, "CLIP image encoder\n(frozen)", FROZEN, 9)
    p.box(98, 72, 50, 20, "+ LoRA", TRAINED, 9, bold=True)
    p.t(74, 106, "1.6M trained params", 9, color=GREY)
    p.dot(220, 70); p.t(220, 60, "early blight", 9)
    p.dot(290, 112); p.t(290, 102, "late blight", 9)
    p.img(250, 150, "#FFFFFF", dashed=True); p.t(250, 172, "before", 9, color=GREY)
    p.arrow([(246, 144), (226, 82)])
    p.img(222, 86, TRAINED); p.t(196, 100, "after", 9, color=GREY)
    p.line([(250, 150), (286, 117)], dashed=True, color=GREY)
    p.line([(222, 86), (284, 110)], dashed=True, color=GREY)

    # ---- E: f-CLSWGAN ----
    p = Panel(c, 3, 0, "E  f-CLSWGAN", ["A GAN turns unseen pair text into synthetic features, so", "the classifier gets examples of every pair (less bias)."])
    p.box(14, 44, 82, 34, "pair text\n+ noise", IMG, 9)
    p.box(116, 44, 52, 34, "G", TRAINED, 12, bold=True)
    p.arrow([(96, 61), (116, 61)])
    p.arrow([(168, 61), (200, 82)])
    for x, y in ((205, 95), (222, 88), (214, 108), (230, 102)):
        p.dot(x, y, "#FFFFFF")
    p.t(218, 126, "synthetic unseen", 9, color=GREY)
    for x, y in ((268, 150), (282, 140), (276, 162), (292, 155)):
        p.img(x, y)
    p.t(280, 182, "real seen", 9, color=GREY)
    p.line([(190, 170), (300, 70)], dashed=True)
    p.t(100, 150, "classifier boundary", 9, color=GREY)
    p.line([(150, 147), (230, 133)], color=GREY, sw=0.6)

    # ---- F: crop-anchored GAN ----
    p = Panel(c, 0, 1, "F  Crop-anchored GAN", ["Start from a real healthy leaf of the crop and learn only", "the disease offset: synthetic features land near real ones."])
    p.img(60, 140, IMG)
    p.t(60, 162, "real healthy", 9, color=GREY)
    p.t(60, 173, "tomato leaf", 9, color=GREY)
    p.arrow([(66, 134), (186, 74)], sw=1.4)
    p.t(80, 82, "+ learned", 9, color=GREY)
    p.t(80, 93, "disease offset", 9, color=GREY)
    p.dot(194, 70, "#FFFFFF")
    p.img(210, 62, IMG, dashed=True)
    p.t(232, 50, "real tomato early blight", 9, color=GREY)
    p.dot(255, 150, "#FFFFFF", dashed=True)
    p.t(255, 170, "E: text + noise only", 9, color=GREY)
    p.t(255, 181, "(lands farther away)", 9, color=GREY)

    # ---- G: disease-only head ----
    p = Panel(c, 1, 1, "G  Disease-only head", ["Score = ConvNeXt crop + CLIP disease. Each part is learned", "from real images; an unseen pair is just a new combination."])
    p.box(14, 52, 90, 40, "ConvNeXt\ncrop score", CNN, 10)
    p.t(115, 76, "+", 16, weight="bold")
    p.box(126, 52, 104, 40, "CLIP disease head\n(175 diseases)", TRAINED, 9)
    p.t(242, 76, "=", 16, weight="bold")
    p.box(254, 52, 56, 40, "pair\nscore", IMG, 10)
    p.t(59, 120, "“tomato”", 10, weight="bold")
    p.t(59, 134, "seen on other tomato pairs", 9, color=GREY)
    p.t(178, 120, "“early blight”", 10, weight="bold")
    p.t(178, 134, "seen on potato (real images)", 9, color=GREY)
    p.t(282, 120, "tomato", 10, weight="bold")
    p.t(282, 134, "early blight", 10, weight="bold")
    p.t(282, 148, "(unseen)", 9, color=GREY)

    # ---- H: seen handoff ----
    p = Panel(c, 2, 1, "H  ConvNeXt seen handoff", ["The router keeps G's seen/unseen decision; seen images", "go to the ConvNeXt, the stronger seen classifier."])
    p.box(14, 82, 60, 34, "image", IMG, 10)
    p.box(96, 76, 88, 46, "seen or\nunseen?", CAL if False else IMG, 10, bold=True)
    p.arrow([(74, 99), (96, 99)])
    p.arrow([(184, 90), (214, 62)])
    p.arrow([(184, 108), (214, 136)])
    p.box(214, 42, 96, 36, "ConvNeXt\ncrop + disease", CNN, 9)
    p.box(214, 118, 96, 36, "CLIP + GAN\n(as G)", TRAINED, 9)
    p.t(190, 66, "seen", 9, "end", color=GREY)
    p.t(190, 140, "unseen", 9, "end", color=GREY)

    # ---- I: calibrated stacking ----
    p = Panel(c, 3, 1, "I  Calibrated stacking (γ)", ["Subtract γ from every seen-pair score: borderline", "images switch to the best unseen pair."])
    base = 170
    p.line([(30, base), (300, base)])
    p.box(50, base - 100, 60, 100, "best\nseen\npair", CNN, 9)
    p.box(190, base - 78, 60, 78, "best\nunseen\npair", TRAINED, 9)
    p.c.rect(p.x + 120, p.y + base - 100, 60, 100, "#FFFFFF", BLACK, 1.0, dashed=True)
    p.c.rect(p.x + 120, p.y + base - 64, 60, 64, CAL, BLACK, 1.0)
    p.t(150, base - 22, "seen − γ", 9, weight="bold")
    p.arrow([(150, base - 98), (150, base - 68)])
    p.t(160, base - 82, "γ", 11, "start", weight="bold")
    p.line([(30, base - 78), (300, base - 78)], dashed=True, color=GREY)
    p.t(298, base - 82, "unseen now wins", 9, "end", color=GREY)

    # ---- legend ----
    ly, lx = TOP + 2 * PH + GAP + 30, M
    for fill, lab in ((FROZEN, "pretrained CLIP (frozen)"), (TRAINED, "trained in this project"), (CNN, "ConvNeXt"),
                      (CAL, "calibration")):
        c.rect(lx, ly - 10, 22, 12, fill)
        c.text(lx + 28, ly, lab, 11, "start")
        lx += 28 + len(lab) * 6.2 + 26
    return c


def main():
    c = draw()
    svg = OUT / "fig_mechanisms.svg"
    svg.write_text(c.svg(), encoding="utf-8")
    (OUT / "fig_mechanisms.drawio").write_text(c.drawio("mechanisms"), encoding="utf-8")
    if EDGE.exists():
        png = OUT / "fig_mechanisms.png"
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
