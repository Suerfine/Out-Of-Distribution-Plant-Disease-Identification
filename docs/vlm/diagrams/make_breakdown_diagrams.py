"""Writes the figures of docs/vlm/VLM_technical_breakdown.docx as draw.io files (docs/vlm/diagrams/breakdown/*.drawio),
one diagram per file, and optionally renders each to PNG with draw.io's own viewer in headless Microsoft Edge.

Run:  python docs/vlm/diagrams/make_breakdown_diagrams.py          (only the .drawio files)
      python docs/vlm/diagrams/make_breakdown_diagrams.py --png    (also PNGs; needs Edge, Pillow and internet access
                                                                    for viewer.diagrams.net)

Colour is used for one meaning only, the same in every figure:
  grey  = pretrained CLIP, frozen          blue  = trained in this project
  green = data going in                    amber = scores / decisions        red = loss (training signal)
"""
import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent / "breakdown"
FONT = "fontFamily=Helvetica;fontSize=12;fontColor=#1B1F24;"
BASE = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;" + FONT
KIND = {
    "frozen": BASE + "fillColor=#F1F3F5;strokeColor=#868E96;",
    "train": BASE + "fillColor=#E7F0FF;strokeColor=#2F6FD0;strokeWidth=2;",
    "data": BASE + "fillColor=#EAF6EC;strokeColor=#4C9A66;",
    "score": BASE + "fillColor=#FFF4D6;strokeColor=#C99A2E;",
    "loss": BASE + "fillColor=#FDECEC;strokeColor=#C9504A;",
    "plain": BASE + "fillColor=#FFFFFF;strokeColor=#495057;",
    "group": "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=none;dashed=1;strokeColor=#ADB5BD;verticalAlign=top;"
             "align=left;spacingLeft=8;spacingTop=2;fontStyle=1;fontFamily=Helvetica;fontSize=12;fontColor=#495057;",
    "text": "text;html=1;whiteSpace=wrap;verticalAlign=middle;" + FONT,
}
LEGEND = {"frozen": "Pretrained CLIP, frozen", "train": "Trained in this project", "data": "Data in",
          "score": "Scores / decisions", "loss": "Loss (training only)"}
SIDE = {"l": (0, 0.5), "r": (1, 0.5), "t": (0.5, 0), "b": (0.5, 1)}


class Diagram:
    def __init__(self, name, title):
        self.name, self.cells, self.n = name, [], 0
        title = title.split(". ", 1)[1]  # the figure number belongs to the caption of whatever document uses the figure
        title = title[0].upper() + title[1:]
        self.text(f"<b>{title}</b>", 40, 8, 900, 30, size=15, align="left")

    def _add(self, value, style, x, y, w, h):
        self.n += 1
        cid = f"c{self.n}"
        self.cells.append(f'<mxCell id="{cid}" value="{html.escape(value)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def box(self, value, kind, x, y, w, h, extra=""):
        return self._add(value, KIND[kind] + extra, x, y, w, h)

    def text(self, value, x, y, w, h, size=12, align="center", color="#1B1F24"):
        return self._add(value, KIND["text"] + f"fontSize={size};align={align};fontColor={color};", x, y, w, h)

    def edge(self, a, b, ex=None, en=None, label="", dashed=False, color="#495057", arrow=True):
        """ex / en: side ('l','r','t','b') or an (x, y) fraction pair on the source / target box."""
        self.n += 1
        st = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor={color};fontFamily=Helvetica;fontSize=11;"
              f"fontColor=#495057;labelBackgroundColor=#FFFFFF;endArrow={'classic' if arrow else 'none'};")
        if dashed:
            st += "dashed=1;"
        for key, side in (("exit", ex), ("entry", en)):
            if side is not None:
                fx, fy = SIDE.get(side, side)
                st += f"{key}X={fx};{key}Y={fy};{key}Dx=0;{key}Dy=0;"
        self.cells.append(f'<mxCell id="e{self.n}" value="{html.escape(label)}" style="{st}" edge="1" parent="1" source="{a}" '
                          f'target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>')

    def legend(self, y, kinds, x=40):
        for k in kinds:
            w = 30 + 7 * len(LEGEND[k])
            self.box(LEGEND[k], k, x, y, w, 26, "fontSize=11;")
            x += w + 12

    def model(self):
        return ('<mxGraphModel grid="0" page="0" background="#FFFFFF"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                + "".join(self.cells) + "</root></mxGraphModel>")

    def mxfile(self):
        return f'<mxfile><diagram name="{html.escape(self.name)}">{self.model()}</diagram></mxfile>'


def fig01_pipeline():
    d = Diagram("fig01_pipeline", "Figure 1. The VLM pipeline: one image is scored against 333 crop-disease prompts")
    photo = d.box("<b>Plant photo</b><br>224 x 224 pixels", "data", 40, 80, 150, 60)
    lora = d.box("<b>LoRA adapters</b> in blocks 5-12<br>1.57M numbers", "train", 240, 170, 230, 46)
    enc = d.box("<b>CLIP image encoder</b> ViT-B/16<br>12 blocks, 86M weights", "frozen", 240, 80, 230, 60)
    v = d.box("<b>Image embedding v</b><br>512 numbers, length 1", "plain", 520, 80, 170, 60)
    prompts = d.box("<b>333 pair prompts</b><br>[CTX] [CTX] [CTX] [CROP] leaves with [DISEASE]: symptoms.", "data", 40, 330, 150, 80)
    vecs = d.box("<b>Learned word vectors</b><br>55 crops + 175 diseases<br>+ 3 context vectors (CoOp)", "train", 240, 240, 230, 60)
    tenc = d.box("<b>CLIP text encoder</b><br>12 blocks", "frozen", 240, 340, 230, 60)
    t = d.box("<b>333 text embeddings t</b><br>512 numbers each, length 1", "plain", 520, 340, 170, 60)
    loss = d.box("<b>Training only:</b> cross-entropy on 100 x scores (pair + crop + disease)", "loss", 750, 80, 190, 60)
    cos = d.box("<b>Cosine similarity</b><br>v . t for every pair = 333 scores", "score", 750, 210, 190, 60)
    cal = d.box("<b>Calibration</b><br>seen-pair scores minus gamma", "score", 750, 310, 190, 60)
    pred = d.box("<b>Prediction</b><br>highest score (Top-1, Top-5)", "plain", 750, 410, 190, 60)
    d.edge(photo, enc, "r", "l")
    d.edge(lora, enc, "t", "b", label="adds to weights")
    d.edge(enc, v, "r", "l")
    d.edge(prompts, tenc, "r", "l")
    d.edge(vecs, tenc, "b", "t", label="fill [CTX], [CROP], [DISEASE]")
    d.edge(tenc, t, "r", "l")
    d.edge(v, cos, "b", (0, 0.3))
    d.edge(t, cos, "t", (0, 0.7))
    d.edge(cos, loss, "t", "b", dashed=True, color="#C9504A")
    d.edge(cos, cal, "b", "t")
    d.edge(cal, pred, "b", "t")
    d.legend(500, ["frozen", "train", "data", "score", "loss"])
    return d


def fig02_prompts():
    d = Diagram("fig02_prompts", "Figure 2. Zero-shot stage: what changed in the prompt text, and the harmonic mean each prompt reached")
    rows = [
        ("v1<br>raw names", "a photo of tomato leaves with early_blight.", "HM 5.4%"),
        ("v2<br>clean names, 5 templates", "a photo of tomato leaves with early blight.&nbsp; (+ 4 other sentence templates, embeddings averaged)", "HM 5.6%"),
        ("v3<br>symptom description", "a photo of tomato leaves with early blight: <b>brown target spots with concentric rings and yellow halos on older leaves</b>.", "HM 10.2%"),
        ("v3.1<br>control: random text", "a photo of tomato leaves with early blight: <b>random letter strings of the same length</b>.", "HM 5.5%"),
        ("v3.6<br>two separate prompts", "Prompt A: ...early blight: <b>brown target spots with concentric rings...</b><br>Prompt B: ...early blight: <b>Alternaria solani, known as early blight</b>.&nbsp; (two embeddings averaged)", "HM 11.9%"),
    ]
    prev = None
    for i, (ver, prompt, hm) in enumerate(rows):
        y = 56 + i * 84
        a = d.box(f"<b>{ver.split('<br>')[0]}</b><br>{ver.split('<br>')[1]}", "plain", 40, y, 150, 60)
        p = d.box(prompt, "data", 220, y, 560, 60, "align=left;spacingLeft=10;spacingRight=10;")
        s = d.box(f"<b>{hm}</b>", "score", 810, y, 130, 60)
        d.edge(a, p, "r", "l", arrow=False)
        d.edge(p, s, "r", "l")
        if prev:
            d.edge(prev, a, "b", "t")
        prev = a
    d.text("No training in this stage. Only the green text changes; both CLIP encoders are used exactly as downloaded.",
           40, 476, 900, 24, size=11, align="left", color="#495057")
    return d


def fig03_text_encoder():
    d = Diagram("fig03_text_encoder", "Figure 3. Inside the text encoder: where the learned context, crop and disease vectors enter")
    toks = ["[start]", "[ctx 1]", "[ctx 2]", "[ctx 3]", "[CROP]", "leaves", "with", "[DISEASE]", ":", "brown", "target", "spots", "...", "[end]"]
    ids = []
    for i, tk in enumerate(toks):
        kind = "train" if tk.startswith("[ctx") or tk in ("[CROP]", "[DISEASE]") else "data"
        ids.append(d.box(tk, kind, 40 + i * 64, 56, 60, 34, "fontSize=11;"))
    ctx = d.box("<b>Context vectors</b> (CoOp, v5.13)<br>3 x 512, shared by every pair", "train", 40, 136, 188, 50)
    cv = d.box("<b>Crop vector</b> (tomato)<br>512 numbers", "train", 241, 136, 170, 50)
    dv = d.box("<b>Disease vector</b> (early blight)<br>512 numbers", "train", 433, 136, 170, 50)
    table = d.box("Fixed words: looked up in CLIP's<br>word-embedding table", "frozen", 625, 136, 188, 50)
    d.text("Before v5.13 the three context slots hold the fixed words <i>a photo of</i>. One crop vector is shared by every "
           "pair with that crop, one disease vector by every pair with that disease; all start from CLIP's embedding of the words.",
           40, 300, 176, 150, size=11, align="left", color="#495057")
    seq = d.box("Sequence of word vectors (one 512-number vector per token) + position embeddings", "frozen", 40, 232, 896, 36)
    d.edge(ids[2], ctx, "b", "t")
    d.edge(ids[4], cv, "b", "t")
    d.edge(ids[7], dv, "b", "t")
    d.edge(ids[9], table, "b", (0.4, 0))
    d.edge(ctx, seq, "b", (0.105, 0))
    d.edge(cv, seq, "b", (0.3192, 0))
    d.edge(dv, seq, "b", (0.5335, 0))
    d.edge(table, seq, "b", (0.7578, 0))
    blocks = d.box("<b>12 Transformer blocks</b><br>causal self-attention: each token only looks at the tokens to its left", "frozen", 228, 304, 520, 50)
    eot = d.box("Keep the output at <b>[end]</b>: the only position that has read the whole sentence", "plain", 228, 386, 520, 40)
    proj = d.box("Layer norm + linear projection to 512 numbers", "frozen", 228, 458, 520, 36)
    out = d.box("<b>Text embedding t</b>, scaled to length 1", "plain", 228, 526, 520, 36)
    d.edge(seq, blocks, "b", "t")
    d.edge(blocks, eot, "b", "t")
    d.edge(eot, proj, "b", "t")
    d.edge(proj, out, "b", "t")
    d.box("<b>Training:</b> the loss gradient travels back up through the frozen blocks and changes only the blue vectors",
          "loss", 776, 304, 160, 122, "fontSize=11;")
    d.legend(590, ["frozen", "train", "data", "loss"])
    return d


def fig04_composition():
    d = Diagram("fig04_composition", "Figure 4. Why an unseen pair can be scored: its prompt is assembled from vectors trained on other pairs")
    d.box("Seen pairs (have training images)", "group", 30, 50, 280, 330)
    s1 = d.box("tomato + late blight", "data", 50, 84, 240, 40)
    s2 = d.box("tomato + septoria leaf spot", "data", 50, 138, 240, 40)
    s3 = d.box("tomato + ... (other tomato diseases)", "data", 50, 192, 240, 40)
    s4 = d.box("potato + early blight", "data", 50, 300, 240, 40)
    c = d.box("<b>Crop vector: tomato</b><br>trained by 1,164 tomato images", "train", 400, 122, 220, 56)
    dz = d.box("<b>Disease vector: early blight</b><br>trained by 102 early-blight images", "train", 400, 292, 220, 56)
    d.edge(s1, c, "r", (0, 0.25))
    d.edge(s2, c, "r", "l")
    d.edge(s3, c, "r", (0, 0.75))
    d.edge(s4, dz, "r", "l")
    u = d.box("<b>Unseen pair: tomato + early blight</b><br>no training image<br><br>a photo of [tomato vector] leaves with [early blight vector]: "
              "brown target spots with concentric rings...", "score", 700, 180, 240, 110)
    d.edge(c, u, "r", (0, 0.3), label="reused")
    d.edge(dz, u, "r", (0, 0.7), label="reused")
    d.text("Gradients from seen pairs update the two vectors. The unseen prompt needs no image of its own: it reuses both.",
           40, 396, 900, 24, size=11, align="left", color="#495057")
    d.legend(430, ["train", "data", "score"])
    return d


def fig05_loss():
    d = Diagram("fig05_loss", "Figure 5. Training objective: three classification heads share one image embedding")
    v = d.box("<b>Image embedding v</b> (from the image encoder, length 1)", "plain", 40, 56, 640, 40)
    heads = [("Pair head", "100 x cos(v, t) against the<br>seen pair prompts (323)", "true pair", "1.0"),
             ("Crop head", "100 x cos(v, t) against the<br>55 crop prompts", "true crop", "0.5"),
             ("Disease head", "100 x cos(v, t) against the<br>175 disease prompts", "true disease", "0.5")]
    total = d.box("<b>Total loss</b> = 1.0 x pair + 0.5 x crop + 0.5 x disease &nbsp;(+ 25 x text anchor from v10.1 on)", "loss", 40, 372, 640, 40)
    for i, (name, body, target, w) in enumerate(heads):
        x = 40 + i * 220
        s = d.box(f"<b>{name}</b><br>{body}", "score", x, 136, 200, 70)
        ce = d.box(f"Softmax + cross-entropy<br>with the {target}<br><b>weight {w}</b>", "loss", x, 250, 200, 70)
        d.edge(v, s, ((x + 100 - 40) / 640, 1), "t")
        d.edge(s, ce, "b", "t")
        d.edge(ce, total, "b", ((x + 100 - 40) / 640, 0))
    upd = d.box("<b>Updated by the gradient</b><br>crop and disease vectors<br>LoRA matrices A and B", "train", 740, 356, 200, 72)
    d.edge(total, upd, "r", "l", dashed=True, color="#C9504A")
    d.text("t = text embeddings from the text encoder. The crop and disease heads force each vector to work on its own, "
           "not only inside the pairs it was trained with.", 740, 136, 200, 110, size=11, align="left", color="#495057")
    d.legend(450, ["train", "score", "loss"])
    return d


def fig06_lora():
    d = Diagram("fig06_lora", "Figure 6. LoRA: the pretrained weight stays frozen, a small trained side path is added to it")
    a = d.box("Image cut into 196 patches<br>+ 1 class token", "frozen", 40, 60, 220, 50)
    b = d.box("<b>Blocks 1-4</b><br>frozen, no adapter", "frozen", 40, 146, 220, 50)
    c = d.box("<b>Blocks 5-12</b><br>frozen weights + LoRA", "train", 40, 232, 220, 50)
    e = d.box("Class token, projection to 512<br>= image embedding v", "plain", 40, 318, 220, 50)
    d.edge(a, b, "b", "t")
    d.edge(b, c, "b", "t")
    d.edge(c, e, "b", "t")
    d.box("Inside one adapted weight matrix (4 per block: attention q/k/v, attention output, MLP in, MLP out)", "group", 320, 50, 620, 318)
    x = d.box("input x", "plain", 340, 190, 80, 44)
    w = d.box("<b>Pretrained weight W</b> (frozen)<br>for example 768 x 768", "frozen", 480, 100, 270, 50)
    la = d.box("<b>A</b>: 16 x 768<br>random start", "train", 480, 274, 120, 50)
    lb = d.box("<b>B</b>: 768 x 16<br>starts at zero", "train", 630, 274, 120, 50)
    plus = d.box("<b>+</b>", "plain", 800, 192, 40, 40, "ellipse;rounded=0;fontSize=16;")
    h = d.box("output", "plain", 860, 190, 66, 44)
    d.edge(x, w, "t", "l")
    d.edge(x, la, "b", "l")
    d.edge(la, lb, "r", "l")
    d.edge(w, plus, "r", "t")
    d.edge(lb, plus, "r", "b")
    d.edge(plus, h, "r", "l")
    d.edge(c, x, "r", "l", dashed=True, arrow=False, color="#ADB5BD")
    d.text("output = W x + 2 x B (A x), where 2 = alpha / rank.&nbsp; B starts at zero, so training begins exactly at pretrained CLIP. "
           "Rank 16 means the update can only change the weight along 16 directions.", 480, 168, 290, 76, size=11, align="left", color="#495057")
    d.text("Trained: 1.57M numbers (8 blocks, rank 16) against 86M frozen in the encoder, about 1.8%.",
           40, 384, 900, 22, size=11, align="left", color="#495057")
    d.legend(418, ["frozen", "train"])
    return d


def fig07_calibration():
    d = Diagram("fig07_calibration", "Figure 7. Calibration: subtracting gamma from seen-pair scores lets an unseen pair win")
    for px, title, seen_w, seen_lab, verdict, vkind in (
            (40, "Before calibration (gamma = 0)", 310, "0.31", "Prediction: tomato late blight (wrong)", "loss"),
            (500, "After calibration (gamma = 0.07)", 240, "0.31 - 0.07 = 0.24", "Prediction: tomato early blight (correct)", "score")):
        d.box(title, "group", px, 50, 440, 200)
        d.text("Best seen pair<br>tomato late blight", px + 10, 84, 110, 40, size=11, align="left")
        d.box(seen_lab, "frozen", px + 120, 88, seen_w, 32, "rounded=0;fontSize=11;")
        d.text("Best unseen pair<br>tomato early blight", px + 10, 138, 110, 40, size=11, align="left")
        d.box("0.27", "score", px + 120, 142, 270, 32, "rounded=0;fontSize=11;")
        d.box(verdict, vkind, px + 10, 198, 420, 34, "fontSize=11;")
    d.text("<b>Example image: a tomato early blight leaf (an unseen pair).</b> Bars are cosine scores. The model has only ever been "
           "rewarded for seen pairs, so their scores run high.", 40, 262, 900, 36, size=11, align="left", color="#495057")
    hdr = ["v5.5 on the test sets", "Seen Top-1", "Unseen Top-1", "Harmonic mean"]
    rows = [["gamma = 0 (no calibration)", "67.6%", "28.7%", "40.3%"], ["gamma = 0.075 (chosen on validation)", "52.2%", "84.3%", "64.5%"]]
    xs, ws = [40, 380, 560, 740], [330, 170, 170, 200]
    for j, (xx, ww) in enumerate(zip(xs, ws)):
        d.box(f"<b>{hdr[j]}</b>", "plain", xx, 312, ww, 30, "rounded=0;fontSize=11;")
        for i, r in enumerate(rows):
            d.box(r[j], "score" if j == 3 else "plain", xx, 342 + i * 30, ww, 30, "rounded=0;fontSize=11;")
    d.text("A larger gamma rescues unseen images but also moves some seen images to unseen pairs. Gamma is the single value "
           "that gives the best harmonic mean on validation.", 40, 412, 900, 36, size=11, align="left", color="#495057")
    return d


def fig08_validation():
    d = Diagram("fig08_validation", "Figure 8. Model selection without touching the test sets: pseudo-unseen validation, then refit")
    tr = d.box("<b>Training set</b>: 8,000 images, 323 seen pairs", "data", 240, 56, 500, 40)
    fit = d.box("<b>Fit set</b><br>the remaining images of 303 pairs", "data", 40, 140, 270, 56)
    vs = d.box("<b>Val-seen</b><br>15% of the images of those 303 pairs", "data", 355, 140, 270, 56)
    vu = d.box("<b>Pseudo-unseen</b><br>20 whole pairs held out (max 40 images each)", "data", 670, 140, 270, 56)
    d.edge(tr, fit, (0.1, 1), "t")
    d.edge(tr, vs, "b", "t")
    d.edge(tr, vu, (0.9, 1), "t")
    s1 = d.box("<b>Stage 1</b>: train on the fit set. After every epoch score val-seen + pseudo-unseen against the pair prompts "
               "and try every gamma.", "plain", 40, 240, 900, 44)
    d.edge(fit, s1, "b", (0.15, 0))
    d.edge(vs, s1, "b", (0.5, 0))
    d.edge(vu, s1, "b", (0.85, 0))
    ch = d.box("<b>Chosen: epoch and gamma</b> with the best validation harmonic mean (averaged over 3 different splits from v5.8 on)",
               "score", 40, 324, 900, 40)
    s2 = d.box("<b>Stage 2</b>: train a fresh model on all 8,000 images for the chosen number of epochs", "plain", 40, 404, 900, 40)
    te = d.box("<b>Test once</b>: 2,000 seen-pair images + 1,448 unseen-pair images, ranked against all 333 pairs with the chosen gamma",
               "data", 40, 484, 900, 40)
    d.edge(s1, ch, "b", "t")
    d.edge(ch, s2, "b", "t")
    d.edge(s2, te, "b", "t")
    d.text("A pseudo-unseen pair is a real seen pair whose images are all removed from training, while its crop and its disease still "
           "appear in other training pairs. It imitates a true unseen pair.", 40, 536, 900, 36, size=11, align="left", color="#495057")
    return d


def fig09_feature_gen():
    d = Diagram("fig09_feature_gen", "Figure 9. Feature generation (v11): invent image embeddings for the unseen pairs from their text")
    d.box("Step 1. Learn text-to-image-embedding on the seen pairs", "group", 30, 50, 910, 170)
    c1 = d.box("Seen pair's text embedding + random noise", "data", 50, 110, 200, 56)
    g1 = d.box("<b>Generator</b><br>2-layer network", "train", 300, 110, 160, 56)
    f1 = d.box("Fake image embedding", "plain", 510, 84, 170, 44)
    r1 = d.box("Real image embedding of that pair", "data", 510, 150, 170, 44)
    cr = d.box("<b>Critic</b>: real or fake?<br>+ still classified as its pair?", "loss", 740, 104, 180, 68)
    d.edge(c1, g1, "r", "l")
    d.edge(g1, f1, "r", "l")
    d.edge(f1, cr, "r", (0, 0.3))
    d.edge(r1, cr, "r", (0, 0.7))
    d.box("Step 2. Generate for the unseen pairs, then train a classifier over all 333 pairs", "group", 30, 240, 910, 150)
    c2 = d.box("Unseen pair's text embedding + noise", "data", 50, 290, 200, 56)
    g2 = d.box("<b>Generator</b><br>(from step 1)", "train", 300, 290, 160, 56)
    f2 = d.box("300 synthetic image embeddings per unseen pair", "score", 510, 290, 170, 56)
    cl = d.box("<b>Cosine classifier</b><br>real seen + synthetic unseen", "train", 740, 290, 180, 56)
    d.edge(c2, g2, "r", "l")
    d.edge(g2, f2, "r", "l")
    d.edge(f2, cl, "r", "l")
    d.text("No unseen image is used. The classifier now has training examples for every pair, so it is less biased toward seen pairs.",
           40, 400, 900, 24, size=11, align="left", color="#495057")
    d.legend(434, ["train", "data", "score", "loss"])
    return d


def fig10_ensemble():
    d = Diagram("fig10_ensemble", "Figure 10. Ensemble (v6.7): three different models are put on one scale, then averaged")
    a1 = d.box("<b>CLIP + LoRA, CoOp + CSP (v5.13)</b><br>333 cosine scores", "train", 40, 60, 200, 56)
    a2 = d.box("x 100, then log-softmax over the 333 pairs", "plain", 290, 60, 220, 56)
    b1 = d.box("<b>ConvNeXt 224 px: crop</b><br>log P(crop)", "train", 40, 146, 200, 42)
    b2 = d.box("<b>ConvNeXt 224 px: disease</b><br>log P(disease)", "train", 40, 194, 200, 42)
    b3 = d.box("log P(crop) + log P(disease), then log-softmax over the 333 pairs", "plain", 290, 150, 220, 82)
    c1 = d.box("<b>ConvNeXt 336 px: crop</b><br>log P(crop)", "train", 40, 262, 200, 42)
    c2 = d.box("<b>ConvNeXt 336 px: disease</b><br>log P(disease)", "train", 40, 310, 200, 42)
    c3 = d.box("log P(crop) + log P(disease), then log-softmax over the 333 pairs", "plain", 290, 266, 220, 82)
    mix = d.box("<b>Weighted sum</b><br>0.2 x CLIP<br>+ 0.3 x CNN 224<br>+ 0.5 x CNN 336", "score", 570, 148, 170, 86)
    cal = d.box("<b>Calibration</b><br>minus gamma (6.4) on seen pairs", "score", 770, 161, 170, 60)
    pred = d.box("<b>Prediction</b>", "plain", 770, 290, 170, 40)
    d.edge(a1, a2, "r", "l")
    d.edge(b1, b3, "r", (0, 0.25))
    d.edge(b2, b3, "r", (0, 0.75))
    d.edge(c1, c3, "r", (0, 0.25))
    d.edge(c2, c3, "r", (0, 0.75))
    d.edge(a2, mix, "r", "t")
    d.edge(b3, mix, "r", "l")
    d.edge(c3, mix, "r", "b")
    d.edge(mix, cal, "r", "l")
    d.edge(cal, pred, "b", "t")
    d.text("All members end as log-probabilities over the same 333 pairs, so adding them is meaningful. The weights and gamma "
           "are chosen on validation. A sentence-based model and CNNs make different mistakes, so the sum is right more often than any member.",
           40, 370, 900, 40, size=11, align="left", color="#495057")
    d.legend(426, ["train", "score"])
    return d


def fig11_progress():
    d = Diagram("fig11_progress", "Figure 11. Test harmonic mean (Top-1) at each step, and the mechanism behind the step")
    steps = [("v1", "5.4%", "CLIP as downloaded", "frozen"), ("v3", "10.2%", "symptom text in the prompt", "frozen"),
             ("v4", "43.6%", "learned crop and disease vectors", "train"), ("v4.1", "52.9%", "validation that mirrors the test", "train"),
             ("v5", "65.1%", "LoRA in the image encoder", "train"), ("v5.13", "66.3%", "more LoRA + learned context (CoOp)", "train"),
             ("v6.7", "77.1%", "ensemble with two ConvNeXt CNNs", "score")]
    prev = None
    for i, (ver, hm, why, kind) in enumerate(steps):
        x = 40 + i * 130
        b = d.box(f"<b>{ver}</b><br><font style=\"font-size:15px\"><b>{hm}</b></font>", kind, x, 60, 112, 60)
        d.text(why, x - 4, 124, 120, 40, size=11, color="#495057")
        if prev:
            d.edge(prev, b, "r", "l")
        prev = b
    d.text("<b>For comparison (same test protocol):</b>", 40, 186, 300, 24, align="left", size=11)
    d.box("Independent CNN baseline, ResNet-50: <b>59.7%</b>", "plain", 40, 214, 330, 34, "fontSize=11;")
    d.box("Independent CNN, ConvNeXt-Base: <b>68.6%</b>", "plain", 390, 214, 330, 34, "fontSize=11;")
    d.text("Grey = nothing trained. Blue = one trained CLIP model. Amber = models combined. v5.13 is chosen on validation; its test HM equals v10.3's.",
           40, 262, 900, 22, size=11, align="left", color="#495057")
    return d


def fig12_lesion():
    d = Diagram("fig12_lesion", "Figure 12. Two attempts to make the image encoder look at small lesions (neither was kept)")
    d.box("v7. Patch-level disease score", "group", 30, 50, 910, 150)
    p1 = d.box("196 patch outputs of the last block (one per 16 x 16 pixel square)", "plain", 50, 96, 210, 70)
    p2 = d.box("Project each patch to 512 numbers, cosine with every disease text embedding", "frozen", 300, 96, 210, 70)
    p3 = d.box("<b>Patch score</b> of a disease = mean of its 8 best patches", "score", 550, 96, 170, 70)
    p4 = d.box("Pair score = class-token cosine + 1.0 x patch score of its disease", "score", 760, 96, 160, 70)
    d.edge(p1, p2, "r", "l")
    d.edge(p2, p3, "r", "l")
    d.edge(p3, p4, "r", "l")
    d.box("v13. Local-to-global consistency (no labels needed)", "group", 30, 220, 910, 190)
    g1 = d.box("<b>Global view</b><br>50-100% of the photo", "data", 50, 262, 180, 50)
    l1 = d.box("<b>Local view</b><br>10-35% of the photo, enlarged", "data", 50, 338, 180, 50)
    e1 = d.box("<b>Image encoder + LoRA</b><br>(the same network for both views)", "train", 290, 290, 220, 70)
    g2 = d.box("Global embedding (treated as a fixed target)", "plain", 570, 262, 180, 50)
    l2 = d.box("Local embedding", "plain", 570, 338, 180, 50)
    ls = d.box("<b>Loss</b> = 1 - cosine(local, global)", "loss", 800, 296, 120, 58)
    d.edge(g1, e1, "r", (0, 0.3))
    d.edge(l1, e1, "r", (0, 0.7))
    d.edge(e1, g2, (1, 0.3), "l")
    d.edge(e1, l2, (1, 0.7), "l")
    d.edge(g2, ls, "r", (0, 0.3))
    d.edge(l2, ls, "r", (0, 0.7))
    d.text("Idea behind both: a lesion covers few pixels, so the single class-token embedding is dominated by leaf shape (the crop). "
           "v7 reads the small squares directly; v13 forces a small region to predict the whole image.",
           40, 420, 900, 36, size=11, align="left", color="#495057")
    d.legend(468, ["frozen", "train", "data", "score", "loss"])
    return d


FIGURES = [fig01_pipeline, fig02_prompts, fig03_text_encoder, fig04_composition, fig05_loss, fig06_lora, fig07_calibration,
           fig08_validation, fig09_feature_gen, fig10_ensemble, fig11_progress, fig12_lesion]
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]


def wait_for_png(png: Path, timeout: float = 120.0):
    """Edge 158+ returns from the launcher before its child process writes the screenshot: wait until the file
    exists and its size has stopped changing."""
    import time

    end, last = time.monotonic() + timeout, -1
    while time.monotonic() < end:
        size = png.stat().st_size if png.exists() else -1
        if size > 0 and size == last:
            return
        last = size
        time.sleep(0.5)


def render_png(d: Diagram, png: Path):
    """Render with draw.io's viewer in headless Edge, then trim the white margin."""
    from PIL import Image, ImageChops

    cfg = {"highlight": "none", "nav": False, "resize": True, "lightbox": False, "edit": None, "xml": d.mxfile()}
    # The charset matters: without it Edge misreads the viewer script and nothing is drawn.
    page = ('<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:24px;background:#fff">'
            f'<div class="mxgraph" data-mxgraph="{html.escape(json.dumps(cfg), quote=True)}"></div>'
            '<script charset="utf-8" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script></body></html>')
    edge = next(p for p in EDGE if Path(p).exists())
    png.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = Path(tmp) / "d.html"
        src.write_text(page, encoding="utf-8")
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--force-device-scale-factor=2",
                        "--window-size=1100,760", "--virtual-time-budget=20000", f"--user-data-dir={Path(tmp) / 'profile'}",
                        f"--screenshot={png}", src.as_uri()], check=True, capture_output=True, timeout=180)
        wait_for_png(png)
    if not png.exists():
        raise RuntimeError(f"{png.name}: Edge wrote no screenshot")
    img = Image.open(png).convert("RGB")
    box = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    if box is None:
        raise RuntimeError(f"{png.name}: the render is blank")
    pad = 16
    img.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, img.width), min(box[3] + pad, img.height))).save(png)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    for make in FIGURES:
        d = make()
        if only and d.name not in only:
            continue
        (OUT / f"{d.name}.drawio").write_text(d.mxfile(), encoding="utf-8")
        if "--png" in sys.argv:
            render_png(d, OUT / f"{d.name}.png")
        print("wrote", d.name)


if __name__ == "__main__":
    main()
