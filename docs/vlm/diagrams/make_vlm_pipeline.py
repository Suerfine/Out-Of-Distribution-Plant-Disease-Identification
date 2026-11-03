"""Writes vlm_pipeline.drawio: every operation of the v5.5 CLIP model (prompt learning + LoRA) during setup,
training, model selection and prediction. Each operation has an id (A1, C6, ...) that docs/vlm_improvement_analysis.md
refers to. Run: python docs/diagrams/make_vlm_pipeline.py
"""
from html import escape
from pathlib import Path

W, H, GX, PAD, TOP = 200, 96, 34, 20, 40   # box size, horizontal gap, container padding, container title height
RH = H + 50                                  # row pitch (leaves room for tags under boxes)
STYLE = {
    "op": "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor=#dae8fc;strokeColor=#6c8ebf;",
    "frozen": "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor=#eeeeee;strokeColor=#666666;",
    "train": "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=2;",
    "data": "shape=cylinder3;whiteSpace=wrap;html=1;fontSize=12;fillColor=#d5e8d4;strokeColor=#82b366;size=10;",
    "out": "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor=#d5e8d4;strokeColor=#82b366;",
    "dec": "rhombus;whiteSpace=wrap;html=1;fontSize=12;fillColor=#fff2cc;strokeColor=#d6b656;",
    "loss": "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor=#f8cecc;strokeColor=#b85450;",
}
TAG = {"no": "fillColor=#f8cecc;strokeColor=#b85450;", "yes": "fillColor=#d5e8d4;strokeColor=#82b366;",
       "run": "fillColor=#ffe6cc;strokeColor=#d79b00;"}
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;fontSize=11;strokeColor=#555555;"
CW = PAD * 2 + 8 * W + 7 * GX               # container width: 8 boxes per row

cells, edges, counter = [], [], [0]
lanes = {}                                   # section letter -> (top, bottom) of its container, absolute


def new_id(prefix):
    counter[0] += 1
    return f"{prefix}{counter[0]}"


def ch(rows):
    return TOP + rows * RH


def container(title, y, rows, fill):
    cid = new_id("g")
    lanes[title[0]] = (y, y + ch(rows))
    cells.append(f'<mxCell id="{cid}" value="{escape(title)}" style="swimlane;startSize=30;html=1;fontSize=15;fontStyle=1;'
                 f'fillColor={fill};swimlaneFillColor={fill};strokeColor=#999999;" vertex="1" parent="1">'
                 f'<mxGeometry x="40" y="{y}" width="{CW}" height="{ch(rows)}" as="geometry"/></mxCell>')
    return cid


def box(parent, kind, code, title, detail, col, row, h=H):
    x, y = PAD + col * (W + GX), TOP + row * RH
    value = escape(f"<b>{code} {title}</b><br><font style='font-size:10px'>{detail}</font>")
    cells.append(f'<mxCell id="{code}" value="{value}" style="{STYLE[kind]}" vertex="1" parent="{parent}">'
                 f'<mxGeometry x="{x}" y="{y}" width="{W}" height="{h}" as="geometry"/></mxCell>')
    return code


def tag(parent, col, row, text, kind, slot=0):
    """Experiment already run on this operation, shown under its box."""
    x, y = PAD + col * (W + GX), TOP + row * RH + H + 3 + slot * 22
    cells.append(f'<mxCell id="{new_id("t")}" value="{escape(text)}" style="rounded=1;arcSize=40;whiteSpace=wrap;html=1;fontSize=10;'
                 f'{TAG[kind]}" vertex="1" parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{W}" height="20" as="geometry"/></mxCell>')


def edge(a, b, text="", dashed=False, color=None, parent="1", exit=None, entry=None, points=(), jump=False):
    """exit / entry: (x, y) port on the source / target box as fractions; points: waypoints in the parent's coordinates;
    jump: draw a bridge where this edge crosses edges created before it."""
    st = EDGE + ("dashed=1;" if dashed else "") + (f"strokeColor={color};fontColor={color};" if color else "")
    st += "jumpStyle=arc;jumpSize=8;" if jump else ""
    if exit:
        st += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
    if entry:
        st += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points)
    geo = f'<mxGeometry relative="1" as="geometry"><Array as="points">{pts}</Array></mxGeometry>' if pts else \
        '<mxGeometry relative="1" as="geometry"/>'
    edges.append(f'<mxCell id="{new_id("e")}" value="{escape(text)}" style="{st}" edge="1" parent="{parent}" source="{a}" '
                 f'target="{b}">{geo}</mxCell>')


def chain(ids, parent):
    for a, b in zip(ids, ids[1:]):
        edge(a, b, parent=parent)


# ---- title + legend
title = ("<b>VLM pipeline (v5.5): CLIP ViT-B/16 + compositional prompt vectors + LoRA on the image encoder</b><br>"
         "<font style='font-size:13px'>Every operation in setup, training, model selection and prediction. Ids are referenced by "
         "docs/vlm_improvement_analysis.md. Tags under a box = experiments already run on that operation (validation HM change vs "
         "its base; keep only if more than +2 points).</font>")
cells.append(f'<mxCell id="title" value="{escape(title)}" style="text;html=1;fontSize=20;align=left;verticalAlign=top;whiteSpace=wrap;" '
             f'vertex="1" parent="1"><mxGeometry x="40" y="10" width="{CW}" height="60" as="geometry"/></mxCell>')
lg = new_id("g")
cells.append(f'<mxCell id="{lg}" value="Legend" style="swimlane;startSize=24;html=1;fontSize=13;fontStyle=1;strokeColor=#999999;" '
             f'vertex="1" parent="1"><mxGeometry x="40" y="80" width="{CW}" height="100" as="geometry"/></mxCell>')
legend = [("op", "operation (no weights)"), ("frozen", "frozen pretrained CLIP weights"), ("train", "TRAINABLE parameters"),
          ("data", "data / cache / saved file"), ("loss", "loss term"), ("dec", "selection")]
for i, (k, t) in enumerate(legend):
    cells.append(f'<mxCell id="lg{i}" value="{escape(t)}" style="{STYLE[k]}" vertex="1" parent="{lg}">'
                 f'<mxGeometry x="{PAD + i * 300}" y="32" width="260" height="30" as="geometry"/></mxCell>')
for i, (k, t) in enumerate([("no", "tried, not kept"), ("yes", "tried, kept"), ("run", "running now")]):
    cells.append(f'<mxCell id="lt{i}" value="{escape(t)}" style="rounded=1;arcSize=40;html=1;fontSize=10;{TAG[k]}" vertex="1" '
                 f'parent="{lg}"><mxGeometry x="{PAD + i * 300}" y="70" width="260" height="20" as="geometry"/></mxCell>')
cells.append(f'<mxCell id="ltx" value="{escape("solid arrow = data flow; dashed purple = parameter update; dashed grey = per-epoch / control")}" '
             f'style="text;html=1;fontSize=11;align=left;" vertex="1" parent="{lg}"><mxGeometry x="{PAD + 900}" y="70" width="700" height="20" as="geometry"/></mxCell>')
y = 200

# ---- A. data
g = container("A. Offline data preparation (once)", y, 1, "#f1f8e9")
chain([box(g, "data", "A1", "Read DPD-Small CSVs", "8,000 train / 2,000 test-seen / 1,448 test-unseen; file-name source prefix never used", 0, 0),
       box(g, "op", "A2", "Resize cache", "shorter side 256 px, saved JPEG q95 in runs/cache/resized_256 (shared with the CNNs)", 1, 0),
       box(g, "op", "A3", "Pair table", "333 pairs = 323 seen + 10 unseen (ids interleaved); pair = (crop of 55, disease of 175)", 2, 0),
       box(g, "op", "A4", "Validation split", "seed 42: 20 whole seen pairs (8+ imgs, cap 40, not healthy) held out as pseudo-unseen; 15% of the rest = val-seen", 3, 0),
       box(g, "data", "A5", "fit / val-seen / pseudo-unseen", "crop and disease of every held-out pair still occur in the fit set", 4, 0)], g)
y += ch(1) + 20

# ---- B. text setup
g = container("B. Text-side setup (once per model build; dpd/prompt_learner.py)", y, 1, "#f3e5f5")
chain([box(g, "op", "B1", "Clean names", "underscore to space; ~45 overrides (paddy = rice, xamthomonas = xanthomonas bacterial blight)", 0, 0),
       box(g, "data", "B2", "Symptom descriptions", "1 short visual description per disease (knowledge/disease_descriptions_v1.json, unverified)", 1, 0),
       box(g, "op", "B3", "Fill 3 templates", "pair: a photo of x leaves with x: {desc}. / crop: a photo of x leaves. / disease: a photo of leaves with x: {desc}.", 2, 0),
       box(g, "op", "B4", "Tokenise (BPE)", "77-token context (error if longer); placeholder x = 1 token; cut after EOT (exact under causal mask)", 3, 0),
       box(g, "frozen", "B5", "Token embeddings", "frozen token_embedding lookup, precomputed per group (333 / 55 / 175 sentences)", 4, 0),
       box(g, "train", "B6", "Crop and disease vectors", "55 + 175 vectors x 512, initialised as the mean token embedding of each name", 5, 0)], g)
tag(g, 1, 0, "v8 contrastive rewrites (33 diseases)", "run")
tag(g, 1, 0, "v3.x prompt-text variants (zero-shot)", "yes", 1)
y += ch(1) + 20

# ---- C. image forward
g = container("C. Image encoder forward (every training step; batch 64; AMP fp16)", y, 1, "#e3f2fd")
chain([box(g, "op", "C1", "Sample batch", "uniform shuffle of fit images, drop_last; 4 DataLoader workers", 0, 0),
       box(g, "op", "C2", "Augment", "RandomResizedCrop 224, scale 0.5-1, bicubic; h-flip 0.5, v-flip 0.5; jitter b/c/s 0.2, hue 0", 1, 0),
       box(g, "op", "C3", "Normalise", "to float, CLIP mean / std", 2, 0),
       box(g, "frozen", "C4", "Patch embed", "16x16 conv: 14x14 = 196 patch tokens + CLS token + positional emb, ln_pre", 3, 0),
       box(g, "frozen", "C5", "ViT blocks 1-4", "frozen self-attention + MLP (QuickGELU), width 768, 12 heads", 4, 0),
       box(g, "train", "C6", "ViT blocks 5-12 + LoRA", "W + (32/16) B A on qkv, attn-out, mlp-in, mlp-out; rank 16; B = 0 at start; 1.57M params", 5, 0),
       box(g, "frozen", "C7", "Pool CLS", "ln_post on the CLS token; patch tokens discarded", 6, 0),
       box(g, "frozen", "C8", "Project + L2 norm", "visual.proj 768 to 512; unit-length image embedding v", 7, 0)], g)
tag(g, 0, 0, "v5.4 class-balanced sampler: -2.2", "no")
tag(g, 1, 0, "v7.1 zoom crops 0.25-1: -1.6", "no")
tag(g, 1, 0, "v5.1 336 px input: -0.2", "no", 1)
tag(g, 5, 0, "v5.3 8 blocks, rank 16: +5.7 KEPT", "yes")
tag(g, 6, 0, "v7 + patch-token disease score: +1.5", "no")
y += ch(1) + 20

# ---- D. text forward
g = container("D. Text encoder forward (every training step; differentiable w.r.t. the vectors only)", y, 1, "#f3e5f5")
chain([box(g, "op", "D1", "Substitute vectors", "write each crop / disease vector into its sentence's placeholder slot", 0, 0),
       box(g, "frozen", "D2", "Add positional emb", "frozen positional embedding (truncated length)", 1, 0),
       box(g, "frozen", "D3", "Text transformer", "12 frozen layers, causal mask, width 512; gradient checkpointing", 2, 0),
       box(g, "frozen", "D4", "ln_final, EOT token", "hidden state at the EOT position", 3, 0),
       box(g, "frozen", "D5", "Project + L2 norm", "text_projection 512 to 512; unit-length", 4, 0),
       box(g, "data", "D6", "Text embeddings", "T_pair 333x512, T_crop 55x512, T_dis 175x512 (recomputed every step)", 5, 0)], g)
y += ch(1) + 20

# ---- E. scoring, losses, optimiser
g = container("E. Scoring, losses and optimisation (train/train_lora.py fit)", y, 2, "#ffebee")
box(g, "op", "E1", "Cosine scores", "v T_pair restricted to TRAINED pair columns (unseen, pseudo-unseen get no gradient); v T_crop; v T_dis", 0, 0)
box(g, "op", "E2", "Temperature", "x logit scale 100 (fixed, CLIP's value)", 1, 0)
box(g, "loss", "E3", "Pair CE (w 1.0)", "softmax over the ~300 trained pairs", 2, 0)
box(g, "loss", "E4", "Crop CE (w 0.5)", "softmax over 55 crop prompts", 3, 0)
box(g, "loss", "E5", "Disease CE (w 0.5)", "softmax over 175 disease prompts", 4, 0)
box(g, "op", "E6", "Weighted sum", "L = 1.0 pair + 0.5 crop + 0.5 disease", 5, 0)
box(g, "op", "E7", "Backward", "AMP; gradients through the frozen encoders into vectors + LoRA only", 6, 0)
box(g, "train", "E8", "AdamW step", "vectors lr 2e-4, LoRA lr 1e-4, weight decay 0", 7, 0)
box(g, "op", "E9", "Cosine LR schedule", "per step, no warm-up; max 15 epochs, ~125 steps / epoch", 7, 1)
edge("E1", "E2", parent=g)
# E2 fans out to the three losses and they merge into E6. Only the neighbours (E2-E3, E5-E6) connect directly; the
# rest run below the row (coordinates relative to the container) so they don't pass through E3-E5. The fan crosses
# the merge twice, so the fan edges come last and bridge over it.
edge("E2", "E3", parent=g)
edge("E3", "E6", parent=g, exit=(1, 0.75), entry=(0.5, 1),
     points=[(PAD + 2 * (W + GX) + W + GX // 2, TOP + 0.75 * H), (PAD + 2 * (W + GX) + W + GX // 2, TOP + H + 50),
             (PAD + 5 * (W + GX) + W / 2, TOP + H + 50)])
edge("E4", "E6", parent=g, exit=(0.7, 1), entry=(0.5, 1),
     points=[(PAD + 3 * (W + GX) + 0.7 * W, TOP + H + 50), (PAD + 5 * (W + GX) + W / 2, TOP + H + 50)])
edge("E5", "E6", parent=g)
for col, s in ((3, "E4"), (4, "E5")):
    edge("E2", s, parent=g, exit=(0.5, 1), entry=(0.3, 1), jump=True,
         points=[(PAD + (W + GX) + W / 2, TOP + H + 74), (PAD + col * (W + GX) + 0.3 * W, TOP + H + 74)])
chain(["E6", "E7", "E8", "E9"], g)
tag(g, 2, 0, "v7.2 + within-crop pair CE: -0.5", "no")
y += ch(2) + 20

# ---- F. model selection
g = container("F. Stage 1: model selection on validation (after every epoch)", y, 1, "#fffde7")
chain([box(g, "op", "F1", "Encode validation", "val-seen + pseudo-unseen; test transform: resize 224 + centre crop, no augmentation", 0, 0),
       box(g, "op", "F2", "Score 333 pairs", "cosine vs T_pair; the real 10 unseen pairs masked to -1e9", 1, 0),
       box(g, "op", "F3", "Gamma sweep", "subtract gamma 0 to 0.1 (step 0.0025) from the SEEN-pair scores", 2, 0),
       box(g, "op", "F4", "Harmonic mean", "HM of val-seen top-1 and pseudo-unseen top-1, per gamma", 3, 0),
       box(g, "dec", "F5", "Pick epoch, gamma", "argmax HM over 15 epochs x 41 gammas (one split)", 4, 0, h=110)], g)
tag(g, 0, 0, "v5.2 flip TTA: +0.4", "no")
tag(g, 2, 0, "per-pair calibration: rejected", "no")
y += ch(1) + 30

# ---- G. refit
g = container("G. Stage 2: refit and save", y, 1, "#f1f8e9")
chain([box(g, "op", "G1", "Fresh model", "rebuild B6 + C6 from scratch, same seed", 0, 0),
       box(g, "op", "G2", "Refit on all train", "all 8,000 images and 323 seen pairs, for the selected number of epochs", 1, 0),
       box(g, "data", "G3", "Saved run", "prompt_vectors.pt, lora_image_encoder.pt, pair_text_embeddings.npy, gamma, metrics.json, scores.npz", 2, 0)], g)
y += ch(1) + 20

# ---- H. prediction
g = container("H. Prediction: test evaluation and the Streamlit app (dpd/inference.py)", y, 2, "#e8f5e9")
chain([box(g, "data", "H1", "Input image", "upload or test image (jpg / png, converted to RGB)", 0, 0),
       box(g, "op", "H2", "Standardise", "downscale to 256 + JPEG q95 round trip (what training saw)", 1, 0),
       box(g, "op", "H3", "Test transform", "resize 224 bicubic, centre crop 224, normalise", 2, 0),
       box(g, "train", "H4", "Image encoder + LoRA", "C4-C8 with the saved adapters: v (512)", 3, 0),
       box(g, "op", "H5", "Score 333 pairs", "v x cached T_pair (text encoder not run at test time)", 4, 0),
       box(g, "op", "H6", "Calibrate", "subtract gamma (0.075) from the 323 seen pairs", 5, 0),
       box(g, "out", "H7", "Rank: top-1 / top-5", "predicted pair, hence crop + disease", 6, 0),
       box(g, "out", "H8", "Confidence", "softmax(100 x scores) over all 333 pairs", 7, 0)], g)
box(g, "data", "H9", "Train bank", "image embeddings of all 8,000 training images (cached by the app)", 3, 1)
box(g, "out", "H10", "Retrieve neighbours", "cosine kNN of v in the train bank, plus same-crop / same-disease images", 4, 1)
box(g, "op", "H11", "Optional ensemble (v6.x)", "w log_softmax(CLIP) + (1-w) log P(CNN pair); w, gamma chosen on validation", 5, 1)
box(g, "out", "H12", "Evaluate", "top-1/5 seen and unseen, HM, crop / disease acc, 323x333 seen confusion matrix", 6, 1)
edge("H4", "H9", parent=g); edge("H9", "H10", parent=g); edge("H5", "H11", dashed=True, parent=g)
edge("H11", "H12", parent=g); edge("H7", "H12", parent=g)
tag(g, 5, 1, "v6 + CNN ensemble: +1.8 (v6.1, v6.2 runs)", "yes")

# ---- cross-section edges, routed by hand so they never run along a container border: horizontal runs go in the
# 20 px gaps between sections, vertical runs in the gutters between box columns or outside the containers (left of
# x = 40, right of x = 40 + CW), clear of the tags under the boxes and of the centred section titles.
def cx(col, f=0.5):                          # absolute x at fraction f across a box in column col
    return 40 + PAD + col * (W + GX) + f * W


def gutter(col):                             # absolute x of the gap right of column col
    return 40 + PAD + col * (W + GX) + W + GX // 2


def gap(sec, d=0):                           # absolute y in the gap below section sec
    return lanes[sec][1] + 10 + d


def row(sec, f=0.5, r=0):                    # absolute y at fraction f down a box in row r of section sec
    return lanes[sec][0] + TOP + r * RH + f * H


L1, L2, L3, R1, R2 = 28, 16, 4, 40 + CW + 22, 40 + CW + 42   # vertical runs outside the containers
edge("A2", "C1", "images", exit=(0.5, 1), entry=(0, 0.25),
     points=[(cx(1), gap("A", -5)), (L1, gap("A", -5)), (L1, row("C", 0.25))])
edge("A3", "B1", "names", exit=(0.5, 1), entry=(0.5, 0), points=[(cx(2), gap("A")), (cx(0), gap("A"))])
edge("A5", "C1", "fit set", exit=(0.8, 0), entry=(0, 0.75),
     points=[(cx(4, 0.8), lanes["A"][0] - 6), (L2, lanes["A"][0] - 6), (L2, row("C", 0.75))])
edge("A5", "F1", "val sets", dashed=True, exit=(0.95, 0), entry=(0, 0.5),
     points=[(cx(4, 0.95), lanes["A"][0] - 14), (L3, lanes["A"][0] - 14), (L3, row("F"))])
edge("B5", "D1", "token emb", exit=(0.5, 1), entry=(0.3, 0),
     points=[(cx(4), gap("B", -5)), (gutter(0), gap("B", -5)), (gutter(0), gap("C", -5)), (cx(0, 0.3), gap("C", -5))])
edge("B6", "D1", "vectors", exit=(0.25, 1), entry=(0.7, 0),
     points=[(cx(5, 0.25), gap("B", 5)), (gutter(1), gap("B", 5)), (gutter(1), gap("C", 5)), (cx(0, 0.7), gap("C", 5))])
edge("C8", "E1", "v", exit=(0.5, 1), entry=(0.7, 0), points=[(cx(7), gap("D", 5)), (cx(0, 0.7), gap("D", 5))])
edge("D6", "E1", "T", exit=(0.5, 1), entry=(0.3, 0), points=[(cx(5), gap("D", -5)), (cx(0, 0.3), gap("D", -5))])
edge("E8", "B6", "update", dashed=True, color="#9673a6", exit=(1, 0.7), entry=(0.5, 0),
     points=[(R2, row("E", 0.7)), (R2, gap("A")), (cx(5), gap("A"))])
edge("E8", "C6", "update", dashed=True, color="#9673a6", exit=(1, 0.3), entry=(0.75, 0),
     points=[(R1, row("E", 0.3)), (R1, gap("B")), (cx(5, 0.75), gap("B"))])
edge("E9", "F1", "end of epoch", dashed=True, exit=(0.5, 1), entry=(0.5, 0), points=[(cx(7), gap("E")), (cx(0), gap("E"))])
edge("F5", "G1", "epoch, gamma", exit=(0.5, 1), entry=(0.5, 0), points=[(cx(4), gap("F")), (cx(0), gap("F"))])
edge("G3", "H4", "adapters", exit=(0.5, 1), entry=(0.25, 0), points=[(cx(2), gap("G")), (cx(3, 0.25), gap("G"))])
edge("G3", "H5", "T_pair, gamma", exit=(1, 0.5), entry=(0.25, 0), points=[(cx(4, 0.25), row("G"))])

xml = ('<mxGraphModel adaptiveColors="auto" grid="1"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) + "".join(edges) + "</root></mxGraphModel>")
here = Path(__file__).parent
(here / "vlm_pipeline.xml").write_text(xml, encoding="utf-8")
(here / "vlm_pipeline.drawio").write_text(f'<mxfile host="drawio"><diagram name="VLM pipeline">{xml}</diagram></mxfile>', encoding="utf-8")
print(len(cells), "cells,", len(edges), "edges, height", y + ch(2))
