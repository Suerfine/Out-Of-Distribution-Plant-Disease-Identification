"""Explanation figure for the v1 (zero-shot CLIP) results section: why off-the-shelf CLIP finds the crop but not the disease.

Run from the repo root:  .venv/Scripts/python docs/vlm/diagrams/make_v1_results_figure.py
Writes docs/vlm/diagrams/results/fig_v1_mechanism.svg and, with Microsoft Edge installed, a PNG of it (headless Edge,
local file only). Every number in the figure is computed here from runs/vlm/v1/scores.npz and from the v1 prompts
re-encoded by the frozen CLIP text encoder (no training, a few seconds on the GPU).

Colours follow the breakdown figures: grey = pretrained CLIP, frozen; green = data in; amber = scores / decisions.
"""
import html
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dpd.clip_utils import build_prompts, encode_prompts, load_clip  # noqa: E402
from dpd.data import pair_table, read_metadata  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
INK, GREY, MUTED = "#1B1F24", "#868E96", "#495057"
FROZEN = ("#F1F3F5", "#868E96")
DATA = ("#EAF6EC", "#4C9A66")
SCORE = ("#FFF4D6", "#C99A2E")
RED = "#C9504A"
EXAMPLE = "tomato_early_blight"


def stats() -> dict:
    cfg = json.loads((ROOT / "runs/vlm/v1/config.json").read_text(encoding="utf-8"))
    pairs = pair_table(read_metadata())
    crop, dis = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()
    z = np.load(ROOT / "runs/vlm/v1/scores.npz")
    model, _, tok = load_clip(cfg["model"], "cuda")
    T = encode_prompts(model, tok, build_prompts(pairs, cfg["names"], cfg["prompts"]), "cuda")

    iu = np.triu_indices(len(pairs), 1)
    C = (T @ T.T)[iu]
    sc, sd = (crop[:, None] == crop[None])[iu], (dis[:, None] == dis[None])[iu]
    X = np.concatenate([z["test_seen_scores"], z["test_unseen_scores"]])
    y = np.concatenate([z["test_seen_labels"], z["test_unseen_labels"]])
    pred = X.argmax(1)
    given = [np.flatnonzero(crop == crop[t]) for t in y]
    counts = np.bincount(pred, minlength=len(pairs))
    hub = int(counts.argmax())

    ex = int(pairs.index[pairs["pair"] == EXAMPLE][0])
    m = y == ex
    ex_pred = Counter(pairs["pair"].to_numpy()[pred[m]])
    return {
        "sim_crop": float(C[sc & ~sd].mean()), "sim_disease": float(C[~sc & sd].mean()), "sim_none": float(C[~sc & ~sd].mean()),
        "crop_acc": float((crop[pred] == crop[y]).mean()),
        "crop_chance": float(np.mean([(crop == crop[t]).sum() / len(pairs) for t in y])),
        "dis_acc": float(np.mean([g[X[i, g].argmax()] == y[i] for i, g in enumerate(given)])),
        "dis_chance": float(np.mean([1 / len(g) for g in given])),
        "pair_acc": float((pred == y).mean()),
        "n_images": int(len(y)), "n_pairs": len(pairs), "n_crops": int(crop.max() + 1),
        "hub": pairs.loc[hub, "pair"], "hub_preds": int(counts[hub]), "hub_true": int((y == hub).sum()),
        "ex_n": int(m.sum()), "ex_crop_right": float((crop[pred[m]] == crop[ex]).mean()),
        "ex_correct": float((pred[m] == ex).mean()), "ex_to_hub": int(ex_pred.get(pairs.loc[hub, "pair"], 0)),
        "ex_crop_pairs": int((crop == crop[ex]).sum()),
        "ex_top": ex_pred.most_common(4),
    }


def svg(s: dict) -> str:
    out = []
    add = out.append

    def box(x, y, w, h, fill, stroke, lines, size=13, bold_first=False, sw=1.4):
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        n = len(lines)
        for i, t in enumerate(lines):
            ty = y + h / 2 + (i - (n - 1) / 2) * (size + 5) + size * 0.35
            fw = "bold" if bold_first and i == 0 else "normal"
            add(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="{size}" font-weight="{fw}">{html.escape(t)}</text>')

    def text(x, y, t, size=13, anchor="start", color=INK, weight="normal", italic=False):
        st = ' font-style="italic"' if italic else ""
        add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-weight="{weight}"{st}>{html.escape(t)}</text>')

    def arrow(x1, y1, x2, y2, color=MUTED, dashed=False, w=1.6):
        d = ' stroke-dasharray="6 4"' if dashed else ""
        add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" marker-end="url(#a)"{d}/>')

    W, H = 1240, 640
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'font-family="Helvetica, Arial, sans-serif" fill="{INK}">')
    add(f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>')
    add(f'<rect width="{W}" height="{H}" fill="white"/>')
    text(30, 36, "Why zero-shot CLIP (v1) finds the crop but not the disease", 19, weight="bold")
    text(30, 60, f"Worked example: the {s['ex_n']} test images of the unseen pair \u201ctomato early blight\u201d. "
                 f"Every number is from the v1 test run.", 13, color=MUTED)

    # ---- left: the inputs ----
    box(30, 100, 220, 60, *DATA, ["Test image", "tomato leaf, early blight"], bold_first=True)
    arrow(140, 160, 140, 190)
    box(30, 190, 220, 52, *FROZEN, ["CLIP image encoder", "(frozen, no training)"], bold_first=True)
    arrow(250, 216, 330, 300)

    box(30, 300, 220, 70, *DATA, ["333 prompts, one per pair", "\u201ca photo of tomato leaves", "with early blight.\u201d"], 12, bold_first=True)
    arrow(140, 370, 140, 400)
    box(30, 400, 220, 52, *FROZEN, ["CLIP text encoder", "(frozen, no training)"], bold_first=True)
    arrow(250, 426, 330, 400)

    # ---- middle: the shared embedding space ----
    add(f'<rect x="330" y="90" width="520" height="430" rx="12" fill="#FAFBFC" stroke="#ADB5BD" stroke-dasharray="6 4"/>')
    text(346, 114, "Shared embedding space (cosine similarity = closeness)", 13, color=MUTED, weight="bold")

    # crop clusters
    def cluster(cx, cy, r, colour, label, n, dots):
        add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{colour}" fill-opacity="0.10" stroke="{colour}" stroke-width="1.2"/>')
        rng = np.random.default_rng(n)
        for _ in range(dots):
            a, rr = rng.uniform(0, 2 * np.pi), r * 0.6 * np.sqrt(rng.uniform())
            add(f'<circle cx="{cx + rr * np.cos(a):.1f}" cy="{cy + rr * np.sin(a):.1f}" r="3.2" fill="{colour}"/>')
        text(cx, cy + r + 16, label, 12, anchor="middle", color=colour, weight="bold")

    cluster(470, 220, 50, "#C9504A", "apple prompts", 1, 14)
    cluster(740, 175, 44, "#4C9A66", "rice prompts", 2, 10)
    cluster(560, 355, 64, "#2F6FD0", f"tomato prompts ({s['ex_crop_pairs']} pairs)", 3, 7)
    cluster(760, 380, 38, "#7048A8", "soybean prompts", 4, 6)
    text(760, 452, "★ hub: soybean bacterial blight", 11, anchor="middle", weight="bold")

    # zoomed tomato labels
    for dx, dy, t in ((-56, -26, "early blight"), (4, -32, "late blight"), (-40, 8, "leaf blight"), (8, 26, "leaf mold")):
        text(560 + dx, 355 + dy, t, 10, color="#2F6FD0", weight="bold")
    # hub prompt
    hx, hy = 735, 375
    add(f'<polygon points="{hx},{hy - 11} {hx + 3.5},{hy - 3.5} {hx + 11},{hy - 3} {hx + 5},{hy + 2.5} {hx + 7},{hy + 10} '
        f'{hx},{hy + 6} {hx - 7},{hy + 10} {hx - 5},{hy + 2.5} {hx - 11},{hy - 3} {hx - 3.5},{hy - 3.5}" fill="{SCORE[1]}" stroke="{INK}" stroke-width="0.8"/>')

    # image embedding
    ix, iy = 640, 265
    add(f'<circle cx="{ix}" cy="{iy}" r="8" fill="{DATA[1]}" stroke="{INK}" stroke-width="1"/>')
    text(ix - 14, iy + 4, "image embedding", 11, anchor="end", color=DATA[1], weight="bold")
    arrow(ix - 5, iy + 7, 590, 318, color="#2F6FD0", w=1.4)
    arrow(ix + 6, iy + 7, hx - 6, hy - 12, color=SCORE[1], w=1.4)

    text(346, 478, f"Prompts that share the crop word:  cos {s['sim_crop']:.2f}", 12)
    text(346, 496, f"Prompts that share the disease words:  cos {s['sim_disease']:.2f}", 12)
    text(346, 514, f"Prompts that share neither:  cos {s['sim_none']:.2f}", 12, color=MUTED)

    # ---- right: what happens, step by step ----
    rx = 880
    box(rx, 90, 330, 110, *SCORE, [
        "1  The crop is found, roughly",
        f"Top-1 pair has the right crop: {s['crop_acc']:.0%}",
        f"(random guess {s['crop_chance']:.0%}; {s['crop_acc'] / s['crop_chance']:.0f}x chance)",
        f"This example: {s['ex_crop_right']:.0%} land in a tomato prompt"], 12, bold_first=True)
    box(rx, 215, 330, 110, *SCORE, [
        "2  The disease is close to a coin toss",
        f"Even with the crop given: {s['dis_acc']:.0%} right",
        f"(random guess {s['dis_chance']:.0%}; {s['dis_acc'] / s['dis_chance']:.1f}x chance)",
        "Same-crop prompts differ by one or two words"], 12, bold_first=True)
    box(rx, 340, 330, 110, *SCORE, [
        "3  A few hub prompts win the ties",
        f"{s['hub'].replace('_', ' ')}: {s['hub_preds']} of {s['n_images']} predictions",
        f"but only {s['hub_true']} true test image",
        f"{s['ex_to_hub']} of the {s['ex_n']} tomato early blight images go to it"], 12, bold_first=True)
    box(rx, 465, 330, 55, "#FFFFFF", RED, [
        f"Result: right pair for {s['pair_acc']:.1%} of images",
        "seen 6.4%, unseen 4.6%, HM 5.4% (chance 0.3%)"], 12, bold_first=True, sw=1.6)

    text(30, 570, "Reading the figure: an image is assigned to the prompt with the highest cosine similarity. "
                  "CLIP's pretraining (internet image-caption pairs) taught it crop names well and rare disease names poorly,", 12, color=MUTED)
    text(30, 590, "so the text embedding of a prompt is dominated by its crop word. The image lands near the right crop's prompts, but those "
                  "prompts sit almost on top of each other (cos 0.87),", 12, color=MUTED)
    text(30, 610, "so tiny, meaningless differences decide the disease, and prompts that happen to sit close to many leaf images (hubs) "
                  "collect predictions from every crop.", 12, color=MUTED)
    add("</svg>")
    return "\n".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    s = stats()
    path = OUT / "fig_v1_mechanism.svg"
    path.write_text(svg(s), encoding="utf-8")
    print(json.dumps(s, indent=1))
    if EDGE.exists():
        png = OUT / "fig_v1_mechanism.png"
        subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        "--window-size=1240,640", f"--screenshot={png}", path.as_uri()], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        print("PNG:", png, png.exists())


if __name__ == "__main__":
    main()
