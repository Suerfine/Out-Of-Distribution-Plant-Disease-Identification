"""Misclassification Gallery Generator for Out-of-Distribution Plant Disease Identification.

Generates visual grid galleries of misclassified test set images, highlighting seen and unseen
classes, organized from most wrong (highest margin/confidence gap) to least wrong.

Default layout: 5 columns x 4 rows (20 images).
Displays top-5 predicted plants (crops) and diseases with probabilities in comparison
to the ground truth.

Usage:
    # Generate unseen-only gallery
    python -m train.run_misclassification_gallery --run-dir runs/multitask/Nat_multitask_resnetv2_50 --split unseen

    # Generate seen-only gallery and sync to remote/cloud drive
    python -m train.run_misclassification_gallery --run-dir runs/multitask/Nat_multitask_resnetv2_50 --split seen --sync-dir /content/drive/MyDrive/Nat_AML_DPD_group_assignment/runs

    # Generate combined gallery (both seen and unseen, organized by most wrong)
    python -m train.run_misclassification_gallery --run-dir runs/multitask/Nat_multitask_resnetv2_50 --split both
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dpd.cnn import HEAD_SIZES, ImageLabels, MultiHeadNet, build_transforms, compose_pair_scores
from dpd.config import load_config
from dpd.data import pair_table, read_metadata
from dpd.evaluate import calibrate
from dpd.resize_cache import cached_name, resized_dir


def load_test_image(file_name: str, dataset_root: Path | None = None) -> Image.Image:
    """Load image from DPD_Small_dataset or resized cache, or fallback to placeholder."""
    candidates = []
    if dataset_root is not None:
        candidates.append(dataset_root / "DPD_Small_dataset" / file_name)
        candidates.append(dataset_root / "DPD_Small_dataset" / cached_name(file_name))
        candidates.append(dataset_root / file_name)

    env_root = os.environ.get("DPD_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "DPD_Small_dataset" / file_name)
        candidates.append(Path(env_root) / "DPD_Small_dataset" / cached_name(file_name))

    project_root = Path(__file__).resolve().parent.parent
    candidates.append(project_root / "DPD_Small" / "DPD_Small_dataset" / file_name)
    candidates.append(project_root / "DPD_Small" / "DPD_Small_dataset" / cached_name(file_name))
    candidates.append(resized_dir(256) / cached_name(file_name))

    for path in candidates:
        if path.exists() and path.is_file():
            try:
                with Image.open(path) as img:
                    return img.convert("RGB")
            except Exception:
                pass

    # Fallback placeholder if image file is not on local disk
    img = Image.new("RGB", (256, 256), color=(235, 238, 242))
    return img


def get_or_compute_head_logp(
    run_dir: Path,
    cfg: dict,
    df_meta: pd.DataFrame,
    image_dir: Path,
    device: str = "auto",
    batch_size: int = 32,
    num_workers: int = 2,
    force_infer: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Load cached head log-probabilities or run inference using multitask_net.pt."""
    cached_path = run_dir / "head_scores.npz"
    if cached_path.exists() and not force_infer:
        data = np.load(cached_path)
        splits = ("test_seen", "test_unseen")
        if all(f"{s}_crop_logp" in data and f"{s}_disease_logp" in data for s in splits):
            return {
                s: {
                    "crop": data[f"{s}_crop_logp"],
                    "disease": data[f"{s}_disease_logp"],
                }
                for s in splits
            }

    # Run inference with model weights
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    mcfg = cfg["model"]
    heads = tuple(cfg.get("heads", ("crop", "disease")))
    head_sizes = {h: HEAD_SIZES[h] for h in heads}
    model = MultiHeadNet(
        mcfg["backbone"],
        pretrained=False,
        heads=head_sizes,
        dropout=mcfg.get("head_dropout", 0.0),
    ).to(dev)

    weights_candidates = [
        run_dir / "multitask_net.pt",
        run_dir / "checkpoints" / "last.pt",
    ]
    weights_path = next((p for p in weights_candidates if p.exists()), None)
    if weights_path is None:
        raise FileNotFoundError(
            f"Could not find model weights in {run_dir} (checked {weights_candidates}) "
            f"and {cached_path} does not exist."
        )

    ckpt = torch.load(weights_path, map_location=dev)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    _, test_tf = build_transforms(cfg["input"], cfg.get("augmentation", {}))
    test_logp = {}
    amp_enabled = bool(cfg.get("runtime", {}).get("amp", True) and dev == "cuda")

    for s in ("test_seen", "test_unseen"):
        sub_df = df_meta[df_meta["split"] == s].reset_index(drop=True)
        loader = DataLoader(
            ImageLabels(sub_df, image_dir, test_tf),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(dev == "cuda"),
        )
        out = {k: [] for k in heads}
        with torch.no_grad():
            for x, _, _ in loader:
                with torch.autocast(dev, enabled=amp_enabled):
                    logits = model(x.to(dev, non_blocking=True))
                for k, v in logits.items():
                    out[k].append(F.log_softmax(v.float(), dim=-1).cpu())
        test_logp[s] = {k: torch.cat(v).numpy() for k, v in out.items()}

    # Save to head_scores.npz for future instant loading
    try:
        np.savez_compressed(
            cached_path,
            **{f"{s}_{h}_logp": test_logp[s][h] for s in test_logp for h in test_logp[s]},
        )
    except Exception as e:
        print(f"Warning: Failed to cache head_scores.npz: {e}")

    return test_logp


def collect_misclassifications(
    run_dir: Path,
    cfg: dict,
    df_meta: pd.DataFrame,
    pairs: pd.DataFrame,
    gamma: float,
    head_logp: dict[str, dict[str, np.ndarray]],
    split_filter: str = "both",
    sort_by: str = "margin",
) -> list[dict]:
    """Identify misclassified images and compute all top-5 plant & disease predictions."""
    crop_map = {row.crop_id: row.crop for row in pairs.itertuples()}
    disease_map = {row.disease_id: row.disease for row in pairs.itertuples()}
    pair_name_map = {row.pair_id: row.pair for row in pairs.itertuples()}
    pair_unseen_map = {row.pair_id: bool(row.is_unseen) for row in pairs.itertuples()}
    unseen_mask = pairs["is_unseen"].to_numpy()

    # Load pre-saved raw scores if present to ensure 100% exact numerical match with evaluation
    saved_scores = {}
    scores_npz = run_dir / "scores.npz"
    if scores_npz.exists():
        try:
            s_data = np.load(scores_npz)
            for s in ("test_seen", "test_unseen"):
                if f"{s}_scores" in s_data:
                    saved_scores[s] = s_data[f"{s}_scores"]
        except Exception:
            pass

    records = []
    splits_to_eval = []
    if split_filter in ("both", "all"):
        splits_to_eval = ["test_seen", "test_unseen"]
    elif split_filter in ("seen", "test_seen"):
        splits_to_eval = ["test_seen"]
    elif split_filter in ("unseen", "test_unseen"):
        splits_to_eval = ["test_unseen"]
    else:
        raise ValueError(f"Unknown split filter: {split_filter!r}")

    for s in splits_to_eval:
        sub_df = df_meta[df_meta["split"] == s].reset_index(drop=True)
        logp_c = head_logp[s]["crop"]
        logp_d = head_logp[s]["disease"]
        prob_c = np.exp(logp_c)
        prob_d = np.exp(logp_d)

        if s in saved_scores:
            pair_scores = saved_scores[s]
        else:
            pair_scores = compose_pair_scores(logp_c, logp_d, pairs, crop_weight=0.5)

        s_cal = calibrate(pair_scores, unseen_mask, gamma)
        pred_pair_ids = s_cal.argmax(axis=1)
        true_pair_ids = sub_df["pair_id"].to_numpy()
        true_crop_ids = sub_df["crop_id"].to_numpy()
        true_disease_ids = sub_df["disease_id"].to_numpy()
        files = sub_df["file"].to_numpy()
        is_unseen_split = (s == "test_unseen")

        # Top 5 plant predictions per image
        order_c = np.argsort(-prob_c, axis=1)[:, :5]
        # Top 5 disease predictions per image
        order_d = np.argsort(-prob_d, axis=1)[:, :5]

        for i in range(len(sub_df)):
            pred_pid = int(pred_pair_ids[i])
            true_pid = int(true_pair_ids[i])

            # Filter only misclassified images
            if pred_pid == true_pid:
                continue

            t_score = float(s_cal[i, true_pid])
            p_score = float(s_cal[i, pred_pid])
            margin = p_score - t_score
            true_rank = int((s_cal[i] >= t_score).sum())

            # Plant (crop) stats
            t_cid = int(true_crop_ids[i])
            t_c_prob = float(prob_c[i, t_cid])
            t_c_rank = int((prob_c[i] >= t_c_prob).sum())
            top5_c = [
                {
                    "crop_id": int(cid),
                    "name": crop_map.get(int(cid), f"crop_{cid}"),
                    "prob": float(prob_c[i, cid]),
                    "is_gt": bool(int(cid) == t_cid),
                }
                for cid in order_c[i]
            ]

            # Disease stats
            t_did = int(true_disease_ids[i])
            t_d_prob = float(prob_d[i, t_did])
            t_d_rank = int((prob_d[i] >= t_d_prob).sum())
            top5_d = [
                {
                    "disease_id": int(did),
                    "name": disease_map.get(int(did), f"disease_{did}"),
                    "prob": float(prob_d[i, did]),
                    "is_gt": bool(int(did) == t_did),
                }
                for did in order_d[i]
            ]

            pred_crop_id = int(pairs.loc[pred_pid, "crop_id"])
            pred_disease_id = int(pairs.loc[pred_pid, "disease_id"])
            pred_crop_name = crop_map.get(pred_crop_id, f"crop_{pred_crop_id}")
            pred_disease_name = disease_map.get(pred_disease_id, f"disease_{pred_disease_id}")

            records.append({
                "file": str(files[i]),
                "split": s,
                "is_unseen": bool(is_unseen_split or pair_unseen_map.get(true_pid, False)),
                "true_pair_id": true_pid,
                "true_pair": pair_name_map.get(true_pid, f"pair_{true_pid}"),
                "true_crop_id": t_cid,
                "true_crop": crop_map.get(t_cid, f"crop_{t_cid}"),
                "true_crop_prob": t_c_prob,
                "true_crop_rank": t_c_rank,
                "true_disease_id": t_did,
                "true_disease": disease_map.get(t_did, f"disease_{t_did}"),
                "true_disease_prob": t_d_prob,
                "true_disease_rank": t_d_rank,
                "pred_pair_id": pred_pid,
                "pred_pair": pair_name_map.get(pred_pid, f"pair_{pred_pid}"),
                "pred_is_unseen": pair_unseen_map.get(pred_pid, False),
                "pred_crop_name": pred_crop_name,
                "pred_disease_name": pred_disease_name,
                "crop_correct": bool(pred_crop_id == t_cid),
                "disease_correct": bool(pred_disease_id == t_did),
                "margin": margin,
                "true_pair_rank": true_rank,
                "pred_score": p_score,
                "true_score": t_score,
                "top5_plants": top5_c,
                "top5_diseases": top5_d,
            })

    # Sort by "most wrong"
    if sort_by == "margin":
        records.sort(key=lambda r: r["margin"], reverse=True)
    elif sort_by == "rank":
        records.sort(key=lambda r: (r["true_pair_rank"], r["margin"]), reverse=True)
    elif sort_by == "prob":
        records.sort(key=lambda r: (r["true_crop_prob"] * r["true_disease_prob"], -r["margin"]))
    else:
        records.sort(key=lambda r: r["margin"], reverse=True)

    return records


def render_matplotlib_gallery(
    records: list[dict],
    output_path: Path,
    dataset_root: Path | None = None,
    cols: int = 5,
    rows: int = 4,
    title: str | None = None,
    subtitle: str | None = None,
) -> Path:
    """Render a clean grid gallery of misclassified images with top-5 prediction statistics."""
    total_slots = cols * rows
    display_records = records[:total_slots]
    n_display = len(display_records)

    # Canvas dimensions: 5.4 inches per column, 7.2 inches per row
    fig_w = cols * 5.4
    fig_h = rows * 7.2
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)

    # 2 sub-rows per item: top sub-row for image, bottom sub-row for text box
    height_ratios = []
    for _ in range(rows):
        height_ratios.extend([1.35, 1.0])

    gs = fig.add_gridspec(
        nrows=rows * 2,
        ncols=cols,
        height_ratios=height_ratios,
        left=0.03,
        right=0.97,
        top=0.94,
        bottom=0.02,
        wspace=0.22,
        hspace=0.28,
    )

    # Main titles
    if title is None:
        title = "Misclassification Gallery - Out-of-Distribution Plant Disease Identification"
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.985, color="#1A202C")
    if subtitle:
        fig.text(0.5, 0.965, subtitle, ha="center", fontsize=12, color="#4A5568")

    for slot_idx in range(total_slots):
        r = slot_idx // cols
        c = slot_idx % cols
        ax_img = fig.add_subplot(gs[2 * r, c])
        ax_txt = fig.add_subplot(gs[2 * r + 1, c])

        if slot_idx >= n_display:
            ax_img.axis("off")
            ax_txt.axis("off")
            continue

        item = display_records[slot_idx]
        is_unseen = item["is_unseen"]

        # Color themes: Vibrant Orange for Unseen, Cool Blue for Seen
        if is_unseen:
            theme_color = "#D9531E"     # Deep Orange / Vermilion
            theme_bg = "#FFF5EE"        # Seashell
            badge_text = "★ UNSEEN TEST"
        else:
            theme_color = "#1565C0"     # Deep Cobalt Blue
            theme_bg = "#F0F7FF"        # Alice Blue
            badge_text = "● SEEN TEST"

        # Load image
        img = load_test_image(item["file"], dataset_root)
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Colored border around the image
        for spine in ax_img.spines.values():
            spine.set_color(theme_color)
            spine.set_linewidth(3.5)

        # Title badge above image
        card_rank = slot_idx + 1
        img_title = f"{badge_text}  |  #{card_rank} Most Wrong  (Margin +{item['margin']:.2f})"
        ax_img.set_title(
            img_title,
            fontsize=10.5,
            fontweight="bold",
            color="white",
            pad=6,
            bbox=dict(boxstyle="square,pad=0.4", facecolor=theme_color, edgecolor="none"),
        )

        # Configure text box
        ax_txt.axis("off")
        rect = patches.FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="square,pad=0.02",
            facecolor=theme_bg,
            edgecolor=theme_color,
            linewidth=1.2,
            transform=ax_txt.transAxes,
            clip_on=False,
        )
        ax_txt.add_patch(rect)

        # Build clean formatted text
        crop_diag = "✓" if item["crop_correct"] else "✗"
        dis_diag = "✓" if item["disease_correct"] else "✗"
        pred_badge = "Unseen" if item["pred_is_unseen"] else "Seen"

        lines = [
            f"FILE: {item['file']}",
            f"TRUE: {item['true_crop']}  //  {item['true_disease']}",
            f"PRED: {item['pred_crop_name']}  //  {item['pred_disease_name']} ({pred_badge})",
            f"CHECK: Plant {crop_diag}  |  Disease {dis_diag}  |  True Rank #{item['true_pair_rank']}",
            "─" * 46,
            f"PLANTS (GT: {item['true_crop']} {item['true_crop_prob']:.1%}, #{item['true_crop_rank']}):",
        ]

        # Top 5 Plants
        for idx, cp in enumerate(item["top5_plants"], 1):
            star = " (GT)" if cp["is_gt"] else ""
            c_name = cp["name"].replace("_", " ")[:18]
            lines.append(f"  {idx}. {c_name:<18} {cp['prob']:>5.1%}{star}")

        lines.append("─" * 46)
        lines.append(f"DISEASES (GT: {item['true_disease']} {item['true_disease_prob']:.1%}, #{item['true_disease_rank']}):")

        # Top 5 Diseases
        for idx, dp in enumerate(item["top5_diseases"], 1):
            star = " (GT)" if dp["is_gt"] else ""
            d_name = dp["name"].replace("_", " ")[:18]
            lines.append(f"  {idx}. {d_name:<18} {dp['prob']:>5.1%}{star}")

        block_text = "\n".join(lines)
        ax_txt.text(
            0.04, 0.96,
            block_text,
            transform=ax_txt.transAxes,
            fontsize=8.2,
            fontfamily="monospace",
            verticalalignment="top",
            color="#2D3748",
            linespacing=1.25,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_html_gallery(
    records: list[dict],
    output_path: Path,
    dataset_root: Path | None = None,
    cols: int = 5,
    rows: int = 4,
    title: str = "Misclassification Gallery",
    subtitle: str | None = None,
) -> Path:
    """Render an interactive, self-contained HTML gallery with embedded base64 thumbnails."""
    total_slots = cols * rows
    display_records = records[:total_slots]

    cards_html = []
    for idx, item in enumerate(display_records, 1):
        is_unseen = item["is_unseen"]
        badge_cls = "badge-unseen" if is_unseen else "badge-seen"
        badge_lbl = "UNSEEN TEST" if is_unseen else "SEEN TEST"

        img = load_test_image(item["file"], dataset_root)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        plant_rows = "".join([
            f"<tr class='{'gt-row' if p['is_gt'] else ''}'><td>{i}. {p['name']}</td>"
            f"<td style='text-align:right'><b>{p['prob']:.1%}</b>{' ★' if p['is_gt'] else ''}</td></tr>"
            for i, p in enumerate(item["top5_plants"], 1)
        ])
        disease_rows = "".join([
            f"<tr class='{'gt-row' if d['is_gt'] else ''}'><td>{i}. {d['name']}</td>"
            f"<td style='text-align:right'><b>{d['prob']:.1%}</b>{' ★' if d['is_gt'] else ''}</td></tr>"
            for i, d in enumerate(item["top5_diseases"], 1)
        ])

        crop_icon = "<span class='correct'>✓ Plant</span>" if item["crop_correct"] else "<span class='wrong'>✗ Plant</span>"
        dis_icon = "<span class='correct'>✓ Disease</span>" if item["disease_correct"] else "<span class='wrong'>✗ Disease</span>"

        card = f"""
        <div class="card {'card-unseen' if is_unseen else 'card-seen'}">
            <div class="card-header">
                <span class="badge {badge_cls}">{badge_lbl}</span>
                <span class="rank">#{idx} Most Wrong</span>
                <span class="margin">+{item['margin']:.2f}</span>
            </div>
            <div class="img-box">
                <img src="data:image/jpeg;base64,{b64_str}" alt="{item['file']}">
            </div>
            <div class="card-body">
                <div class="pair-info">
                    <div><b>True:</b> {item['true_crop']} · {item['true_disease']}</div>
                    <div><b>Pred:</b> {item['pred_crop_name']} · {item['pred_disease_name']} ({'Unseen' if item['pred_is_unseen'] else 'Seen'})</div>
                    <div class="tags">{crop_icon} {dis_icon} <span class="tag">Rank #{item['true_pair_rank']}</span></div>
                </div>
                <div class="tables-grid">
                    <div class="col-table">
                        <div class="table-hdr">Plants (GT: {item['true_crop_prob']:.1%}, #{item['true_crop_rank']})</div>
                        <table>{plant_rows}</table>
                    </div>
                    <div class="col-table">
                        <div class="table-hdr">Diseases (GT: {item['true_disease_prob']:.1%}, #{item['true_disease_rank']})</div>
                        <table>{disease_rows}</table>
                    </div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ text-align: center; margin: 0 0 6px 0; font-size: 26px; }}
        .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 24px; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 16px; max-width: 1900px; margin: 0 auto; }}
        @media (max-width: 1400px) {{ .grid {{ grid-template-columns: repeat(3, 1fr); }} }}
        @media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
        @media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        .card {{ background: #1e293b; border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; border: 2px solid transparent; }}
        .card-unseen {{ border-color: #ea580c; }}
        .card-seen {{ border-color: #2563eb; }}
        .card-header {{ padding: 10px 12px; display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.25); font-size: 12px; font-weight: bold; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .badge-unseen {{ background: #ea580c; color: white; }}
        .badge-seen {{ background: #2563eb; color: white; }}
        .margin {{ color: #fbbf24; font-family: monospace; }}
        .img-box {{ width: 100%; aspect-ratio: 1; overflow: hidden; background: #000; }}
        .img-box img {{ width: 100%; height: 100%; object-fit: cover; transition: transform 0.2s; }}
        .img-box:hover img {{ transform: scale(1.05); }}
        .card-body {{ padding: 12px; font-size: 12px; display: flex; flex-direction: column; gap: 8px; flex: 1; }}
        .pair-info div {{ margin-bottom: 3px; word-break: break-word; }}
        .tags {{ margin-top: 4px; display: flex; gap: 6px; font-size: 11px; }}
        .correct {{ color: #4ade80; font-weight: bold; }}
        .wrong {{ color: #f87171; font-weight: bold; }}
        .tag {{ background: #334155; padding: 1px 6px; border-radius: 4px; color: #cbd5e1; }}
        .tables-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; font-size: 11px; }}
        .col-table {{ background: #0f172a; border-radius: 6px; padding: 6px; }}
        .table-hdr {{ font-weight: bold; margin-bottom: 4px; color: #cbd5e1; border-bottom: 1px solid #334155; padding-bottom: 2px; font-size: 10.5px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        td {{ padding: 2px 1px; }}
        .gt-row {{ color: #38bdf8; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="subtitle">{subtitle or ''}</div>
    <div class="grid">
        {''.join(cards_html)}
    </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def sync_gallery_to_drive(file_path: Path, sync_dir: Path | None, run_name: str | None = None):
    """Mirror generated gallery image / HTML to Google Drive or OneDrive runs folder."""
    if sync_dir is None or not file_path.exists():
        return
    try:
        sync_base = Path(sync_dir)
        dest_folder = (sync_base / run_name) if run_name else sync_base
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_file = dest_folder / file_path.name
        shutil.copy2(file_path, dest_file)
        print(f"Synced gallery -> {dest_file}", flush=True)
    except Exception as e:
        print(f"Warning: Failed to sync gallery to {sync_dir}: {e}", flush=True)


def generate_misclassification_gallery(
    run_dir: str | Path,
    split: str = "both",
    cols: int = 5,
    rows: int = 4,
    gamma: float | None = None,
    dataset_root: str | Path | None = None,
    output: str | Path | None = None,
    output_html: str | Path | None = None,
    sync_dir: str | Path | None = None,
    sort_by: str = "margin",
    device: str = "auto",
    make_html: bool = False,
    force_infer: bool = False,
) -> tuple[Path, Path | None]:
    """Main programmatic API to generate and sync misclassification galleries."""
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    # Load configuration
    cfg_path = run_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_path}")
    cfg = load_config(cfg_path)
    version = cfg.get("version", run_path.name)

    # Determine gamma
    if gamma is None:
        gamma_file = run_path / "metrics_gamma.json"
        if not gamma_file.exists():
            gamma_file = run_path / "_gamma_metrics.json"  # legacy fallback
        metrics_file = run_path / "metrics.json"
        if gamma_file.exists():
            try:
                m = json.loads(gamma_file.read_text(encoding="utf-8"))
                gamma = float(m.get("gamma", 0.0))
            except Exception:
                gamma = None
        if gamma is None and metrics_file.exists():
            try:
                m = json.loads(metrics_file.read_text(encoding="utf-8"))
                gamma = float(m.get("gamma", 0.0))
            except Exception:
                gamma = float(cfg.get("evaluation", {}).get("gamma", 0.0))
        if gamma is None:
            gamma = float(cfg.get("evaluation", {}).get("gamma", 0.0))

    # Read dataset metadata
    ds_root = Path(dataset_root) if dataset_root else None
    if ds_root is not None:
        os.environ["DPD_ROOT"] = str(ds_root)
    df_meta = read_metadata()
    pairs = pair_table(df_meta)

    # Resolve image folder
    image_dir = (ds_root / "DPD_Small_dataset") if ds_root else (Path(os.environ.get("DPD_ROOT", "DPD_Small")) / "DPD_Small_dataset")

    # Load or compute model log-probabilities
    head_logp = get_or_compute_head_logp(
        run_path,
        cfg,
        df_meta,
        image_dir,
        device=device,
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        num_workers=cfg.get("runtime", {}).get("num_workers", 2),
        force_infer=force_infer,
    )

    # Find and rank misclassifications
    records = collect_misclassifications(
        run_dir=run_path,
        cfg=cfg,
        df_meta=df_meta,
        pairs=pairs,
        gamma=gamma,
        head_logp=head_logp,
        split_filter=split,
        sort_by=sort_by,
    )

    print(f"\n=======================================================")
    print(f"Misclassification Gallery Generator")
    print(f"Run:                  {run_path.name}")
    print(f"Split selection:      {split.upper()}")
    print(f"Calibrated Gamma:     {gamma}")
    print(f"Total Misclassified:  {len(records)}")
    print(f"Display Grid:         {cols} columns x {rows} rows ({cols * rows} images)")
    print(f"Sorted By:            Most Wrong ({sort_by})")
    print(f"=======================================================\n", flush=True)

    if not records:
        print(f"Note: No misclassified images found for split filter {split!r} (100% accuracy)!")
        return Path(), None

    # Determine output path
    split_tag = split.lower().replace("test_", "")
    if output is None:
        out_png = run_path / f"misclassification_gallery_{split_tag}.png"
    else:
        out_png = Path(output)

    title = f"Misclassification Gallery ({split_tag.upper()} Test Set) - Top {cols * rows} Most Wrong"
    subtitle = f"Model: {version}  |  Gamma: {gamma:.2f}  |  Total Mistakes: {len(records)}  |  Grid: {cols}x{rows}"

    saved_png = render_matplotlib_gallery(
        records=records,
        output_path=out_png,
        dataset_root=ds_root,
        cols=cols,
        rows=rows,
        title=title,
        subtitle=subtitle,
    )
    print(f"Saved gallery image -> {saved_png}")

    saved_html = None
    if make_html or output_html is not None:
        if output_html is None:
            out_html = run_path / f"misclassification_gallery_{split_tag}.html"
        else:
            out_html = Path(output_html)
        saved_html = render_html_gallery(
            records=records,
            output_path=out_html,
            dataset_root=ds_root,
            cols=cols,
            rows=rows,
            title=title,
            subtitle=subtitle,
        )
        print(f"Saved interactive HTML gallery -> {saved_html}")

    # Mirror to remote sync directory (Google Drive / OneDrive)
    if sync_dir:
        sync_gallery_to_drive(saved_png, Path(sync_dir), run_name=run_path.name)
        if saved_html:
            sync_gallery_to_drive(saved_html, Path(sync_dir), run_name=run_path.name)

    return saved_png, saved_html


def parse_args():
    parser = argparse.ArgumentParser(description="Generate galleries of misclassified test images for multi-task CNN.")
    parser.add_argument("--run-dir", default=None, help="Directory containing model run outputs (config.json, weights, etc.)")
    parser.add_argument("--config", default=None, help="Path to config JSON (default: run_dir/config.json)")
    parser.add_argument("--split", choices=["seen", "unseen", "both", "all"], default="both", help="Test split to include: seen only, unseen only, or both (default: both)")
    parser.add_argument("--seen-only", action="store_true", help="Shortcut for --split seen")
    parser.add_argument("--unseen-only", action="store_true", help="Shortcut for --split unseen")
    parser.add_argument("--include-seen", action="store_true", default=None, help="Include seen test set")
    parser.add_argument("--exclude-seen", action="store_true", default=None, help="Exclude seen test set")
    parser.add_argument("--include-unseen", action="store_true", default=None, help="Include unseen test set")
    parser.add_argument("--exclude-unseen", action="store_true", default=None, help="Exclude unseen test set")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns in gallery grid (default: 5)")
    parser.add_argument("--rows", type=int, default=4, help="Number of rows in gallery grid (default: 4)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K predicted plants & diseases to show (default: 5)")
    parser.add_argument("--sort-by", choices=["margin", "rank", "prob"], default="margin", help="Wrongness sorting criteria (default: margin)")
    parser.add_argument("--gamma", type=float, default=None, help="Calibration gamma (default: read from metrics.json)")
    parser.add_argument("--output", default=None, help="Output image file path (default: run_dir/misclassification_gallery_<split>.png)")
    parser.add_argument("--output-html", default=None, help="Output interactive HTML gallery file path")
    parser.add_argument("--html", action="store_true", help="Also generate an interactive HTML gallery file")
    parser.add_argument("--sync-dir", default=None, help="Remote directory (e.g. Google Drive/OneDrive runs folder) to mirror gallery files")
    parser.add_argument("--dataset-root", default=None, help="Root path of DPD dataset (default: $DPD_ROOT or DPD_Small)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--force-infer", action="store_true", help="Force re-running inference even if head_scores.npz exists")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve split selection
    split = args.split
    if args.seen_only or (args.exclude_unseen and not args.exclude_seen):
        split = "seen"
    elif args.unseen_only or (args.exclude_seen and not args.exclude_unseen):
        split = "unseen"
    elif args.include_seen and not args.include_unseen:
        split = "seen"
    elif args.include_unseen and not args.include_seen:
        split = "unseen"

    # Auto-resolve run_dir if not specified
    run_dir = args.run_dir
    if run_dir is None:
        if args.config:
            cfg = load_config(args.config)
            from dpd.config import output_dir
            run_dir = output_dir(cfg)
        else:
            candidates = sorted(Path("runs/multitask").glob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
            if candidates:
                run_dir = candidates[0]
                print(f"Auto-selected latest run directory: {run_dir}")
            else:
                raise ValueError("No --run-dir specified and no runs found in runs/multitask.")

    generate_misclassification_gallery(
        run_dir=run_dir,
        split=split,
        cols=args.cols,
        rows=args.rows,
        gamma=args.gamma,
        dataset_root=args.dataset_root,
        output=args.output,
        output_html=args.output_html,
        sync_dir=args.sync_dir,
        sort_by=args.sort_by,
        device=args.device,
        make_html=args.html or (args.output_html is not None),
        force_infer=args.force_infer,
    )


if __name__ == "__main__":
    main()
