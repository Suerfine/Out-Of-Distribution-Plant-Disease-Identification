import io
import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from dpd.data import IMAGE_DIR, pair_table
from dpd.evaluate import evaluate
from dpd.inference import list_runs, nearest, rank_scores
from dpd.ui import (
    OUTCOMES, SPLIT_LABELS, best_run_index, get_model, get_train_bank, image_grid, load_metadata, outcome, run_label,
    test_predictions, thumbnail, thumbnail_bytes,
)

BADGE = {
    "Correct": ("green", ":material/check_circle:"),
    "Wrong crop": ("yellow", ":material/eco:"),
    "Wrong disease": ("orange", ":material/coronavirus:"),
    "Both wrong": ("red", ":material/cancel:"),
    "No label": ("gray", ":material/help:"),
}
SORTS = ["Random", "Least confident first", "Most confident first", "True pair"]


@st.cache_data(max_entries=256, show_spinner=False)
def predict_bytes(run_dir: str, data: bytes):
    """(retrieval embedding, uncalibrated pair scores) for one uploaded image; cached so reruns don't re-run the model."""
    return get_model(run_dir).predict(io.BytesIO(data))


@st.cache_data(max_entries=64, show_spinner="Computing Grad-CAM attention…")
def explain_bytes(run_dir: str, data: bytes, pred_crop_id: int, pred_disease_id: int):
    """Grad-CAM explanation for an image; cached so slider/toggle changes don't re-run backprop."""
    return get_model(run_dir).explain(io.BytesIO(data), pred_crop_id, pred_disease_id)


def check_gradcam_support(run_cfg: dict) -> tuple[bool, str]:
    bb = run_cfg.get("model", {}).get("backbone", "").lower()
    method = run_cfg.get("method", "")
    if "vit" in bb:
        return False, "ViT backbones do not use 2D convolutional feature maps"
    if method in ("clip_zeroshot", "clip_prompt_learning", "clip_prompt_lora", "clip_feature_gen"):
        return False, "VLM / CLIP zero-shot models compute multimodal embeddings without convolutional classification heads"
    if method == "ensemble":
        return False, "Multi-model ensembles do not support direct single-backbone Grad-CAM"
    if "cnn" in method:
        return True, "Grad-CAM supported"
    return False, "Unsupported architecture"


@st.cache_data(max_entries=64, show_spinner=False)
def gamma_metrics(run_dir: str, gamma: float) -> dict:
    """Test-set results at any gamma, from the run's saved raw scores (no model needed, milliseconds)."""
    z = np.load(Path(run_dir) / "scores.npz")
    r = evaluate(z["test_seen_scores"], z["test_seen_labels"], z["test_unseen_scores"], z["test_unseen_labels"],
                 pair_table(load_metadata()), gamma, (1, 5))
    return {"seen": r["test_seen"]["top1"], "unseen": r["test_unseen"]["top1"], "hm": r["harmonic_mean_top1"],
            "hm5": r["harmonic_mean_top5"], "seen_to_unseen": r["test_seen"]["predicted_as_unseen"]}


@st.cache_data(max_entries=16, show_spinner=False)
def gamma_curve(run_dir: str, grid: tuple) -> pd.DataFrame:
    rows = []
    for g in np.round(np.arange(*grid), 6):
        m = gamma_metrics(run_dir, float(g))
        rows += [{"gamma": g, "metric": "Seen Top-1", "value": m["seen"]},
                 {"gamma": g, "metric": "Unseen Top-1", "value": m["unseen"]},
                 {"gamma": g, "metric": "HM Top-1", "value": m["hm"]}]
    return pd.DataFrame(rows)


def gamma_grid(run: dict) -> tuple:
    """The (start, stop, step) range validation searched for this run; zero-shot runs have none, so use the CLIP one."""
    grid = run["cfg"].get("validation", {}).get("gamma_grid")
    if isinstance(grid, (list, tuple)) and len(grid) == 3:
        return tuple(grid)
    return (0.0, 10.0, 0.1) if ("cnn" in run["method"]) else (0.0, 0.1, 0.0025)


df = load_metadata()
upload_bytes = {}
by_file = df.set_index("file")
runs = list_runs()
if not runs:
    st.info("No finished runs in `runs/`. Train or evaluate a model first.", title="No model yet", icon=":material/model_training:")
    st.stop()

with st.container(horizontal=True, vertical_alignment="bottom"):
    run = st.selectbox("Model", runs, index=best_run_index(runs), format_func=run_label, key="infer_run")
    source = st.segmented_control(
        "Images", ["Test set", "Upload"], default="Test set", required=True, key="infer_source"
    )
run_dir, reported_gamma = str(run["dir"]), float(run["metrics"].get("gamma", 0.0))
gc_supported, gc_msg = check_gradcam_support(run["cfg"])
if gc_supported:
    st.caption(":material/visibility: **Grad-CAM supported**: Visual attention heatmaps available in Image Details.")
else:
    st.caption(f":material/visibility_off: **Grad-CAM disabled**: {gc_msg}.")

# ---- Calibration gamma: adjustable for exploration; reported results always use the validation choice ----
grid = gamma_grid(run)
gamma_key = f"infer_gamma_{run['version']}"  # one slider state per model, so switching models restores its own value
if gamma_key not in st.session_state:
    st.session_state[gamma_key] = reported_gamma
with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        gamma = st.slider(
            "Calibration γ", min_value=float(grid[0]), max_value=float(max(grid[1], reported_gamma)), step=float(grid[2]),
            key=gamma_key, format="%.4f" if grid[2] < 0.01 else "%.2f",
            help="Subtracted from every seen pair's score before ranking. Higher γ favours the 10 unseen pairs. "
                 "Changing it needs no retraining: it is applied after the model scores the image.",
        )
        st.button("Reset to validation γ", icon=":material/restart_alt:",
                  on_click=lambda: st.session_state.update({gamma_key: reported_gamma}))
    live, ref = gamma_metrics(run_dir, gamma), gamma_metrics(run_dir, reported_gamma)
    delta = lambda k: None if gamma == reported_gamma else f"{(live[k] - ref[k]) * 100:+.1f} pts"
    with st.container(horizontal=True):
        st.metric("Seen Top-1", f"{live['seen']:.1%}", delta('seen'), border=True)
        st.metric("Unseen Top-1", f"{live['unseen']:.1%}", delta('unseen'), border=True)
        st.metric("HM Top-1", f"{live['hm']:.1%}", delta('hm'), border=True)
        st.metric("HM Top-5", f"{live['hm5']:.1%}", delta('hm5'), border=True)
        st.metric("Seen images → unseen pairs", f"{live['seen_to_unseen']:.1%}", delta('seen_to_unseen'),
                  delta_color="inverse", border=True)
    if gamma != reported_gamma:
        st.warning(
            f"Exploration only. Reported results use γ = {reported_gamma:g}, chosen on validation; picking γ by "
            "looking at these test numbers would be tuning on the test set.", icon=":material/science:",
        )
    with st.expander("Seen / unseen trade-off across γ", icon=":material/show_chart:"):
        curve = gamma_curve(run_dir, grid)
        lines = alt.Chart(curve).mark_line().encode(
            x=alt.X("gamma:Q", title="Calibration γ"),
            y=alt.Y("value:Q", title="Test accuracy", axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", title=None),
            tooltip=["metric", alt.Tooltip("gamma:Q", format=".4f"), alt.Tooltip("value:Q", format=".1%")],
        )
        marks = pd.DataFrame({"gamma": [reported_gamma, gamma], "which": ["validation γ", "current γ"]})
        rules = alt.Chart(marks).mark_rule(strokeDash=[4, 4]).encode(
            x="gamma:Q", color=alt.Color("which:N", title=None, scale=alt.Scale(range=["#888888", "#e45756"])),
            tooltip=["which", alt.Tooltip("gamma:Q", format=".4f")],
        )
        st.altair_chart(alt.layer(lines, rules).resolve_scale(color="independent").properties(height=280),
                        alt="Seen, unseen and harmonic-mean accuracy as calibration gamma increases")
        st.caption("Raising γ moves images from seen to unseen pairs: unseen accuracy rises, seen accuracy falls, "
                   "and the harmonic mean peaks in between.")

# Every image becomes one row: file name, image bytes loader key, true pair id (or -1), predicted top-5 pair ids,
# top-1 confidence and outcome. Test-set rows come from the run's saved scores (instant, identical to metrics.json);
# uploads run through the model.
if source == "Test set":
    preds = test_predictions(run_dir, run["method"], gamma)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        split = st.segmented_control(
            "Test split", ["test_seen", "test_unseen", "both"], default="test_unseen", required=True,
            format_func=lambda s: SPLIT_LABELS.get(s, "Both"), key="infer_split",
        )
        if split != "both":
            preds = preds[preds["split"] == split]
        true_names = df.set_index("pair_id")["pair"].groupby(level=0).first()
        pair_filter = st.selectbox(
            "True pair", ["Any pair"] + sorted(true_names[preds["true_id"].unique()]), key="infer_pair"
        )
        if pair_filter != "Any pair":
            preds = preds[true_names[preds["true_id"]].to_numpy() == pair_filter]
    items = preds.assign(name=preds["file"], uploaded=False)
else:
    ups = st.file_uploader(
        "Plant images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="infer_upload",
        help="Images whose file name matches a DPD-Small image are scored against its true label.",
    )
    if not ups:
        st.caption("Upload one or more leaf photos to identify them.")
        st.stop()
    m = get_model(run_dir)
    unseen_mask = m.pairs["is_unseen"].to_numpy()
    crop_ids, disease_ids = m.pairs["crop_id"].to_numpy(), m.pairs["disease_id"].to_numpy()
    rows = []
    bar = st.progress(0.0, text="Identifying…")
    for i, up in enumerate(ups):
        _, raw = predict_bytes(run_dir, up.getvalue())
        order, _, probs = rank_scores(raw, unseen_mask, gamma, m.logit_scale)
        true_id = int(by_file.loc[up.name, "pair_id"]) if up.name in by_file.index else -1
        rows.append({"name": up.name, "uploaded": True, "true_id": true_id, "pred_id": order[0], "topk": order[:5],
                     "confidence": probs[order[0]]})
        bar.progress((i + 1) / len(ups), text=f"Identifying… {i + 1}/{len(ups)}")
    bar.empty()
    items = pd.DataFrame(rows)
    labelled = items["true_id"] >= 0
    t, p = items["true_id"].where(labelled, 0), items["pred_id"]
    items["outcome"] = np.where(labelled, outcome(crop_ids[p] == crop_ids[t], disease_ids[p] == disease_ids[t]), "No label")
    upload_bytes = {up.name: up.getvalue() for up in ups}

pairs = pair_table(df).set_index("pair_id")

# Outcome counts before the outcome filter, so the filter labels show what each option would keep.
counts = items["outcome"].value_counts()
present = [o for o in OUTCOMES + ["No label"] if counts.get(o, 0)]
with st.container(horizontal=True, vertical_alignment="bottom"):
    shown = st.pills(
        "Show", present, default=present, selection_mode="multi", key=f"infer_outcomes_{source}",
        format_func=lambda o: f"{o} ({counts.get(o, 0):,})",
        help="Wrong crop: right disease on the wrong crop. Wrong disease: right crop, wrong disease. Both wrong: neither matches.",
    )
    sort = st.selectbox("Sort", SORTS, key="infer_sort", width=220)
    if sort == "Random" and st.button("Shuffle", icon=":material/shuffle:"):
        st.session_state.infer_seed = int(np.random.default_rng().integers(1 << 31))

labelled = items[items["outcome"] != "No label"]
with st.container(horizontal=True):
    st.metric("Images", f"{len(items):,}", border=True)
    if len(labelled):
        st.metric("Pair Top-1", f"{(labelled['outcome'] == 'Correct').mean():.1%}", border=True,
                  help="Share of these images whose top prediction is the true crop–disease pair")
        st.metric("Crop right", f"{labelled['outcome'].isin(['Correct', 'Wrong disease']).mean():.1%}", border=True)
        st.metric("Disease right", f"{labelled['outcome'].isin(['Correct', 'Wrong crop']).mean():.1%}", border=True)
        st.metric("Top-5", f"{np.mean([t in k for t, k in zip(labelled['true_id'], labelled['topk'])]):.1%}", border=True)

view = items[items["outcome"].isin(shown or [])]
if sort == "Random":
    view = view.sample(frac=1, random_state=st.session_state.get("infer_seed", 0))
elif sort == "Least confident first":
    view = view.sort_values("confidence")
elif sort == "Most confident first":
    view = view.sort_values("confidence", ascending=False)
else:
    view = view.assign(true_pair=pairs["pair"].reindex(view["true_id"]).to_numpy()).sort_values(["true_pair", "name"])

if view.empty:
    st.info("No images match these filters.")
    st.stop()


@st.dialog("Image details", width="large")
def show_details(name: str, uploaded: bool) -> None:
    data = upload_bytes[name] if uploaded else (IMAGE_DIR / name).read_bytes()
    m = get_model(run_dir)
    unseen_mask = m.pairs["is_unseen"].to_numpy()
    emb, raw = predict_bytes(run_dir, data) if uploaded else m.predict(IMAGE_DIR / name)
    order, scores, probs = rank_scores(raw, unseen_mask, gamma, m.logit_scale)
    top = m.pairs.iloc[order[0]]
    truth = by_file.loc[name] if name in by_file.index else None

    left, right = st.columns([2, 3])
    with left:
        st.image(Image.open(io.BytesIO(data)).convert("RGB"), width="stretch", alt="Plant image being identified")
        if truth is not None:
            st.caption(f"True label: **{truth['pair']}** · {SPLIT_LABELS[truth['split']]} · source {truth['source']}")
        else:
            st.caption(name)
    with right:
        with st.container(horizontal=True):
            st.metric("Prediction", top["pair"].replace("_", " "), border=True)
            st.metric("Confidence", f"{probs[order[0]]:.0%}", border=True)
        if top["is_unseen"]:
            st.info(
                "This pair never appeared in training. It was recognised by combining what the model learned "
                f"about **{top['crop']}** and about **{top['disease'].replace('_', ' ')}** from other pairs.",
                icon=":material/join_inner:",
            )
        top5 = m.pairs.iloc[order[:5]].assign(
            confidence=probs[order[:5]], score=scores[order[:5]], status=np.where(unseen_mask[order[:5]], "unseen", "seen"),
        )[["pair", "crop", "disease", "status", "confidence", "score"]]
        st.dataframe(
            top5, hide_index=True, alt="Top five predicted crop-disease pairs",
            column_config={
                "confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0.0, max_value=1.0),
                "score": st.column_config.NumberColumn("Score", format="%.3f", help=m.score_help),
            },
        )

    # ---- Visual Explanations (Grad-CAM) ----
    st.divider()
    st.markdown("#### Model Attention (Grad-CAM)")
    if not getattr(m, "supports_gradcam", False):
        st.info(
            f"**Grad-CAM is unavailable for this model:** {getattr(m, 'gradcam_reason', 'Unsupported model type')}. "
            "To inspect Grad-CAM attention heatmaps, select a convolutional model (e.g. Multi-Task CNN or Independent CNN).",
            icon=":material/visibility_off:",
        )
    else:
        pred_c, pred_d = int(top["crop_id"]), int(top["disease_id"])
        expl = explain_bytes(run_dir, data, pred_c, pred_d)
        if not expl.get("supported"):
            st.warning(f"Grad-CAM could not be computed: {expl.get('reason', 'Unknown error')}", icon=":material/warning:")
        else:
            cams = expl.get("cams", {})
            orig_pil = Image.open(io.BytesIO(data)).convert("RGB")
            with st.container(border=True):
                c_ctrl1, c_ctrl2 = st.columns([3, 2], vertical_alignment="bottom")
                with c_ctrl1:
                    view_opts = ["Side-by-Side"] + [k for k in ("disease", "crop", "combined") if k in cams]
                    view_mode = st.segmented_control(
                        "Attention Target",
                        view_opts,
                        default="Side-by-Side",
                        format_func=lambda k: {
                            "Side-by-Side": "Side-by-Side View",
                            "disease": "Disease Focus",
                            "crop": "Crop Focus",
                            "combined": "Combined Pair",
                        }.get(k, k),
                        key=f"cam_target_{name}",
                    )
                with c_ctrl2:
                    cam_alpha = st.slider(
                        "Heatmap Opacity", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                        key=f"cam_alpha_{name}",
                        help="Blend level between the original leaf photo and the attention heatmap."
                    )

                from dpd.gradcam import overlay_cam

                if view_mode == "Side-by-Side":
                    show_keys = [k for k in ("disease", "crop") if k in cams]
                    cols = st.columns(1 + len(show_keys))
                    with cols[0]:
                        st.image(orig_pil, caption="Original Input", width="stretch")
                    for col, k in zip(cols[1:], show_keys):
                        entry = cams[k]
                        blended = overlay_cam(orig_pil, entry["cam"], alpha=cam_alpha)
                        with col:
                            st.image(blended, caption=entry["title"], width="stretch")
                            st.caption(entry["desc"])
                else:
                    entry = cams[view_mode]
                    blended = overlay_cam(orig_pil, entry["cam"], alpha=cam_alpha)
                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.image(orig_pil, caption="Original Input", width="stretch")
                    with c_right:
                        st.image(blended, caption=entry["title"], width="stretch")
                        st.caption(entry["desc"])
    st.divider()

    st.markdown("**Similar training images**")
    train, bank = get_train_bank(run_dir)
    idx, _ = nearest(emb, bank, 8)
    image_grid(train.iloc[idx], n_cols=4, caption_fields=("pair",))

    same_pair = train[train["pair_id"] == top["pair_id"]]
    if len(same_pair):
        st.markdown(f"**Training images of {top['pair']}** ({len(same_pair)})")
        image_grid(same_pair.head(4), n_cols=4, caption_fields=("source",))
    else:
        st.caption(f"No training images of {top['pair']}: it is an unseen pair. These are the pieces it was built from.")
    c1, c2 = st.columns(2)
    with c1:
        same_crop = train[(train["crop_id"] == top["crop_id"]) & (train["pair_id"] != top["pair_id"])]
        st.markdown(f"**Same crop ({top['crop']}), other diseases** ({len(same_crop)})")
        image_grid(same_crop.sample(min(4, len(same_crop)), random_state=0), n_cols=2, caption_fields=("disease",))
    with c2:
        same_dis = train[(train["disease_id"] == top["disease_id"]) & (train["pair_id"] != top["pair_id"])]
        st.markdown(f"**Same disease ({top['disease'].replace('_', ' ')}), other crops** ({len(same_dis)})")
        image_grid(same_dis.sample(min(4, len(same_dis)), random_state=0), n_cols=2, caption_fields=("crop",))


with st.container(horizontal=True, vertical_alignment="bottom"):
    n_cols = st.segmented_control("Columns", [4, 6, 8], default=6, required=True, key="infer_cols")
    per_page = n_cols * 4
    n_pages = math.ceil(len(view) / per_page)
    if st.session_state.get("infer_page", 1) > n_pages:  # filters shrank the list
        st.session_state.infer_page = 1
    page_no = st.number_input(f"Page (of {n_pages})", 1, n_pages, 1, key="infer_page", width=160)
    st.caption(f"{len(view):,} images match")
page = view.iloc[(page_no - 1) * per_page :][:per_page]

for start in range(0, len(page), n_cols):
    for col, r in zip(st.columns(n_cols), page.iloc[start : start + n_cols].itertuples(index=False)):
        pred = pairs.loc[r.pred_id]
        with col, st.container(border=True, height="stretch"):
            st.image(thumbnail_bytes(upload_bytes[r.name]) if r.uploaded else thumbnail(r.name), width="stretch",
                     alt=f"Predicted {pred['pair'].replace('_', ' ')}")
            color, icon = BADGE[r.outcome]
            st.badge(r.outcome, color=color, icon=icon)
            st.markdown(f"**{pred['pair'].replace('_', ' ')}** · {r.confidence:.0%}"
                        + (" · :violet-badge[unseen]" if pred["is_unseen"] else ""))
            if r.true_id >= 0 and r.outcome != "Correct":
                st.caption(f"True: {pairs.loc[r.true_id, 'pair'].replace('_', ' ')}")
            if st.button("Details", key=f"det_{r.name}", icon=":material/open_in_full:", type="tertiary"):
                show_details(r.name, r.uploaded)

st.caption(f"{run['version']}: calibration γ = {gamma:g} subtracted from seen-pair scores"
           + ("" if gamma == reported_gamma else f" (validation choice: {reported_gamma:g})") + ". "
           "Test-set predictions are read from the run's saved scores; *Details* re-runs the model on the image.")
