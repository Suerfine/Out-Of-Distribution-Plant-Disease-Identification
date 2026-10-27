import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from dpd.data import IMAGE_DIR, pair_table
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
run_dir, gamma = str(run["dir"]), float(run["metrics"].get("gamma", 0.0))

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
        order, _, probs = rank_scores(raw, unseen_mask, m.gamma, m.logit_scale)
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
    order, scores, probs = rank_scores(raw, unseen_mask, m.gamma, m.logit_scale)
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

st.caption(f"{run['version']}: calibration γ = {gamma:g} subtracted from seen-pair scores. "
           "Test-set predictions are read from the run's saved scores; *Details* re-runs the model on the image.")
