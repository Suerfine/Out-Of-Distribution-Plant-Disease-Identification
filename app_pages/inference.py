import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from dpd.data import load_image
from dpd.inference import embed_image, list_runs, load_run_model, nearest, rank_pairs, train_embeddings
from dpd.ui import SPLIT_LABELS, image_grid, load_metadata


@st.cache_resource(max_entries=2, show_spinner="Loading model…")
def get_model(run_dir: str):
    run = next(r for r in list_runs() if str(r["dir"]) == run_dir)
    return load_run_model(run, load_metadata())


@st.cache_resource(max_entries=2, show_spinner="Loading training-image embeddings…")
def get_train_bank(run_dir: str):
    run = next(r for r in list_runs() if str(r["dir"]) == run_dir)
    m = get_model(run_dir)
    train = load_metadata().query("split == 'train'").reset_index(drop=True)
    return train, train_embeddings(m["clip"], m["preprocess"], run["cfg"]["model"], train["file"])


df = load_metadata()
runs = list_runs()
if not runs:
    st.info("No finished runs in `runs/`. Train or evaluate a model first.", title="No model yet", icon=":material/model_training:")
    st.stop()


def run_label(r):
    m = r["metrics"]
    kind = "trained prompts" if r["method"] == "clip_prompt_learning" else "zero-shot"
    return f"{r['version']} · {kind} · test HM {m['harmonic_mean_top1']:.1%}"


with st.container(horizontal=True, vertical_alignment="bottom"):
    best_idx = max(range(len(runs)), key=lambda i: runs[i]["metrics"]["harmonic_mean_top1"])
    run = st.selectbox("Model", runs, index=best_idx, format_func=run_label, key="infer_run")
    source = st.segmented_control(
        "Image", ["Test image", "Upload"], default="Test image", required=True, key="infer_source"
    )

metrics = run["metrics"]
with st.container(horizontal=True):
    st.metric("Seen Top-1", f"{metrics['test_seen']['top1']:.1%}", border=True,
              help="Accuracy on the 2,000 test images of the 323 seen pairs, ranked against all 333 pairs.")
    st.metric("Unseen Top-1", f"{metrics['test_unseen']['top1']:.1%}", border=True,
              help="Accuracy on the 1,448 test images of the 10 pairs never seen in training.")
    st.metric("HM Top-1", f"{metrics['harmonic_mean_top1']:.1%}", border=True,
              help="Harmonic mean of seen and unseen Top-1: high only when both are high.")
    st.metric("HM Top-5", f"{metrics['harmonic_mean_top5']:.1%}", border=True,
              help="Harmonic mean of seen and unseen Top-5 accuracy.")

seen_acc, unseen_acc = metrics["test_seen"]["top1"], metrics["test_unseen"]["top1"]
with st.container(horizontal=True):
    st.metric("Seen : unseen accuracy", f"{seen_acc / unseen_acc:.2f} : 1" if unseen_acc else "no unseen hits", border=True,
              help="Seen Top-1 divided by unseen Top-1. 1 : 1 is balanced; above 1 the model favours seen pairs, below 1 unseen pairs.")
    st.metric("Seen images → unseen pairs", f"{metrics['test_seen']['predicted_as_unseen']:.1%}", border=True,
              help="Share of seen-test images whose top prediction is one of the 10 unseen pairs (always a mistake). "
                   "Calibration γ raises this.")
    st.metric("Unseen images → unseen pairs", f"{metrics['test_unseen']['predicted_as_unseen']:.1%}", border=True,
              help="Share of unseen-test images whose top prediction is an unseen pair (right group, not necessarily the right pair).")

with st.expander("Compare all models", icon=":material/leaderboard:"):
    table = pd.DataFrame(
        [
            {
                "model": r["version"],
                "method": "trained prompts" if r["method"] == "clip_prompt_learning" else "zero-shot",
                "backbone": "BioCLIP" if "bioclip" in r["cfg"]["model"]["arch"] else "OpenAI CLIP",
                "seen_top1": r["metrics"]["test_seen"]["top1"],
                "unseen_top1": r["metrics"]["test_unseen"]["top1"],
                "hm_top1": r["metrics"]["harmonic_mean_top1"],
                "hm_top5": r["metrics"]["harmonic_mean_top5"],
                "seen_unseen_ratio": (r["metrics"]["test_seen"]["top1"] / r["metrics"]["test_unseen"]["top1"]
                                      if r["metrics"]["test_unseen"]["top1"] else None),
                "seen_to_unseen": r["metrics"]["test_seen"]["predicted_as_unseen"],
                "unseen_to_unseen": r["metrics"]["test_unseen"]["predicted_as_unseen"],
                "gamma": r["metrics"].get("gamma", 0.0),
            }
            for r in runs
        ]
    )
    pct = lambda label: st.column_config.ProgressColumn(label, format="percent", min_value=0.0, max_value=1.0)
    st.dataframe(
        table,
        hide_index=True,
        alt="Test metrics of every finished model run",
        column_config={
            "model": "Model",
            "method": "Method",
            "backbone": "Backbone",
            "seen_top1": pct("Seen Top-1"),
            "unseen_top1": pct("Unseen Top-1"),
            "hm_top1": pct("HM Top-1"),
            "hm_top5": pct("HM Top-5"),
            "seen_unseen_ratio": st.column_config.NumberColumn("Seen : unseen", format="%.2f", help="Seen Top-1 / unseen Top-1; 1.00 is balanced"),
            "seen_to_unseen": pct("Seen → unseen"),
            "unseen_to_unseen": pct("Unseen → unseen"),
            "gamma": st.column_config.NumberColumn("γ", format="%.4f", help="Calibration subtracted from seen-pair scores"),
        },
    )
    st.caption("All rows use the same protocol: every test image ranked against all 333 pairs (dpd/evaluate.py).")

image, truth = None, None
if source == "Upload":
    up = st.file_uploader("Plant image", type=["jpg", "jpeg", "png"], key="infer_upload")
    if up:
        image = Image.open(io.BytesIO(up.getvalue())).convert("RGB")
else:
    test = df[df["split"].isin(["test_seen", "test_unseen"])]
    with st.container(horizontal=True, vertical_alignment="bottom"):
        split = st.segmented_control(
            "Test split", ["test_seen", "test_unseen"], default="test_unseen", required=True,
            format_func=SPLIT_LABELS.get, key="infer_split",
        )
        pool = test[test["split"] == split]
        pair_filter = st.selectbox("Pair", ["Any pair"] + sorted(pool["pair"].unique()), key="infer_pair")
        if pair_filter != "Any pair":
            pool = pool[pool["pair"] == pair_filter]
        if st.button("Random image", icon=":material/shuffle:") or st.session_state.get("infer_file") not in set(pool["file"]):
            st.session_state.infer_file = pool["file"].sample(1).iloc[0]
    row = pool[pool["file"] == st.session_state.infer_file].iloc[0]
    image, truth = load_image(row["file"]), row

if image is None:
    st.caption("Upload a leaf photo to identify it.")
    st.stop()

m = get_model(str(run["dir"]))
pairs = m["pairs"]
unseen_mask = pairs["is_unseen"].to_numpy()
emb = embed_image(m["clip"], m["preprocess"], image)
order, scores, probs = rank_pairs(emb, m["pair_emb"], unseen_mask, m["gamma"])
top = pairs.iloc[order[0]]

left, right = st.columns([2, 3])
with left:
    st.image(image, width="stretch", alt="Plant image being identified")
    if truth is not None:
        st.caption(f"True label: **{truth['pair']}** · {SPLIT_LABELS[truth['split']]} · source {truth['source']}")
with right:
    correct = truth is not None and top["pair_id"] == truth["pair_id"]
    with st.container(horizontal=True):
        st.metric("Prediction", top["pair"].replace("_", " "), border=True)
        st.metric("Confidence", f"{probs[order[0]]:.0%}", border=True)
    if top["is_unseen"]:
        st.info(
            "This pair never appeared in training. It was recognised by combining what the model learned "
            f"about **{top['crop']}** and about **{top['disease'].replace('_', ' ')}** from other pairs.",
            icon=":material/join_inner:",
        )
    if truth is not None:
        (st.success if correct else st.error)("Correct" if correct else f"Wrong: true label is {truth['pair']}")

    top5 = pairs.iloc[order[:5]].assign(
        confidence=probs[order[:5]],
        score=scores[order[:5]],
        status=np.where(unseen_mask[order[:5]], "unseen", "seen"),
    )[["pair", "crop", "disease", "status", "confidence", "score"]]
    st.dataframe(
        top5,
        hide_index=True,
        alt="Top five predicted crop-disease pairs",
        column_config={
            "confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0.0, max_value=1.0),
            "score": st.column_config.NumberColumn("Score", format="%.3f", help="Cosine similarity after calibration"),
        },
    )
    st.caption(f"{run['version']}: calibration γ = {m['gamma']:g} subtracted from seen-pair scores.")

st.subheader("Similar training images")
train, bank = get_train_bank(str(run["dir"]))
idx, sims = nearest(emb, bank, 8)
image_grid(train.iloc[idx], n_cols=8, caption_fields=("pair",))

st.subheader(f"What the model knows about {top['pair']}")
same_pair = train[train["pair_id"] == top["pair_id"]]
if len(same_pair):
    st.markdown(f"**Training images of {top['pair']}** ({len(same_pair)})")
    image_grid(same_pair.head(8), n_cols=8, caption_fields=("source",))
else:
    st.caption(f"No training images of {top['pair']}: it is an unseen pair. These are the pieces it was built from.")
c1, c2 = st.columns(2)
with c1:
    same_crop = train[(train["crop_id"] == top["crop_id"]) & (train["pair_id"] != top["pair_id"])]
    st.markdown(f"**Same crop ({top['crop']}), other diseases** ({len(same_crop)})")
    image_grid(same_crop.sample(min(4, len(same_crop)), random_state=0), n_cols=4, caption_fields=("disease",))
with c2:
    same_dis = train[(train["disease_id"] == top["disease_id"]) & (train["pair_id"] != top["pair_id"])]
    st.markdown(f"**Same disease ({top['disease'].replace('_', ' ')}), other crops** ({len(same_dis)})")
    image_grid(same_dis.sample(min(4, len(same_dis)), random_state=0), n_cols=4, caption_fields=("crop",))
