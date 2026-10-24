import altair as alt
import pandas as pd
import streamlit as st

from dpd.data import SPLITS
from dpd.ui import SPLIT_LABELS, image_sizes, load_metadata

df = load_metadata()
train = df[df["split"] == "train"]

with st.container(horizontal=True):
    st.metric("Images", f"{len(df):,}", border=True)
    st.metric("Crops", df["crop"].nunique(), border=True)
    st.metric("Diseases", df["disease"].nunique(), border=True)
    st.metric("Seen pairs", train["pair_id"].nunique(), border=True)
    st.metric("Unseen pairs", df.loc[df["is_unseen"], "pair_id"].nunique(), border=True)
    st.metric("Source datasets", df["source"].nunique(), border=True)

split_table = (
    df.groupby("split")
    .agg(images=("file", "size"), crops=("crop", "nunique"), diseases=("disease", "nunique"), pairs=("pair", "nunique"))
    .reindex(SPLITS)
    .rename(index=SPLIT_LABELS)
)
st.table(split_table, alt="Image, crop, disease and pair counts per split")

st.subheader("Class balance")
level = st.segmented_control(
    "Group by", ["pair", "crop", "disease", "source"], default="pair", required=True, key="overview_level"
)
counts = train[level].value_counts().rename_axis(level).reset_index(name="train_images")

with st.container(horizontal=True):
    st.metric(f"Train images per {level} (median)", f"{counts['train_images'].median():g}", border=True)
    st.metric("Min", int(counts["train_images"].min()), border=True)
    st.metric("Max", int(counts["train_images"].max()), border=True)
    st.metric(f"{level.capitalize()}s with ≤5 images", int((counts["train_images"] <= 5).sum()), border=True)

ranked = counts.assign(rank=range(1, len(counts) + 1))
chart = (
    alt.Chart(ranked)
    .mark_bar()
    .encode(
        x=alt.X("rank:Q", title=f"{level.capitalize()} rank (most to fewest images)"),
        y=alt.Y("train_images:Q", title="Train images", scale=alt.Scale(type="symlog")),
        tooltip=[level, "train_images"],
    )
    .properties(height=280)
)
st.altair_chart(chart, alt=f"Long-tailed distribution of training images per {level}")

left, right = st.columns(2)
with left:
    st.caption(f"Most common {level}s")
    st.dataframe(counts.head(15), hide_index=True, alt=f"Top 15 {level}s by training images")
with right:
    st.caption(f"Rarest {level}s")
    st.dataframe(counts.tail(15).iloc[::-1], hide_index=True, alt=f"Bottom 15 {level}s by training images")

st.subheader("Diseases shared across crops")
st.caption(
    "Unseen pairs can only be recognised compositionally if their disease also appears on other crops. "
    "Most diseases belong to a single crop."
)
crops_per_disease = (
    df.groupby("disease")["crop"].agg(crops="nunique", crop_list=lambda s: ", ".join(sorted(s.unique())))
    .sort_values("crops", ascending=False)
    .reset_index()
)
c1, c2 = st.columns([1, 2])
with c1:
    dist = crops_per_disease["crops"].value_counts().sort_index().rename_axis("crops per disease").reset_index(name="diseases")
    st.bar_chart(dist, x="crops per disease", y="diseases", alt="Most diseases appear on only one crop")
with c2:
    st.dataframe(
        crops_per_disease[crops_per_disease["crops"] > 1],
        hide_index=True,
        height=300,
        alt="Diseases that appear on more than one crop",
    )

st.subheader("Source datasets")
st.caption(
    "The file-name prefix is the original dataset. Each source has its own photo style, "
    "so a model can learn the source instead of the disease."
)
src = pd.crosstab(df["source"], df["split"]).reindex(columns=SPLITS, fill_value=0).rename(columns=SPLIT_LABELS)
src["crops"] = df.groupby("source")["crop"].nunique()
src = src.sort_values("Train", ascending=False)
st.dataframe(
    src,
    alt="Images per source dataset and split",
    column_config={"Train": st.column_config.ProgressColumn("Train", format="%d", max_value=int(src["Train"].max()))},
)
only_test = src[src["Train"] == 0].index.tolist()
if only_test:
    st.info(f"Sources with no training images: {', '.join(only_test)}. These are a small built-in domain-shift check.")

st.subheader("Image sizes")
if st.toggle("Scan image sizes (reads every file header once, then cached)", key="overview_scan_sizes"):
    sizes = image_sizes(tuple(df["file"])).merge(df[["file", "split", "source"]], on="file")
    with st.container(horizontal=True):
        st.metric("Median width", int(sizes["width"].median()), border=True)
        st.metric("Median height", int(sizes["height"].median()), border=True)
        st.metric("Smallest side (min)", int(sizes[["width", "height"]].min(axis=1).min()), border=True)
        st.metric("Non-RGB images", int((sizes["mode"] != "RGB").sum()), border=True)
    st.altair_chart(
        alt.Chart(sizes.sample(min(len(sizes), 4000), random_state=0))
        .mark_circle(opacity=0.4)
        .encode(
            x=alt.X("width:Q", scale=alt.Scale(type="log")),
            y=alt.Y("height:Q", scale=alt.Scale(type="log")),
            color="split:N",
            tooltip=["file", "width", "height", "source"],
        )
        .properties(height=320),
        alt="Image width versus height, coloured by split",
    )
    st.dataframe(
        sizes.groupby("source")[["width", "height"]].median().astype(int).sort_values("width"),
        alt="Median image size per source dataset",
    )
