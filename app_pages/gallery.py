import math

import streamlit as st

from dpd.data import SPLITS, load_image, load_metadata
from dpd.gallery import SPLIT_LABELS, image_grid

df = load_metadata()

with st.sidebar:
    st.header("Filters")
    splits = st.pills(
        "Split", SPLITS, selection_mode="multi", default=SPLITS, format_func=SPLIT_LABELS.get, key="gallery_splits"
    )
    crops = st.multiselect("Crop", sorted(df["crop"].unique()), key="gallery_crops", placeholder="All crops")
    disease_pool = df[df["crop"].isin(crops)] if crops else df
    diseases = st.multiselect(
        "Disease", sorted(disease_pool["disease"].unique()), key="gallery_diseases", placeholder="All diseases"
    )
    sources = st.multiselect("Source dataset", sorted(df["source"].unique()), key="gallery_sources", placeholder="All sources")
    only_unseen = st.toggle("Only the 10 unseen pairs", key="gallery_only_unseen")

    st.header("Display")
    view = st.segmented_control(
        "View", ["Images", "One row per pair"], default="Images", required=True, key="gallery_view"
    )
    n_cols = st.slider("Columns", 3, 10, 6, key="gallery_cols")
    order = st.segmented_control("Order", ["Shuffled", "By pair"], default="Shuffled", required=True, key="gallery_order")
    seed = st.number_input("Shuffle seed", 0, 10_000, 0, key="gallery_seed")

sel = df[df["split"].isin(splits)]
if crops:
    sel = sel[sel["crop"].isin(crops)]
if diseases:
    sel = sel[sel["disease"].isin(diseases)]
if sources:
    sel = sel[sel["source"].isin(sources)]
if only_unseen:
    sel = sel[sel["is_unseen"]]
sel = sel.sample(frac=1, random_state=seed) if order == "Shuffled" else sel.sort_values(["pair", "split", "file"])

with st.container(horizontal=True):
    st.metric("Matching images", f"{len(sel):,}", border=True)
    st.metric("Pairs", sel["pair"].nunique(), border=True)
    st.metric("Crops", sel["crop"].nunique(), border=True)
    st.metric("Diseases", sel["disease"].nunique(), border=True)

if sel.empty:
    st.info("No images match these filters.")
    st.stop()

if view == "Images":
    per_page = n_cols * 4
    n_pages = math.ceil(len(sel) / per_page)
    page_no = st.number_input(f"Page (of {n_pages})", 1, n_pages, 1, key="gallery_page")
    page_df = sel.iloc[(page_no - 1) * per_page : page_no * per_page]
    image_grid(page_df, n_cols=n_cols)

    @st.dialog("Image details", width="large")
    def show_details(file: str) -> None:
        r = df[df["file"] == file].iloc[0]
        img = load_image(file)
        st.image(img, width="stretch", alt=f"{r.crop} leaf, {r.disease.replace('_', ' ')}")
        st.table(
            {
                "File": r.file,
                "Split": SPLIT_LABELS[r.split],
                "Crop": f"{r.crop} (id {r.crop_id})",
                "Disease": f"{r.disease} (id {r.disease_id})",
                "Pair": f"{r.pair} (id {r.pair_id}){' · unseen' if r.is_unseen else ''}",
                "Source": r.source,
                "Size": f"{img.width} × {img.height}",
            },
            alt="Metadata for the selected image",
        )

    with st.container(horizontal=True, vertical_alignment="bottom"):
        picked = st.selectbox("Inspect an image from this page", page_df["file"], key="gallery_inspect")
        if st.button("Open", icon=":material/zoom_in:"):
            show_details(picked)
else:
    per_pair = n_cols
    pairs = sel["pair"].drop_duplicates().sort_values().tolist()
    pairs_per_page = 10
    n_pages = math.ceil(len(pairs) / pairs_per_page)
    page_no = st.number_input(f"Page (of {n_pages}, {pairs_per_page} pairs each)", 1, n_pages, 1, key="gallery_pair_page")
    for pair in pairs[(page_no - 1) * pairs_per_page : page_no * pairs_per_page]:
        group = sel[sel["pair"] == pair]
        counts = group["split"].value_counts()
        badge = " :orange-badge[unseen]" if group["is_unseen"].iloc[0] else ""
        summary = " · ".join(f"{SPLIT_LABELS[s]} {counts.get(s, 0)}" for s in SPLITS if s in splits)
        with st.container(border=True):
            st.markdown(f"**{pair}**{badge}  \n:gray[{summary}]")
            image_grid(group.head(per_pair), n_cols=n_cols, caption_fields=("split", "source"))
