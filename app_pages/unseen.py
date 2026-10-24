import streamlit as st

from dpd.ui import image_grid, load_metadata

df = load_metadata()
train = df[df["split"] == "train"]
unseen = df[df["split"] == "test_unseen"]

st.caption(
    "Each unseen pair is a new combination of a crop and a disease that were each seen in training, "
    "but never together. Compare the test images with what the model can learn from: the same crop "
    "with other diseases, and the same disease on other crops."
)

summary = (
    unseen.groupby(["pair", "crop", "disease"]).size().rename("test_images").reset_index()
    .assign(
        crop_train_images=lambda d: d["crop"].map(train["crop"].value_counts()).fillna(0).astype(int),
        disease_train_images=lambda d: d["disease"].map(train["disease"].value_counts()).fillna(0).astype(int),
        disease_seen_on=lambda d: d["disease"].map(
            train.groupby("disease")["crop"].agg(lambda s: ", ".join(sorted(s.unique())))
        ),
        sources=lambda d: d["pair"].map(unseen.groupby("pair")["source"].agg(lambda s: ", ".join(sorted(s.unique())))),
    )
    .sort_values("test_images", ascending=False)
)
st.dataframe(
    summary,
    hide_index=True,
    alt="Unseen pairs with how much training evidence exists for their crop and disease",
    column_config={
        "test_images": st.column_config.NumberColumn("Test images"),
        "crop_train_images": st.column_config.ProgressColumn(
            "Crop train imgs", format="%d", max_value=int(summary["crop_train_images"].max())
        ),
        "disease_train_images": st.column_config.ProgressColumn(
            "Disease train imgs", format="%d", max_value=int(summary["disease_train_images"].max())
        ),
    },
)

pair = st.selectbox(
    "Unseen pair",
    summary["pair"],
    format_func=lambda p: f"{p} ({int(summary.loc[summary['pair'] == p, 'test_images'].iloc[0])} test images)",
    key="unseen_pair",
)
row = summary[summary["pair"] == pair].iloc[0]
n_cols = st.slider("Images per row", 4, 10, 8, key="unseen_cols")
seed = st.number_input("Shuffle seed", 0, 10_000, 0, key="unseen_seed")


def sample(d, n):
    return d.sample(min(len(d), n), random_state=seed)


with st.container(border=True):
    st.markdown(f"**Test images of `{pair}`** (never seen in training)")
    image_grid(sample(unseen[unseen["pair"] == pair], n_cols * 2), n_cols=n_cols, caption_fields=("source",))

same_crop = train[(train["crop"] == row["crop"])]
with st.container(border=True):
    st.markdown(
        f"**Same crop, other diseases**: `{row['crop']}` in training "
        f"({len(same_crop)} images, {same_crop['disease'].nunique()} diseases)"
    )
    image_grid(sample(same_crop, n_cols * 2), n_cols=n_cols, caption_fields=("disease", "source"))

same_disease = train[(train["disease"] == row["disease"])]
with st.container(border=True):
    st.markdown(
        f"**Same disease, other crops**: `{row['disease']}` in training "
        f"({len(same_disease)} images on {same_disease['crop'].nunique()} crops)"
    )
    image_grid(sample(same_disease, n_cols * 2), n_cols=n_cols, caption_fields=("crop", "source"))
