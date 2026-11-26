import streamlit as st

from dpd.data import ROOT
from dpd.hf_sync import is_hf_configured, sync_all

st.set_page_config(page_title="Plant disease ID", page_icon=":material/eco:", layout="wide")


@st.cache_resource(show_spinner="Downloading dataset and model metadata from Hugging Face…")
def _bootstrap_hf() -> bool:
    return sync_all()


if is_hf_configured():
    _bootstrap_hf()

page = st.navigation(
    {
        "Model": [
            st.Page("app_pages/inference.py", title="Identify disease", icon=":material/search:", default=True),
            st.Page("app_pages/dashboard.py", title="Model dashboard", icon=":material/leaderboard:"),
        ],
        "Dataset": [
            st.Page("app_pages/overview.py", title="Overview", icon=":material/monitoring:"),
            st.Page("app_pages/gallery.py", title="Gallery", icon=":material/photo_library:"),
            st.Page("app_pages/unseen.py", title="Unseen pairs", icon=":material/join_inner:"),
        ],
    },
    position="top",
)

if not ROOT.exists():
    st.error(
        f"Expected the dataset at `{ROOT}`. Put `DPD_Small/` in the project root, set `DPD_ROOT`, "
        "or configure `HF_REPO_ID` in Streamlit secrets / environment variables.",
        title="Dataset not found",
    )
    st.stop()

st.title(page.title, icon=page.icon)
page.run()
