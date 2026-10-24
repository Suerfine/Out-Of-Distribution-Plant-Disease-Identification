import streamlit as st

from dpd.data import ROOT

st.set_page_config(page_title="Plant disease ID", page_icon=":material/eco:", layout="wide")

page = st.navigation(
    {
        "Dataset": [
            st.Page("app_pages/overview.py", title="Overview", icon=":material/monitoring:"),
            st.Page("app_pages/gallery.py", title="Gallery", icon=":material/photo_library:"),
            st.Page("app_pages/unseen.py", title="Unseen pairs", icon=":material/join_inner:"),
        ],
        "Model": [
            st.Page("app_pages/inference.py", title="Identify disease", icon=":material/search:"),
        ],
    },
    position="top",
)

if not ROOT.exists():
    st.error(
        f"Expected the dataset at `{ROOT}`. Put `DPD_Small/` in the project root or set the `DPD_ROOT` environment variable.",
        title="Dataset not found",
    )
    st.stop()

st.title(page.title, icon=page.icon)
page.run()
