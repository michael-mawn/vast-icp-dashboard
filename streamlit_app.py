import os

import streamlit as st

from src.classify import classify_all
from src.features import write_weekly_profiles
from src.generate import generate_dataset
from src.schema import DB_PATH, migrate
from src.views import v1_composition

st.set_page_config(page_title="Vast.ai Workload Archetype & Migration Dashboard", layout="wide")


@st.cache_resource
def ensure_data() -> None:
    """Streamlit Cloud clones the repo fresh each deploy; data/*.db is
    gitignored, so the app must build it once on first load: generate,
    then compute features, then classify."""
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        generate_dataset()
        write_weekly_profiles()
        classify_all()


ensure_data()
migrate(DB_PATH)  # no-op if already current; cheap safety net on every load

st.title("Vast.ai Workload Archetype & Migration Dashboard")

tab_v1, tab_v2, tab_v4 = st.tabs(["Base composition", "Migration", "Assumptions"])

with tab_v1:
    st.caption("🔬 Runs entirely on synthetic data — not a findings document about Vast's real business.")
    v1_composition.render(DB_PATH)

with tab_v2:
    st.caption("🔬 Runs entirely on synthetic data — not a findings document about Vast's real business.")
    st.info("V2 (migration matrix) lands in S6.")

with tab_v4:
    st.caption("🔬 Runs entirely on synthetic data — not a findings document about Vast's real business.")
    st.info("V4 (assumption sliders) lands in S8.")
