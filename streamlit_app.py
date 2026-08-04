import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from src.generate import run_a1_only
from src.schema import DB_PATH, migrate

st.set_page_config(page_title="Vast.ai Workload Archetype & Migration Dashboard", layout="wide")


@st.cache_resource
def ensure_data() -> None:
    """Streamlit Cloud clones the repo fresh each deploy; data/*.db is
    gitignored, so the app must generate it once on first load rather than
    assuming a local run already produced it."""
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        run_a1_only()


ensure_data()

st.title("Vast.ai Workload Archetype & Migration Dashboard")
st.caption(
    "🔬 Runs entirely on synthetic data — not a findings document about Vast's real business."
)

st.subheader("S2 diagnostic — does the generator actually produce Tinkerer-shaped behavior?")
st.caption(
    "Temporary view. Confirms the generation mechanism end to end before S3 adds the other "
    "three archetypes and the real classifier. Replaced by the V1-V4 views in later slices."
)

if not os.path.exists(DB_PATH):
    st.warning("No generated data yet. Run `python -m src.generate` locally, or wait for the next deploy.")
else:
    migrate(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    rentals = pd.read_sql("SELECT gpu_hours FROM rentals", conn)
    conn.close()

    if rentals.empty:
        st.warning("Database exists but has no rentals yet.")
    else:
        median_hours = rentals["gpu_hours"].median()
        fig = px.histogram(
            rentals, x="gpu_hours", nbins=60,
            title="A1 (Tinkerer) session duration — PRD §4 defines A1 as median session < 4h",
            labels={"gpu_hours": "Session duration (hours)"},
        )
        fig.add_vline(x=median_hours, line_dash="dash", annotation_text=f"median = {median_hours:.2f}h")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Rentals generated", f"{len(rentals):,}")
