import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from src.generate import generate_dataset
from src.schema import DB_PATH, migrate

st.set_page_config(page_title="Vast.ai Workload Archetype & Migration Dashboard", layout="wide")


@st.cache_resource
def ensure_data() -> None:
    """Streamlit Cloud clones the repo fresh each deploy; data/*.db is
    gitignored, so the app must generate it once on first load rather than
    assuming a local run already produced it."""
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        generate_dataset()


ensure_data()

st.title("Vast.ai Workload Archetype & Migration Dashboard")
st.caption(
    "🔬 Runs entirely on synthetic data — not a findings document about Vast's real business."
)

st.subheader("S3 diagnostic — do all four archetypes generate distinguishably?")
st.caption(
    "Temporary view, checking the generator before S4's classifier and the real V1-V4 views "
    "replace it. Session duration in each account's first 30 days, by ground-truth archetype "
    "(a later week can look different — that's what migration means)."
)

if not os.path.exists(DB_PATH):
    st.warning("No generated data yet. Run `python -m src.generate` locally, or wait for the next deploy.")
else:
    migrate(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    rentals = pd.read_sql("SELECT account_id, start_ts, gpu_hours FROM rentals", conn)
    accounts = pd.read_sql("SELECT account_id, signup_date FROM accounts", conn)
    ground_truth = pd.read_sql("SELECT account_id, archetype FROM ground_truth", conn)
    conn.close()

    if rentals.empty:
        st.warning("Database exists but has no rentals yet.")
    else:
        merged = rentals.merge(ground_truth, on="account_id").merge(accounts, on="account_id")
        merged["start_ts"] = pd.to_datetime(merged["start_ts"], format="ISO8601")
        merged["signup_date"] = pd.to_datetime(merged["signup_date"], format="ISO8601")
        early = merged[(merged["start_ts"] - merged["signup_date"]).dt.days <= 30]

        label = {
            "A1_TINKERER": "A1 Tinkerer (median < 4h)",
            "A2_RESEARCHER": "A2 Researcher (6-72h)",
            "A3_BATCH": "A3 Batch (short, high concurrency)",
            "A4_PRODUCTION": "A4 Production (> 168h)",
        }
        early = early.assign(archetype_label=early["archetype"].map(label))

        st.caption("Independent bins per archetype below — durations span 100x+ (A1/A3 in hours, A4 in days).")
        colors = {"A1_TINKERER": "#7cb9f2", "A2_RESEARCHER": "#2b5fad", "A3_BATCH": "#e33d3d", "A4_PRODUCTION": "#f2a6a6"}
        grid = st.columns(2)
        for i, (key, name) in enumerate(label.items()):
            subset = early[early["archetype"] == key]
            fig = px.histogram(
                subset, x="gpu_hours", nbins=40, title=name,
                labels={"gpu_hours": "Session duration (hours)"},
                color_discrete_sequence=[colors[key]],
            )
            fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))
            grid[i % 2].plotly_chart(fig, use_container_width=True)

        cols = st.columns(4)
        for col, (key, name) in zip(cols, label.items()):
            n_accounts = (ground_truth["archetype"] == key).sum()
            col.metric(name.split(" (")[0], f"{n_accounts:,} accounts")
        st.caption(f"{len(rentals):,} total rentals generated across {accounts['account_id'].nunique():,} accounts.")
