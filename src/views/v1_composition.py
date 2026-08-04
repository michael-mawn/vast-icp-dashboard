"""
V1 — Base composition. PRD §6: "Who are we serving, by workload?"
Accounts by archetype alongside lifetime rental spend by archetype.

Chart type (grouped bar, % of accounts vs % of lifetime spend per
archetype) is a provisional call — user was unavailable to confirm at S5
per CLAUDE.md's "chart type for any new view" rule. Reasoning: PRD's own
framing ("the gap between the two distributions is the thing to look at")
calls for two parallel distributions over the same categories, and a
grouped bar makes that gap directly visible without implying which side
is "better" (a pie-chart pair would invite exactly that reading, and is
also harder to compare precisely). Logged in the build plan for review.
"""

import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARCHETYPE_ORDER = ["A1_TINKERER", "A2_RESEARCHER", "A3_BATCH", "A4_PRODUCTION", "A0_UNCLASSIFIED"]
ARCHETYPE_LABELS = {
    "A1_TINKERER": "A1 Tinkerer",
    "A2_RESEARCHER": "A2 Researcher",
    "A3_BATCH": "A3 Batch",
    "A4_PRODUCTION": "A4 Production",
    "A0_UNCLASSIFIED": "A0 Unclassified",
}


def _current_classification(conn: sqlite3.Connection) -> pd.DataFrame:
    """Most recent classified_archetype per account; accounts with no
    weekly_profile row at all (never enough activity to compute one) are
    A0 by the same 'insufficient activity, reported explicitly' logic."""
    accounts = pd.read_sql("SELECT account_id FROM accounts", conn)
    wp = pd.read_sql("SELECT account_id, week_start, classified_archetype FROM weekly_profile", conn)
    if wp.empty:
        current = pd.DataFrame(columns=["account_id", "classified_archetype"])
    else:
        current = wp.sort_values("week_start").groupby("account_id").last().reset_index()[
            ["account_id", "classified_archetype"]
        ]
    merged = accounts.merge(current, on="account_id", how="left")
    merged["classified_archetype"] = merged["classified_archetype"].fillna("A0_UNCLASSIFIED")
    return merged


def render(db_path: str) -> None:
    st.subheader("V1 — Who are we actually serving, in terms of workload rather than firmographics?")
    st.caption(
        "🔬 Synthetic data. Accounts grouped by their most recent classified archetype "
        "(not ground truth — the classifier never sees it). Compare the two bars per "
        "category; this chart does not say which gap is good or bad."
    )

    conn = sqlite3.connect(db_path)
    current = _current_classification(conn)
    rentals = pd.read_sql("SELECT account_id, total_cost FROM rentals", conn)
    conn.close()

    spend = rentals.groupby("account_id")["total_cost"].sum().reset_index()
    merged = current.merge(spend, on="account_id", how="left")
    merged["total_cost"] = merged["total_cost"].fillna(0.0)

    by_archetype = merged.groupby("classified_archetype").agg(
        n_accounts=("account_id", "count"),
        lifetime_spend=("total_cost", "sum"),
    ).reindex(ARCHETYPE_ORDER).fillna(0)

    pct_accounts = 100 * by_archetype["n_accounts"] / by_archetype["n_accounts"].sum()
    pct_spend = 100 * by_archetype["lifetime_spend"] / by_archetype["lifetime_spend"].sum()

    labels = [ARCHETYPE_LABELS[k] for k in ARCHETYPE_ORDER]
    fig = go.Figure()
    fig.add_bar(name="% of accounts", x=labels, y=pct_accounts.values, marker_color="#7cb9f2")
    fig.add_bar(name="% of lifetime rental spend", x=labels, y=pct_spend.values, marker_color="#2b5fad")
    fig.update_layout(
        barmode="group", yaxis_title="% of total", legend_title=None,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    n_total = int(by_archetype["n_accounts"].sum())
    n_a0 = int(by_archetype.loc["A0_UNCLASSIFIED", "n_accounts"])
    st.caption(
        f"{n_total:,} accounts total, {n_a0:,} ({100*n_a0/n_total:.0f}%) unclassified "
        "(insufficient trailing-30-day activity — see A0 in Assumptions for the exact floor)."
    )
