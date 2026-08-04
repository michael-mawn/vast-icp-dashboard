"""
V2 — Migration matrix. PRD §6: "Do accounts move from experimental to
production usage?" Heatmap, churn as a terminal column (PRD's own chart
type choice — no provisional call needed here, unlike V1).

Required copy (build plan "Committed requirements", not optional):
- Tautology disclosure: the migration RATE is a generator input
  (config/generator.py MIGRATION_FORWARD_MONTHLY etc.), not something
  measured from real behavior. This matrix demonstrates the MECHANISM
  (hold-rule + cohort logic), not a finding about real migration rates.
- Launch-method dependency: A3 and A4 rows/columns both lean on
  launch_method, which PRD §7 flags as unverified server-side.
"""

import streamlit as st

from config.archetypes import LAUNCH_METHOD_DEPENDENT_ARCHETYPES
from src.migrate_analysis import compute_migration_matrix

LABELS = {
    "A1_TINKERER": "A1 Tinkerer",
    "A2_RESEARCHER": "A2 Researcher",
    "A3_BATCH": "A3 Batch",
    "A4_PRODUCTION": "A4 Production",
    "A0_UNCLASSIFIED": "A0 Unclassified",
    "CHURNED": "Churned",
}


def render(db_path: str) -> None:
    st.subheader("V2 — Do accounts migrate from experimental usage into production usage, at what rate, and in what direction?")
    st.caption(
        "Day-30 archetype (rows) vs day-180 archetype (columns), each using the classification "
        "that held for 2+ consecutive weekly recomputations — single-week flips don't count as "
        "migration."
    )
    st.warning(
        "⚠️ **The migration rate here is a generator input, not a measurement.** This dataset's "
        "forward/backward/churn probabilities are config values (config/generator.py) chosen to "
        "produce a plausible demo, not observed behavior. This matrix demonstrates the "
        "**mechanism** — cohort eligibility, the two-week hold rule, churn as a terminal state — "
        "that would measure real migration if pointed at real data. It is not itself a finding "
        "about how often accounts actually graduate."
    )
    st.caption(
        f"⚠️ A3 and A4 (rows/columns) both depend partly on `launch_method` (console vs "
        "programmatic), which PRD §7 flags as **unconfirmed whether it's distinguishable "
        "server-side** on Vast's real API — the console, CLI, and SDK all call the same REST "
        "endpoints. Treat any A3/A4 cell as resting on that unverified field."
    )

    result = compute_migration_matrix(db_path)
    n_eligible, n_total = result["n_eligible"], result["n_total"]
    pct_eligible = 100 * n_eligible / n_total if n_total else 0
    st.caption(
        f"**{n_eligible:,} of {n_total:,} accounts ({pct_eligible:.0f}%)** have a full 180-day "
        f"observation window and are eligible for this matrix. The excluded {result['n_ineligible']:,} "
        "are disproportionately the platform's most recent (and, given accelerating signups, "
        "largest) cohort — they simply haven't been around long enough yet."
    )

    view_mode = st.radio("Show", ["Counts", "Row %"], horizontal=True, label_visibility="collapsed")
    counts = result["counts"]
    if view_mode == "Row %":
        row_sums = counts.sum(axis=1).replace(0, 1)
        display_df = (100 * counts.div(row_sums, axis=0)).round(1)
        text_fmt = ".1f"
        colorbar_title = "% of row"
    else:
        display_df = counts
        text_fmt = "d"
        colorbar_title = "accounts"

    import plotly.graph_objects as go
    labels_y = [LABELS[k] for k in counts.index]
    labels_x = [LABELS[k] for k in counts.columns]
    fig = go.Figure(data=go.Heatmap(
        z=display_df.values, x=labels_x, y=labels_y,
        colorscale="Blues", colorbar_title=colorbar_title,
        text=display_df.values, texttemplate="%{text:" + text_fmt + "}",
    ))
    fig.update_layout(
        xaxis_title="Day 180 (held classification, or churned)",
        yaxis_title="Day 30 (held classification)",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
