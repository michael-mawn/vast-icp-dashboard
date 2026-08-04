"""
V2 — Migration matrix. PRD §6: "Do accounts move from experimental to
production usage?"

Redesigned after a cold-read failure on the original single-heatmap
layout: a matrix with no plain-language entry point asks a first-time,
un-narrated viewer to do the row/column arithmetic themselves before they
can answer the tab's own question. Now three levels, headline first:

1. Lead bar chart — "How many reached Production?" One bar per starting
   (day-30) archetype, % of that group classified A4 by day 180. This is
   the answer to the tab's question, stated as directly as a single
   number per group can.
2. Horizontal 100% stacked bars — "Where did each group end up?" Full
   day-180 breakdown per starting group, fixed segment order and color so
   every bar reads the same way.
3. Full matrix, collapsed in an expander — the original heatmap, kept for
   anyone who wants the raw cross-tab, with the same axis fixed so the
   diagonal is an actual diagonal (Plotly renders a categorical y-axis
   bottom-to-top by default, which put A1 at the bottom and A0 at the top
   — backwards from how the x-axis reads left-to-right).

Required copy (build plan "Committed requirements", not optional):
- Tautology disclosure: the migration RATE is a generator input, not
  something measured. Unchanged position — before any chart.
- Launch-method dependency: A3/A4 depend partly on launch_method, which
  PRD §7 flags as unverified server-side. Unchanged position.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.archetypes import LAUNCH_METHOD_DEPENDENT_ARCHETYPES  # noqa: F401 (referenced in disclosure copy below)
from src.migrate_analysis import compute_migration_matrix
from src.views.colors import ARCHETYPE_COLORS, ARCHETYPE_LABELS

DAY30_GROUPS = ["A1_TINKERER", "A2_RESEARCHER", "A3_BATCH", "A4_PRODUCTION", "A0_UNCLASSIFIED"]
LEAD_CHART_GROUPS = ["A1_TINKERER", "A2_RESEARCHER", "A3_BATCH", "A0_UNCLASSIFIED"]  # excludes A4 — trivially ~100%
DESTINATION_ORDER = ["A1_TINKERER", "A2_RESEARCHER", "A3_BATCH", "A4_PRODUCTION", "A0_UNCLASSIFIED", "CHURNED"]
DESTINATION_LABELS = {**ARCHETYPE_LABELS, "A0_UNCLASSIFIED": "Dormant (A0)"}  # per this chart's own spec


def render(db_path: str) -> None:
    st.subheader("V2 — Do accounts migrate from experimental usage into production usage, at what rate, and in what direction?")
    st.caption(
        "Day-30 archetype vs day-180 archetype, each using the classification that held for 2+ "
        "consecutive weekly recomputations — single-week flips don't count as migration."
    )
    st.warning(
        "⚠️ **The migration rate here is a generator input, not a measurement.** This dataset's "
        "forward/backward/churn probabilities are config values (config/generator.py) chosen to "
        "produce a plausible demo, not observed behavior. These charts demonstrate the "
        "**mechanism** — cohort eligibility, the two-week hold rule, churn as a terminal state — "
        "that would measure real migration if pointed at real data. They are not themselves a "
        "finding about how often accounts actually graduate."
    )
    st.caption(
        "⚠️ A3 and A4 both depend partly on `launch_method` (console vs programmatic), which PRD "
        "§7 flags as **unconfirmed whether it's distinguishable server-side** on Vast's real API — "
        "the console, CLI, and SDK all call the same REST endpoints. Treat any A3/A4 figure below "
        "as resting on that unverified field."
    )

    # Falls back to the config default when no V4 regeneration has set a
    # churn window this session — see v4_assumptions.py for why this can't
    # just read accounts.churn_date (that's evaluated as of today, not
    # day-180-relative, a different question).
    churn_days = st.session_state.get("last_churn_days")
    result = compute_migration_matrix(db_path, churn_days=churn_days)
    counts = result["counts"]
    n_eligible, n_total = result["n_eligible"], result["n_total"]
    pct_eligible = 100 * n_eligible / n_total if n_total else 0
    st.caption(
        f"**{n_eligible:,} of {n_total:,} accounts ({pct_eligible:.0f}%)** have a full 180-day "
        f"observation window and are eligible for these charts. The excluded {result['n_ineligible']:,} "
        "are disproportionately the platform's most recent (and, given accelerating signups, "
        "largest) cohort — they simply haven't been around long enough yet."
    )

    row_sums = counts.sum(axis=1)
    row_pct = (100 * counts.div(row_sums.replace(0, 1), axis=0)).round(1)

    # --- Level 1: lead chart -------------------------------------------------
    st.markdown("#### How many reached Production?")
    st.caption("% of each starting group classified A4 Production by day 180.")

    lead_x = [ARCHETYPE_LABELS[k] for k in LEAD_CHART_GROUPS]
    lead_y = [row_pct.loc[k, "A4_PRODUCTION"] if row_sums[k] > 0 else 0 for k in LEAD_CHART_GROUPS]
    lead_colors = [ARCHETYPE_COLORS[k] for k in LEAD_CHART_GROUPS]
    lead_fig = go.Figure(go.Bar(
        x=lead_x, y=lead_y, marker_color=lead_colors,
        text=[f"{v:.0f}%" for v in lead_y], textposition="outside",
    ))
    lead_fig.update_layout(
        yaxis_title="% reached Production by day 180",
        xaxis_title="Starting archetype (day 30)",
        yaxis_range=[0, max(lead_y + [10]) * 1.25],
        margin=dict(t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(lead_fig, use_container_width=True)

    # --- Level 2: 100% stacked bars -------------------------------------------
    st.markdown("#### Where did each group end up?")
    st.caption("Full day-180 breakdown per starting group. Same segment order and color on every bar.")

    # Same fix as the matrix below: Plotly renders a categorical y-axis
    # bottom-to-top by default, which would put A1 (first in DAY30_GROUPS)
    # at the bottom of the chart — backwards from a top-to-bottom reading
    # order. Reverse so A1 renders at the top.
    stack_row_order = list(reversed(DAY30_GROUPS))
    stack_fig = go.Figure()
    y_labels = [f"{ARCHETYPE_LABELS[k]} — {int(row_sums[k]):,} accounts" for k in stack_row_order]
    for dest in DESTINATION_ORDER:
        values = [row_pct.loc[k, dest] if dest in row_pct.columns and row_sums[k] > 0 else 0 for k in stack_row_order]
        text = [f"{v:.0f}%" if v > 5 else "" for v in values]
        stack_fig.add_bar(
            name=DESTINATION_LABELS[dest], x=values, y=y_labels, orientation="h",
            marker_color=ARCHETYPE_COLORS[dest], text=text, textposition="inside",
        )
    stack_fig.update_layout(
        barmode="stack",
        xaxis_title="% of starting group",
        yaxis_title=None,
        xaxis_range=[0, 100],
        legend_title=None,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(stack_fig, use_container_width=True)

    # --- Level 3: full matrix, collapsed --------------------------------------
    with st.expander("Full matrix (counts)"):
        view_mode = st.radio("Show", ["Counts", "Row %"], horizontal=True, label_visibility="collapsed")
        if view_mode == "Row %":
            display_df = row_pct
            text_fmt = ".1f"
            colorbar_title = "% of row"
        else:
            display_df = counts
            text_fmt = "d"
            colorbar_title = "accounts"

        # Plotly renders a categorical y-axis bottom-to-top by default, which
        # would put A1 (first in DAY30_GROUPS) at the bottom — backwards from
        # the x-axis reading left-to-right. Reverse the row order fed in so
        # the visual top-to-bottom order matches the x-axis, making same-
        # archetype cells line up on an actual diagonal.
        row_order = list(reversed(DAY30_GROUPS))
        labels_y = [ARCHETYPE_LABELS[k] for k in row_order]
        labels_x = [ARCHETYPE_LABELS[k] if k != "CHURNED" else "Churned" for k in DESTINATION_ORDER]
        z = display_df.reindex(index=row_order, columns=DESTINATION_ORDER).values

        matrix_fig = go.Figure(data=go.Heatmap(
            z=z, x=labels_x, y=labels_y,
            colorscale="Blues", colorbar_title=colorbar_title,
            text=z, texttemplate="%{text:" + text_fmt + "}",
        ))
        matrix_fig.update_layout(
            xaxis_title="Day 180 status",
            yaxis_title="Day 30 archetype",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(matrix_fig, use_container_width=True)
