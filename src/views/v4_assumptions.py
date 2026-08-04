"""
V4 — Assumptions. PRD §6: "What would have to be true for this to look
different?" Scope cut from the full PRD spec (build plan): three sliders
only — archetype mix at signup, migration dynamics, and the churn window —
not every generator parameter. Full regeneration with a fixed seed on each
change (build plan S8 decision), benchmarked at S2/S3 to confirm it stays
within PRD §11's "a few seconds" bar.
"""

import streamlit as st

from config.generator import (
    ARCHETYPE_MIX_AT_SIGNUP, MIGRATION_FORWARD_MONTHLY, MIGRATION_BACKWARD_MONTHLY,
    CHURN_PROBABILITY_MONTHLY,
)
from config.settings import CHURN_DAYS_NO_ACTIVITY
from src.pipeline import rebuild

ARCHETYPE_LABELS = {
    "A1_TINKERER": "A1 Tinkerer",
    "A2_RESEARCHER": "A2 Researcher",
    "A3_BATCH": "A3 Batch",
    "A4_PRODUCTION": "A4 Production",
}


def render(db_path: str) -> None:
    st.subheader("V4 — What would have to be true for the base-composition and migration answers to look different?")
    st.caption(
        "Every value below is a config assumption this synthetic dataset was built from, not "
        "something measured. Move a slider and click Regenerate to rebuild the entire dataset "
        "(same fixed seed) under the new assumption, then revisit the Base composition and "
        "Migration tabs to see the effect."
    )

    st.markdown("**Archetype mix at signup** — what share of new accounts start as each workload type. "
                 "*What would settle this: actual signup-time behavior from real accounts, not survey or "
                 "firmographic data (§4 — archetype is inferred from platform activity, not who the customer is).*")
    mix_cols = st.columns(4)
    raw_mix = {}
    for col, (key, label) in zip(mix_cols, ARCHETYPE_LABELS.items()):
        raw_mix[key] = col.slider(label, 0, 100, int(round(ARCHETYPE_MIX_AT_SIGNUP[key] * 100)), key=f"mix_{key}")
    mix_total = sum(raw_mix.values()) or 1
    normalized_mix = {k: v / mix_total for k, v in raw_mix.items()}
    st.caption(f"Normalized to 100%: " + ", ".join(f"{ARCHETYPE_LABELS[k]} {v*100:.0f}%" for k, v in normalized_mix.items()))

    st.markdown("**Migration dynamics** — monthly probability an account's true underlying behavior drifts "
                 "toward a more production-like archetype (forward), back toward a more experimental one "
                 "(backward), or goes permanently dormant (churn). *What would settle this: repeat-account "
                 "behavior over multiple quarters on real data — this cannot be inferred from a single "
                 "snapshot.*")
    mig_cols = st.columns(3)
    forward_pct = mig_cols[0].slider("Forward %/month", 0.0, 20.0, MIGRATION_FORWARD_MONTHLY * 100, step=0.5)
    backward_pct = mig_cols[1].slider("Backward %/month", 0.0, 20.0, MIGRATION_BACKWARD_MONTHLY * 100, step=0.5)
    churn_pct = mig_cols[2].slider("Churn %/month", 0.0, 20.0, CHURN_PROBABILITY_MONTHLY * 100, step=0.5)

    st.markdown("**Churn window** — consecutive days with no rental before an account is considered churned. "
                 "*What would settle this: Vast's own reactivation-rate data — the right window is however "
                 "long real dormant-then-returning accounts actually take, which this dataset cannot tell you.*")
    churn_days = st.slider("Churn window (days)", 14, 120, CHURN_DAYS_NO_ACTIVITY, step=7)

    if st.button("🔄 Regenerate dataset with these assumptions", type="primary"):
        with st.spinner("Rebuilding dataset (generate → features → classify)..."):
            report = rebuild(
                db_path,
                archetype_mix=normalized_mix,
                forward_p=forward_pct / 100,
                backward_p=backward_pct / 100,
                churn_p=churn_pct / 100,
                churn_days=churn_days,
            )
        # V2's matrix computes its own churn column at read time (day-180-
        # relative, not the persisted accounts.churn_date) — it has no way
        # to know what churn_days this rebuild used unless we hand it off
        # via session_state. Missing this made the churn-window slider
        # cosmetically move but have zero effect on V2's output; caught by
        # re-checking V2 after a slider-only regeneration during S8 testing.
        st.session_state["last_churn_days"] = churn_days
        st.session_state["last_regen_message"] = (
            f"Regenerated in {report['total_seconds']}s — "
            f"{report['generation_report']['n_rentals']:,} rentals across "
            f"{report['generation_report']['n_accounts']:,} accounts."
        )
        # tab_v2 is rendered BEFORE tab_v4 in streamlit_app.py's script order,
        # but Streamlit re-executes the whole script top-to-bottom on every
        # interaction — so on THIS run, V2 already rendered using the OLD
        # session_state value before the line above updated it. st.rerun()
        # forces a fresh top-to-bottom pass so V2 picks up the new value.
        # Caught by re-checking V2's actual numbers after a churn-window-only
        # regeneration during S8 testing — they hadn't moved at all.
        st.rerun()

    if "last_regen_message" in st.session_state:
        st.success(st.session_state["last_regen_message"] + " Revisit Base composition or Migration to see the new data.")

    st.divider()
    st.markdown("**Deferred from this build, listed as next rather than omitted (PRD §6):**")
    st.markdown(
        "- Expansion / net revenue retention by cohort\n"
        "- Early-signature analysis (what an eventual A3/A4 account's first 14 days look like)\n"
        "- Reliability-tax analysis (retention/spend delta following a host-side failure)\n"
        "- **Classifier accuracy vs. ground truth (originally scoped as V3).** Cut for a one-day, "
        "single-reader build — see the build plan for the full reasoning. In short: the generator "
        "creates realistic overlap between archetypes so classification is a real exercise, but "
        "this dataset never shows how well the classifier actually recovers the labels it was "
        "built from. On real data, that check would need to be rebuilt before trusting the "
        "classifier's output."
    )
