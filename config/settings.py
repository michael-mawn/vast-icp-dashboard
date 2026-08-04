"""Global run parameters. Nothing archetype-specific lives here — see archetypes.py and generator.py."""

RANDOM_SEED = 42

# Scale (PRD §8)
N_ACCOUNTS = 3000
SIMULATION_MONTHS = 12

# Classification window (PRD §5): trailing behavior window used to compute
# a weekly_profile row, recomputed weekly.
CLASSIFICATION_WINDOW_DAYS = 30

# Migration (PRD §5, resolved in build plan): matrix compares archetype at
# day 30 vs day 180. Each endpoint uses the archetype that has HELD for this
# many consecutive weekly recomputations, not the raw week's label.
MIGRATION_SNAPSHOT_DAYS = (30, 180)
MIGRATION_HOLD_WEEKS = 2

# Churn: an account is churned after this many consecutive days with no
# rental activity. churn_date is set to the last rental's end_ts.
CHURN_DAYS_NO_ACTIVITY = 60

# A0 floor: minimum activity for an account to be scored at all. Below this,
# ratio features (e.g. verified-host share) have a denominator too small to
# mean anything, and the account is reported as A0 rather than misclassified
# with false confidence.
#
# Both numbers below were approved at 3 rentals / 10.0 GPU-hours during spec
# review, before concrete generation numbers existed. Measured at S4:
#
# - A0_MIN_GPU_HOURS: a typical A1 (Tinkerer) account's 30-day GPU-hours has
#   median 9.58, 25th percentile 5.64 — the floor sat almost exactly at the
#   median, so ~half of genuinely-active Tinkerers failed it, contradicting
#   the floor's stated intent ("without excluding genuine light tinkerers").
#   Lowered to 4.0 (below the 25th percentile).
# - A0_MIN_RENTALS: A2 (Researcher) runs rare, long sessions (6-72h) by
#   design — 833 of ~6,900 A2 account-weeks have exactly 1 rental, 1,011 have
#   exactly 2, often already well past the GPU-hours floor on session length
#   alone. Requiring 3 rentals penalizes A2 specifically for behaving as
#   PRD §4 describes it. Lowered to 2 — still enough that ratio features
#   (interruptible_share etc.) aren't a literal single coin flip, while a
#   real 2-session Researcher month clears it. The remaining 1-rental weeks
#   stay A0: a single rental genuinely can't support any ratio feature
#   (trivially 0% or 100%), so that residual A0 rate is accepted as correct
#   behavior, not a bug — PRD §4 requires A0 be "reported explicitly, never
#   silently dropped."
#
# Both provisional — user was unavailable to confirm at S4; logged in the
# build plan for review.
A0_MIN_RENTALS = 2
A0_MIN_GPU_HOURS = 4.0

# A3 fan-out cap: A3 is defined by 5-50 concurrent instances (PRD §4), which
# is the actual row-count driver at 3,000 accounts, not account count itself.
# Set from the S2 benchmark: A1 (concurrency=1) averaged ~33 rentals/account/
# year at ~150 bytes/row. 300/account (~9x A1) keeps total rows across a
# plausible archetype mix well inside a ~1.3M-row / ~200MB budget, leaving
# headroom in the 1GB Streamlit Cloud ceiling for app + pandas + plotly
# overhead across concurrent viewers.
A3_MAX_RENTALS_PER_ACCOUNT = 300
