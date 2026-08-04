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
A0_MIN_RENTALS = 3
A0_MIN_GPU_HOURS = 10.0

# A3 fan-out cap: A3 is defined by 5-50 concurrent instances (PRD §4), which
# is the actual row-count driver at 3,000 accounts, not account count itself.
# Set from the S2 benchmark: A1 (concurrency=1) averaged ~33 rentals/account/
# year at ~150 bytes/row. 300/account (~9x A1) keeps total rows across a
# plausible archetype mix well inside a ~1.3M-row / ~200MB budget, leaving
# headroom in the 1GB Streamlit Cloud ceiling for app + pandas + plotly
# overhead across concurrent viewers.
A3_MAX_RENTALS_PER_ACCOUNT = 300
