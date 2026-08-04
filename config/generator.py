"""
Synthetic data generation parameters. PRD §8: ground-truth labels are
assigned first, behavior is generated FROM those labels, then the classifier
independently rediscovers them (V3, which reports that accuracy, is CUT —
see build plan; deferred-view note explains why in the UI).

FIREWALL: this module must never be imported by config/archetypes.py or
src/classify.py. Generation logic and classification logic must have no
shared code path — see src/tests for the enforced import-boundary test.

All values below are placeholders pending sign-off at S3 (build plan:
"archetype mix at signup" and "migration probability matrix" are listed in
PRD §8 as tunable with no defaults given — not invented here).
"""

from config.settings import RANDOM_SEED  # noqa: F401 (re-exported for generate.py)

ARCHETYPE_KEYS = ("A1_TINKERER", "A2_RESEARCHER", "A3_BATCH", "A4_PRODUCTION")

# Share of accounts assigned each ground-truth (signup-time) archetype.
# Approved at S3: bottom-heavy funnel matching PRD §2's framing — most
# accounts start experimental, few sign up already doing production
# inference.
ARCHETYPE_MIX_AT_SIGNUP = {
    "A1_TINKERER": 0.55,
    "A2_RESEARCHER": 0.25,
    "A3_BATCH": 0.12,
    "A4_PRODUCTION": 0.08,
}

# Migration dynamics. PRD §8 asks for "migration probability per archetype
# pair" (12 cells) — approved at S3 as 3 aggregate monthly rates instead,
# applied over MIGRATION_GRAPH's directed edges. This governs the account's
# TRUE underlying archetype for a given month (used to pick which behavior
# profile shapes that month's generated rentals). It is NOT the same thing
# as ground_truth_archetype, which stays fixed at the signup value per the
# schema (see src/generate.py docstring for why: V3, the only view that
# would have compared classifier output to a persisted ground truth, is
# cut). V2's migration matrix is built entirely from the classifier's own
# weekly re-classification of this generated behavior — nothing here is
# read directly by src/classify.py.
MIGRATION_FORWARD_MONTHLY = 0.04
MIGRATION_BACKWARD_MONTHLY = 0.01
CHURN_PROBABILITY_MONTHLY = 0.02

# Forward = toward more production-like usage. A1 can branch into either
# established path; both established paths converge on A4. No direct
# A1->A4 edge — that jump happens (if at all) across multiple months.
MIGRATION_GRAPH = {
    "A1_TINKERER": ["A2_RESEARCHER", "A3_BATCH"],
    "A2_RESEARCHER": ["A4_PRODUCTION"],
    "A3_BATCH": ["A4_PRODUCTION"],
    "A4_PRODUCTION": [],
}

# Magnitude of deliberate overlap between archetype feature distributions
# (spec-review decision: modest overlap so the classifier isn't reading back
# stamped values, without full continuous latent profiles). Approved at S3.
# Applies only at the boundaries spec review identified as genuinely
# ambiguous in §4 — not every archetype pair.
FEATURE_OVERLAP = 0.15
ADJACENT_ARCHETYPE = {
    "A1_TINKERER": "A3_BATCH",
    "A3_BATCH": "A1_TINKERER",
    "A2_RESEARCHER": "A4_PRODUCTION",
    "A4_PRODUCTION": "A2_RESEARCHER",
}

# Host-side interruption rate for interruptible rentals, and the odds a
# given interruption gets restarted rather than abandoned (per-archetype,
# see ARCHETYPE_PROFILES in src/generate.py for the latter). Not a PRD
# threshold — this shapes raw end_reason/restart data for src/features.py
# to compute restart_after_interruption_rate from in S4. No view uses a
# "retention penalty" concept (the PRD's reliability-tax view is deferred),
# so it is not modeled.
HOST_FAILURE_RATE = 0.12

# Signup growth rate: PRD §8 says "accelerating" without a magnitude. Shape
# only (monotonically increasing signups across the window) — exact curve
# is a generation detail, not a threshold, so it isn't gated behind sign-off.
SIGNUP_GROWTH_POWER = 2.5  # rng.power(a) skews samples toward 1 (= late window)

# Magnitude discipline (PRD §8): generated values must stay inside the
# envelope implied by Vast's real public numbers.
MAGNITUDE_ENVELOPE = {
    "total_gpus": 17_000,
    "total_providers": 1_400,
    "verified_h100_sxm_price_per_hour": 2.89,
}
