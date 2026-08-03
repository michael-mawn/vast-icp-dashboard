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

# Share of accounts assigned each ground-truth archetype at signup.
# Placeholder — proposed at S3 for approval, not invented silently.
ARCHETYPE_MIX_AT_SIGNUP = None

# Per-(archetype_pair) weekly probability of a ground-truth-level migration
# event. Placeholder — proposed at S3 for approval.
MIGRATION_PROBABILITY = None

# Magnitude of deliberate overlap between archetype feature distributions
# (spec-review decision: modest overlap so the classifier isn't reading back
# stamped values, without full continuous latent profiles). Placeholder —
# tuned at S3 against measured classifier behavior.
FEATURE_OVERLAP = None

# Host-side failure rate and post-failure retention penalty (PRD §8).
# Placeholder — proposed at S3.
HOST_FAILURE_RATE = None
RETENTION_PENALTY_AFTER_FAILURE = None

# Signup growth rate across the 12-month window (PRD §8: "accelerating").
# Placeholder — proposed at S3.
SIGNUP_GROWTH_RATE = None

# Deposit size distribution per archetype (PRD §8).
# Placeholder — proposed at S3.
DEPOSIT_DISTRIBUTION = None

# Magnitude discipline (PRD §8): generated values must stay inside the
# envelope implied by Vast's real public numbers.
MAGNITUDE_ENVELOPE = {
    "total_gpus": 17_000,
    "total_providers": 1_400,
    "verified_h100_sxm_price_per_hour": 2.89,
}
