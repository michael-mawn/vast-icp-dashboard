"""
Rule-based scoring classifier (PRD §10: readable on purpose, not ML).

FIREWALL: imports ONLY config.archetypes (thresholds/weights) and reads
ONLY the weekly_profile table (computed by src/features.py from raw
rentals). Never imports config.generator and never touches ground_truth —
this is the enforced half of CLAUDE.md's "classifier must rediscover
labels independently" rule. See tests/test_firewall.py.
"""

import sqlite3

import pandas as pd

from config.archetypes import ARCHETYPES, MIN_FIT_FRACTION, A0_FLOOR, MIN_RENTALS_FOR_RATIO_TRUST, RATIO_FEATURES
from src.schema import DB_PATH

COMPARATORS = {
    "gt": lambda v, t: v is not None and v > t,
    "gte": lambda v, t: v is not None and v >= t,
    "lt": lambda v, t: v is not None and v < t,
    "lte": lambda v, t: v is not None and v <= t,
    "eq": lambda v, t: v == t,
    "in": lambda v, t: v in t,
    "between": lambda v, t: v is not None and t[0] <= v <= t[1],
}

# Two max-score tables per archetype: full (all criteria) and ratio-excluded
# (denominator recomputed so the fit fraction stays a fair 0-1 scale when
# ratio criteria are dropped below MIN_RENTALS_FOR_RATIO_TRUST — see
# config/archetypes.py for why they're excluded rather than evaluated at a
# low, noisy denominator).
_MAX_SCORE_FULL = {key: sum(w for *_, w in spec["criteria"]) for key, spec in ARCHETYPES.items()}
_MAX_SCORE_NO_RATIO = {
    key: sum(w for feature_name, *_, w in spec["criteria"] if feature_name not in RATIO_FEATURES)
    for key, spec in ARCHETYPES.items()
}


def _feature_value(row: pd.Series, feature_name: str):
    if feature_name == "gpu_class":
        return row["dominant_gpu_class"]
    if feature_name == "template_category":
        return row["dominant_template_category"]
    return row.get(feature_name)


def score_row(row: pd.Series) -> dict:
    """Fit fraction (0-1) per archetype for one weekly_profile row. Ratio
    criteria (interruptible_share etc.) are excluded below
    MIN_RENTALS_FOR_RATIO_TRUST rentals rather than evaluated at a
    low-denominator, high-noise value."""
    trust_ratios = row["rental_count"] >= MIN_RENTALS_FOR_RATIO_TRUST
    max_score = _MAX_SCORE_FULL if trust_ratios else _MAX_SCORE_NO_RATIO

    scores = {}
    for key, spec in ARCHETYPES.items():
        total = 0.0
        for feature_name, comparator, threshold, weight in spec["criteria"]:
            if not trust_ratios and feature_name in RATIO_FEATURES:
                continue
            value = _feature_value(row, feature_name)
            if COMPARATORS[comparator](value, threshold):
                total += weight
        denom = max_score[key]
        scores[key] = (total / denom) if denom > 0 else 0.0
    return scores


def classify_row(row: pd.Series) -> str:
    if row["rental_count"] < A0_FLOOR["min_rentals"] or row["gpu_hours_sum"] < A0_FLOOR["min_gpu_hours"]:
        return "A0_UNCLASSIFIED"

    scores = score_row(row)
    best_key = max(scores, key=scores.get)
    best_score = scores[best_key]
    if best_score < MIN_FIT_FRACTION:
        return "A0_UNCLASSIFIED"

    tied = [k for k, v in scores.items() if v == best_score]
    if len(tied) > 1:
        return "A0_UNCLASSIFIED"

    return best_key


def classify_all(db_path: str = DB_PATH) -> int:
    """Classify every weekly_profile row and write classified_archetype
    back. Returns the number of rows updated."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM weekly_profile", conn)
    if df.empty:
        conn.close()
        return 0

    df["classified_archetype"] = df.apply(classify_row, axis=1)
    conn.executemany(
        "UPDATE weekly_profile SET classified_archetype = ? WHERE account_id = ? AND week_start = ?",
        list(zip(df["classified_archetype"], df["account_id"], df["week_start"])),
    )
    conn.commit()
    conn.close()
    return len(df)


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    n = classify_all()
    print(f"Classified {n} weekly_profile rows in {time.perf_counter() - t0:.2f}s")
