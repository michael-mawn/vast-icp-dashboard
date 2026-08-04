"""
Full reset-and-rebuild pipeline: wipes the DB and regenerates everything
from scratch with a fixed seed. Used both for the app's first-load
bootstrap and V4's "Regenerate" button (S8) — same code path either way,
so a slider-triggered run and app startup can never silently diverge.
"""

# See src/generate.py for why this is required: `X | None` annotations
# below raise TypeError at import time on Python <3.10 without it.
from __future__ import annotations

import os
import time

from src.classify import classify_all
from src.features import write_weekly_profiles
from src.generate import generate_dataset
from src.migrate_analysis import compute_and_store_churn_dates
from src.schema import DB_PATH, migrate


def rebuild(
    db_path: str = DB_PATH,
    archetype_mix: dict | None = None,
    forward_p: float | None = None,
    backward_p: float | None = None,
    churn_p: float | None = None,
    churn_days: int | None = None,
) -> dict:
    """Deletes any existing DB file and rebuilds it end to end: schema,
    generation, features, classification, churn dates. Returns timing."""
    t0 = time.perf_counter()
    if os.path.exists(db_path):
        os.remove(db_path)
    migrate(db_path)

    gen_report = generate_dataset(
        archetype_mix=archetype_mix, forward_p=forward_p, backward_p=backward_p,
        churn_p=churn_p, db_path=db_path,
    )
    write_weekly_profiles(db_path)
    classify_all(db_path)
    compute_and_store_churn_dates(db_path, churn_days)

    return {
        "total_seconds": round(time.perf_counter() - t0, 2),
        "generation_report": gen_report,
    }
