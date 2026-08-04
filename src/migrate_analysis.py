"""
V2 migration matrix: day-30 vs day-180 classified archetype, each endpoint
using the two-week-held classification (build plan: resolves PRD §5's
internal contradiction between "holds 2+ weeks" and "day-30 vs day-180
snapshot" by applying the hold rule AT each snapshot rather than treating
them as two competing definitions).

Eligible cohort = accounts with a full 180-day observation window as of
today. Ineligible accounts (most of the base, since signups accelerate)
are excluded from the matrix body but counted and reported, never silently
dropped (PRD §6 / CLAUDE.md).

Churn: PRD §9 reserves accounts.churn_date but nothing populated it before
this module. Defined here as CHURN_DAYS_NO_ACTIVITY (60) consecutive days
with no rental, evaluated as of a given reference date (day 180 for the
matrix's terminal column; "today" for the persisted accounts.churn_date).
"""

import datetime as dt
import sqlite3

import numpy as np
import pandas as pd

from config.settings import CHURN_DAYS_NO_ACTIVITY, MIGRATION_SNAPSHOT_DAYS, MIGRATION_HOLD_WEEKS

ARCHETYPE_ORDER = ["A1_TINKERER", "A2_RESEARCHER", "A3_BATCH", "A4_PRODUCTION", "A0_UNCLASSIFIED"]


def _held_classification(profile_rows: pd.DataFrame, as_of_date: np.datetime64) -> str:
    """The classified_archetype that has held for >=MIGRATION_HOLD_WEEKS
    consecutive weekly snapshots as of as_of_date. Looks at the closest
    week_start <= as_of_date and the (MIGRATION_HOLD_WEEKS - 1) prior weeks;
    if they don't all agree, or there isn't enough history yet, the
    classification is "not held" -> reported as A0 (insufficient stability,
    not insufficient activity, but the same terminal bucket)."""
    eligible = profile_rows[profile_rows["week_start"] <= as_of_date].sort_values("week_start")
    if len(eligible) < MIGRATION_HOLD_WEEKS:
        return "A0_UNCLASSIFIED"
    recent = eligible.tail(MIGRATION_HOLD_WEEKS)
    if recent["classified_archetype"].nunique() == 1:
        return recent["classified_archetype"].iloc[0]
    return "A0_UNCLASSIFIED"


def _churned_by(rentals: pd.DataFrame, reference_date: np.datetime64, churn_days: int) -> bool:
    """True if, as of reference_date, the account had gone churn_days
    consecutive days with no rental (i.e. its last rental ended more than
    that many days before reference_date, and no rental starts between
    then and reference_date — guaranteed by using the max end_ts among
    rentals starting on/before reference_date)."""
    prior = rentals[rentals["start_ts"] <= reference_date]
    if prior.empty:
        return False  # never active yet is A0, not churned — a distinct concept
    last_end = prior["end_ts"].max()
    return (reference_date - last_end) >= np.timedelta64(churn_days, "D")


def compute_migration_matrix(db_path: str, churn_days: int | None = None) -> dict:
    churn_days = churn_days if churn_days is not None else CHURN_DAYS_NO_ACTIVITY
    conn = sqlite3.connect(db_path)
    accounts = pd.read_sql("SELECT account_id, signup_date FROM accounts", conn)
    wp = pd.read_sql("SELECT account_id, week_start, classified_archetype FROM weekly_profile", conn)
    rentals = pd.read_sql("SELECT account_id, start_ts, end_ts FROM rentals", conn)
    conn.close()

    accounts["signup_date"] = pd.to_datetime(accounts["signup_date"])
    wp["week_start"] = pd.to_datetime(wp["week_start"])
    rentals["start_ts"] = pd.to_datetime(rentals["start_ts"], format="ISO8601")
    rentals["end_ts"] = pd.to_datetime(rentals["end_ts"], format="ISO8601")

    today = pd.Timestamp(dt.date.today())
    day30, day180 = MIGRATION_SNAPSHOT_DAYS

    accounts["day180_date"] = accounts["signup_date"] + pd.Timedelta(days=day180)
    eligible_mask = accounts["day180_date"] <= today
    eligible = accounts[eligible_mask].copy()
    ineligible_count = int((~eligible_mask).sum())

    wp_by_account = {aid: df for aid, df in wp.groupby("account_id", sort=False)}
    rentals_by_account = {aid: df for aid, df in rentals.groupby("account_id", sort=False)}

    rows = []
    for account_id, signup in zip(eligible["account_id"], eligible["signup_date"]):
        profile_rows = wp_by_account.get(account_id, pd.DataFrame(columns=["week_start", "classified_archetype"]))
        acct_rentals = rentals_by_account.get(account_id, pd.DataFrame(columns=["start_ts", "end_ts"]))

        day30_date = signup + pd.Timedelta(days=day30)
        day180_date = signup + pd.Timedelta(days=day180)

        state_30 = _held_classification(profile_rows, day30_date)

        if _churned_by(acct_rentals, day180_date, churn_days):
            state_180 = "CHURNED"
        else:
            state_180 = _held_classification(profile_rows, day180_date)

        rows.append({"account_id": account_id, "day30": state_30, "day180": state_180})

    matrix_df = pd.DataFrame(rows)
    column_order = ARCHETYPE_ORDER + ["CHURNED"]
    if matrix_df.empty:
        counts = pd.DataFrame(0, index=ARCHETYPE_ORDER, columns=column_order)
    else:
        counts = pd.crosstab(matrix_df["day30"], matrix_df["day180"]).reindex(
            index=ARCHETYPE_ORDER, columns=column_order, fill_value=0
        )

    return {
        "counts": counts,
        "n_eligible": len(eligible),
        "n_ineligible": ineligible_count,
        "n_total": len(accounts),
    }


def compute_and_store_churn_dates(db_path: str, churn_days: int | None = None) -> int:
    """Populates accounts.churn_date (PRD §9, previously always NULL) using
    the same churn-window rule, evaluated as of today rather than a
    per-account day180 — a separate concern from the matrix's terminal
    column, which needs churn status AT day180, not today."""
    churn_days = churn_days if churn_days is not None else CHURN_DAYS_NO_ACTIVITY
    conn = sqlite3.connect(db_path)
    rentals = pd.read_sql("SELECT account_id, end_ts FROM rentals", conn)
    rentals["end_ts"] = pd.to_datetime(rentals["end_ts"], format="ISO8601")
    today = np.datetime64(dt.date.today())

    last_end = rentals.groupby("account_id")["end_ts"].max()
    churned = last_end[(today - last_end.values.astype("datetime64[D]")) >= np.timedelta64(churn_days, "D")]

    conn.executemany(
        "UPDATE accounts SET churn_date = ? WHERE account_id = ?",
        [(pd.Timestamp(ts).isoformat(), int(aid)) for aid, ts in churned.items()],
    )
    conn.commit()
    conn.close()
    return len(churned)
