"""
Computes weekly_profile from raw rentals/machines/templates — the trailing
30-day feature vector src/classify.py scores against. Recomputed weekly per
PRD §5. Reads only raw generated data, never ground_truth: this is the
input side of the classifier firewall (config/archetypes.py is the other).

Restart-after-interruption is detected structurally: an interrupted rental
(end_reason host_reclaim/machine_offline) counts as "restarted" if another
rental for the same account starts within 30 minutes of its end — matching
how src/generate.py actually constructs restart pairs, rather than assuming
a stored label.
"""

import datetime as dt

import numpy as np
import pandas as pd
import sqlite3

from config.settings import CLASSIFICATION_WINDOW_DAYS
from src.schema import DB_PATH

WEEK_STEP_DAYS = 7
RESTART_WINDOW_HOURS = 0.5


def _sweep_max_concurrent(starts: np.ndarray, ends: np.ndarray) -> int:
    if len(starts) == 0:
        return 0
    events = np.concatenate([starts, ends])
    deltas = np.concatenate([np.ones(len(starts), dtype=int), -np.ones(len(ends), dtype=int)])
    order = np.argsort(events, kind="stable")
    return int(np.max(np.cumsum(deltas[order])))


def _restart_rate(start_ts: np.ndarray, end_ts: np.ndarray, end_reason: np.ndarray) -> float:
    interrupted_mask = np.isin(end_reason, ["host_reclaim", "machine_offline"])
    n_interrupted = int(interrupted_mask.sum())
    if n_interrupted == 0:
        return np.nan
    sorted_starts = np.sort(start_ts)
    window = np.timedelta64(int(RESTART_WINDOW_HOURS * 3600), "s")
    restarted = 0
    for e in end_ts[interrupted_mask]:
        lo = np.searchsorted(sorted_starts, e, side="left")
        hi = np.searchsorted(sorted_starts, e + window, side="right")
        if hi > lo:
            restarted += 1
    return restarted / n_interrupted


def _mode_or_none(values: np.ndarray):
    if len(values) == 0:
        return None
    vals, counts = np.unique(values, return_counts=True)
    return vals[np.argmax(counts)]


def compute_weekly_profiles(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    rentals = pd.read_sql(
        "SELECT r.*, m.gpu_model, m.verified, t.category AS template_category "
        "FROM rentals r JOIN machines m ON r.machine_id = m.machine_id "
        "JOIN templates t ON r.template_id = t.template_id",
        conn,
    )
    accounts = pd.read_sql("SELECT account_id, signup_date FROM accounts", conn)
    conn.close()

    rentals["start_ts"] = pd.to_datetime(rentals["start_ts"], format="ISO8601").values.astype("datetime64[s]")
    rentals["end_ts"] = pd.to_datetime(rentals["end_ts"], format="ISO8601").values.astype("datetime64[s]")
    rentals = rentals.sort_values("start_ts")

    global_end = np.datetime64(dt.date.today())
    rows = []

    rentals_by_account = {aid: df for aid, df in rentals.groupby("account_id", sort=False)}

    for account_id, signup_iso in zip(accounts["account_id"], accounts["signup_date"]):
        acct = rentals_by_account.get(account_id)
        if acct is None or acct.empty:
            continue
        starts = acct["start_ts"].to_numpy()
        ends = acct["end_ts"].to_numpy()
        instance_type = acct["instance_type"].to_numpy()
        launch_method = acct["launch_method"].to_numpy()
        end_reason = acct["end_reason"].to_numpy()
        gpu_hours = acct["gpu_hours"].to_numpy()
        verified = acct["verified"].to_numpy()
        is_serverless = acct["is_serverless"].to_numpy()
        gpu_model = acct["gpu_model"].to_numpy()
        template_category = acct["template_category"].to_numpy()

        signup = np.datetime64(dt.date.fromisoformat(signup_iso))
        last_activity_end = ends.max()
        sim_end_for_account = min(global_end, last_activity_end + np.timedelta64(365, "D"))

        week_start = signup
        while week_start <= sim_end_for_account:
            window_lo = week_start - np.timedelta64(CLASSIFICATION_WINDOW_DAYS, "D")
            lo = np.searchsorted(starts, window_lo, side="left")
            hi = np.searchsorted(starts, week_start, side="left")
            if hi <= lo:
                week_start += np.timedelta64(WEEK_STEP_DAYS, "D")
                continue

            w_starts, w_ends = starts[lo:hi], ends[lo:hi]
            w_instance_type, w_launch, w_reason = instance_type[lo:hi], launch_method[lo:hi], end_reason[lo:hi]
            w_gpu_hours, w_verified, w_serverless = gpu_hours[lo:hi], verified[lo:hi], is_serverless[lo:hi]
            w_gpu_model, w_template_cat = gpu_model[lo:hi], template_category[lo:hi]

            n = len(w_starts)
            rows.append({
                "account_id": int(account_id),
                "week_start": pd.Timestamp(week_start).date().isoformat(),
                "classified_archetype": None,  # filled by src/classify.py
                "rental_count": n,
                "gpu_hours_sum": round(float(w_gpu_hours.sum()), 3),
                "interruptible_share": round(float((w_instance_type == "interruptible").mean()), 4),
                "on_demand_or_reserved_share": round(float(np.isin(w_instance_type, ["on_demand", "reserved"]).mean()), 4),
                "median_session_hours": round(float(np.median(w_gpu_hours)), 3),
                "max_concurrent_instances": _sweep_max_concurrent(w_starts, w_ends),
                "verified_host_share": round(float(w_verified.mean()), 4),
                "console_launch_share": round(float((w_launch == "console").mean()), 4),
                "restart_after_interruption_rate": _restart_rate(w_starts, w_ends, w_reason),
                "serverless_present": int(w_serverless.any()),
                "dominant_gpu_class": _mode_or_none(w_gpu_model),
                "dominant_template_category": _mode_or_none(w_template_cat),
            })
            week_start += np.timedelta64(WEEK_STEP_DAYS, "D")

    return pd.DataFrame(rows)


def write_weekly_profiles(db_path: str = DB_PATH) -> int:
    df = compute_weekly_profiles(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM weekly_profile")  # derived table, safe to rebuild each run
    df.to_sql("weekly_profile", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return len(df)


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    n = write_weekly_profiles()
    print(f"Wrote {n} weekly_profile rows in {time.perf_counter() - t0:.2f}s")
