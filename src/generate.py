"""
Synthetic data generator.

S2 scope: A1 (Tinkerer) only, proving the pipeline end to end and
benchmarking row/memory/time cost so the A3 rental cap (config/settings.py
A3_MAX_RENTALS_PER_ACCOUNT) can be set from measurement rather than guesswork.
S3 replaces the provisional bits marked below with all four archetypes,
calibrated mix/migration parameters, and deliberate feature overlap.

Provisional for S2 (flagged, not final):
- signup dates drawn uniformly across the window, not the accelerating
  curve PRD §8 calls for (SIGNUP_GROWTH_RATE is decided at S3)
- session frequency / deposit cadence are reasonable stand-ins, not
  PRD-derived numbers (PRD gives shape, e.g. "small infrequent deposits",
  not a rate)
"""

import datetime as dt
import time

import numpy as np
import pandas as pd

from config.settings import N_ACCOUNTS, RANDOM_SEED, SIMULATION_MONTHS
from src.schema import DB_PATH, migrate
import sqlite3

SIMULATION_END = dt.date.today()
SIMULATION_START = SIMULATION_END - dt.timedelta(days=30 * SIMULATION_MONTHS)

CONSUMER_GPUS = ["3090", "4090", "5090"]
DATACENTER_GPUS = ["A100", "H100", "B200"]

A1_TEMPLATES = ["ollama", "text_generation_webui", "stable_diffusion_webui", "base_pytorch_jupyter"]

TEMPLATE_CATALOG = [
    # (name, category, implied_workload)
    ("Ollama", "ollama", "A1"),
    ("text-generation-webui", "text_generation_webui", "A1"),
    ("Stable Diffusion WebUI", "stable_diffusion_webui", "A1"),
    ("Base PyTorch/Jupyter", "base_pytorch_jupyter", "A1/A2"),
    ("Axolotl", "axolotl", "A2"),
    ("torchtune", "torchtune", "A2"),
    ("LLaMA-Factory", "llama_factory", "A2"),
    ("Unsloth", "unsloth", "A2"),
    ("Whisper / faster-whisper", "whisper", "A3"),
    ("Offline vLLM batch", "offline_vllm_batch", "A3"),
    ("Embedding pipeline", "embedding_pipeline", "A3"),
    ("Render job", "render_job", "A3"),
    ("vLLM serving", "vllm_serving", "A4"),
    ("TGI", "tgi", "A4"),
    ("SGLang", "sglang", "A4"),
    ("TensorRT-LLM", "tensorrt_llm", "A4"),
]


def _write_templates(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.DataFrame(TEMPLATE_CATALOG, columns=["name", "category", "implied_workload"])
    df.insert(0, "template_id", range(1, len(df) + 1))
    df.to_sql("templates", conn, if_exists="append", index=False)
    return df


def _generate_machines(conn: sqlite3.Connection, rng: np.random.Generator, n_machines: int = 300) -> pd.DataFrame:
    gpu_model = rng.choice(CONSUMER_GPUS, size=n_machines)
    verified = rng.random(n_machines) < 0.35
    df = pd.DataFrame({
        "machine_id": range(1, n_machines + 1),
        "gpu_model": gpu_model,
        "gpu_count": rng.integers(1, 5, size=n_machines),
        "verified": verified.astype(int),
        "reliability_score": rng.uniform(0.85, 0.999, size=n_machines).round(4),
        "region": rng.choice(["us-east", "us-west", "eu", "apac"], size=n_machines),
        "base_price_per_hour": rng.uniform(0.15, 0.55, size=n_machines).round(3),
    })
    df.to_sql("machines", conn, if_exists="append", index=False)
    return df


def _generate_accounts_a1(rng: np.random.Generator, n_accounts: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_days = (SIMULATION_END - SIMULATION_START).days
    signup_offsets = rng.integers(0, window_days, size=n_accounts)  # provisional: uniform, see module docstring
    signup_dates = [SIMULATION_START + dt.timedelta(days=int(o)) for o in signup_offsets]

    accounts = pd.DataFrame({
        "account_id": range(1, n_accounts + 1),
        "signup_date": [d.isoformat() for d in signup_dates],
        "first_deposit_amount": rng.uniform(5, 40, size=n_accounts).round(2),
        "auto_reload_enabled": (rng.random(n_accounts) < 0.05).astype(int),  # rare for Tinkerers
        "churn_date": None,
    })
    ground_truth = pd.DataFrame({
        "account_id": accounts["account_id"],
        "archetype": "A1_TINKERER",
    })
    return accounts, ground_truth


def _generate_rentals_a1(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    machines: pd.DataFrame,
    templates: pd.DataFrame,
) -> pd.DataFrame:
    """Vectorized: first decide how many sessions each account gets (numpy),
    then build all row-level fields as whole-array numpy operations. No
    per-rental Python-level DataFrame lookups — those were the dominant cost
    in the first S2 pass (~14s for 100k rows; see build notes)."""
    a1_template_ids = templates.loc[templates["category"].isin(A1_TEMPLATES), "template_id"].to_numpy()
    machine_ids = machines["machine_id"].to_numpy()
    machine_price = machines.set_index("machine_id")["base_price_per_hour"].to_dict()
    machine_price_arr = np.array([machine_price[m] for m in machine_ids])

    signup_dates = pd.to_datetime(accounts["signup_date"]).to_numpy()
    active_weeks = np.maximum(1, ((np.datetime64(SIMULATION_END) - signup_dates) / np.timedelta64(1, "D")).astype(int) // 7)

    # Total (account, week) pairs, expanded once.
    account_ids = accounts["account_id"].to_numpy()
    week_counts = active_weeks
    total_account_weeks = int(week_counts.sum())

    acct_week_account_id = np.repeat(account_ids, week_counts)
    acct_week_signup = np.repeat(signup_dates, week_counts)
    week_idx_within_account = np.concatenate([np.arange(c) for c in week_counts])
    week_start = acct_week_signup + (week_idx_within_account * 7).astype("timedelta64[D]")

    sessions_per_week = rng.poisson(1.3, size=total_account_weeks)  # provisional: infrequent, see module docstring
    n_rentals = int(sessions_per_week.sum())

    row_account_id = np.repeat(acct_week_account_id, sessions_per_week)
    row_week_start = np.repeat(week_start, sessions_per_week)

    duration_hours = rng.lognormal(mean=np.log(1.5), sigma=0.7, size=n_rentals)
    duration_hours = np.minimum(duration_hours, 20.0)  # cap outliers; keeps median well under 4h

    start_offset_hours = rng.uniform(19, 19 + 48, size=n_rentals)  # evening-skewed, spread over the week
    start_ts = row_week_start + (start_offset_hours * np.timedelta64(1, "h"))
    end_ts = start_ts + (duration_hours * np.timedelta64(1, "h"))

    instance_type = rng.choice(["interruptible", "on_demand", "reserved"], p=[0.85, 0.15, 0.0], size=n_rentals)
    launch_method = rng.choice(["console", "programmatic"], p=[0.9, 0.1], size=n_rentals)
    machine_choice_idx = rng.integers(0, len(machine_ids), size=n_rentals)
    row_machine_id = machine_ids[machine_choice_idx]
    row_price = machine_price_arr[machine_choice_idx]
    row_template_id = rng.choice(a1_template_ids, size=n_rentals)

    total_cost = np.round(duration_hours * row_price, 2)

    df = pd.DataFrame({
        "rental_id": np.arange(1, n_rentals + 1),
        "account_id": row_account_id,
        "machine_id": row_machine_id,
        "template_id": row_template_id,
        "start_ts": pd.Series(start_ts).dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_ts": pd.Series(end_ts).dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "instance_type": instance_type,
        "launch_method": launch_method,
        "end_reason": "user",
        "gpu_hours": np.round(duration_hours, 3),
        "total_cost": total_cost,
    })
    # drop any session that starts after the simulation window closes
    df = df[pd.to_datetime(df["start_ts"]) < pd.Timestamp(SIMULATION_END)].reset_index(drop=True)
    df["rental_id"] = np.arange(1, len(df) + 1)
    return df


def _generate_deposits_a1(rng: np.random.Generator, accounts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    deposit_id = 1
    for account_id, signup_iso in zip(accounts["account_id"], accounts["signup_date"]):
        signup = dt.date.fromisoformat(signup_iso)
        cursor = signup
        while cursor < SIMULATION_END:
            cursor += dt.timedelta(days=int(rng.integers(28, 56)))  # infrequent, provisional
            if cursor >= SIMULATION_END:
                break
            rows.append((deposit_id, int(account_id), cursor.isoformat(), round(float(rng.uniform(5, 30)), 2)))
            deposit_id += 1
    return pd.DataFrame(rows, columns=["deposit_id", "account_id", "ts", "amount"])


def run_a1_only(n_accounts: int = N_ACCOUNTS, seed: int = RANDOM_SEED) -> dict:
    """Generate an A1-only synthetic dataset, write it to SQLite, and return
    a benchmark report (row counts, wall time, estimated in-memory size)."""
    rng = np.random.default_rng(seed)
    migrate(DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    report = {}
    t0 = time.perf_counter()

    templates = _write_templates(conn)
    machines = _generate_machines(conn, rng)
    accounts, ground_truth = _generate_accounts_a1(rng, n_accounts)
    accounts.to_sql("accounts", conn, if_exists="append", index=False)
    ground_truth.to_sql("ground_truth", conn, if_exists="append", index=False)

    t_rentals_start = time.perf_counter()
    rentals = _generate_rentals_a1(rng, accounts, machines, templates)
    t_rentals = time.perf_counter() - t_rentals_start
    rentals.to_sql("rentals", conn, if_exists="append", index=False)

    deposits = _generate_deposits_a1(rng, accounts)
    deposits.to_sql("deposits", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    total_time = time.perf_counter() - t0

    import os
    report["n_accounts"] = n_accounts
    report["n_rentals"] = len(rentals)
    report["n_deposits"] = len(deposits)
    report["rentals_per_account_mean"] = len(rentals) / n_accounts
    report["rentals_generation_seconds"] = round(t_rentals, 3)
    report["total_pipeline_seconds"] = round(total_time, 3)
    report["rentals_df_memory_mb"] = round(rentals.memory_usage(deep=True).sum() / 1e6, 3)
    report["bytes_per_rental_row"] = round(rentals.memory_usage(deep=True).sum() / max(len(rentals), 1), 1)
    report["db_file_size_mb"] = round(os.path.getsize(DB_PATH) / 1e6, 3)
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(run_a1_only(), indent=2))
