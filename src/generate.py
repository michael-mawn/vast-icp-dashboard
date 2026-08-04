"""
Synthetic data generator — S3: all four archetypes.

Ground truth (PRD §8): each account is assigned ONE archetype at signup
(config.generator.ARCHETYPE_MIX_AT_SIGNUP), stored in ground_truth and never
read by src/classify.py. Behavior is generated FROM archetype profiles below.

Migration tension, resolved (see config/generator.py docstring): "migration"
implies an account's TRUE archetype can drift, but the schema (correctly,
per CLAUDE.md) stores only one ground_truth_archetype per account. This
generator tracks a per-account, per-month TRUE archetype trajectory
internally — driven by MIGRATION_GRAPH / FORWARD / BACKWARD / CHURN rates —
purely to select which behavior profile shapes that month's rentals. That
trajectory is never persisted. V2's migration matrix is built entirely from
the CLASSIFIER's own weekly re-classification of the resulting behavior
(day 30 vs day 180, each held 2+ weeks) — not from this internal trajectory.
This is why cutting V3 (classifier-vs-ground-truth accuracy) is safe: no
view depends on the true trajectory being retrievable.

Feature overlap (FEATURE_OVERLAP / ADJACENT_ARCHETYPE): with that
probability, a given account-month's behavior is drawn from the adjacent
archetype's profile instead of its own, at the two boundaries spec review
identified as genuinely ambiguous in PRD §4 (A1<->A3, A2<->A4). This makes
classification a real exercise without building full continuous latent
profiles (scope cut from the original spec-review proposal).
"""

# Defers annotation evaluation to strings — without this, `dict | None`-style
# hints below raise TypeError at import time on Python <3.10 (the `|` union
# operator on types was added in PEP 604 / 3.10). This broke the live
# Streamlit Cloud deploy: runtime.txt requests 3.12, but Cloud's actual
# interpreter didn't honor it. Caught by comparing the deploy traceback
# against a runtime-version hypothesis, since a from-scratch local clone
# under the intended 3.12 never surfaced this.
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd

from config.settings import N_ACCOUNTS, RANDOM_SEED, SIMULATION_MONTHS, A3_MAX_RENTALS_PER_ACCOUNT
from config.generator import (
    ARCHETYPE_MIX_AT_SIGNUP, MIGRATION_GRAPH, MIGRATION_FORWARD_MONTHLY,
    MIGRATION_BACKWARD_MONTHLY, CHURN_PROBABILITY_MONTHLY, FEATURE_OVERLAP,
    ADJACENT_ARCHETYPE, HOST_FAILURE_RATE, SIGNUP_GROWTH_POWER,
)
from src.schema import DB_PATH, migrate
import sqlite3

SIMULATION_END = dt.date.today()
SIMULATION_START = SIMULATION_END - dt.timedelta(days=30 * SIMULATION_MONTHS)

CONSUMER_GPUS = ["3090", "4090", "5090"]
DATACENTER_GPUS = ["A100", "H100", "B200"]

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

# Per-archetype generation profile. Distribution centers sit inside the
# classifier threshold ranges transcribed in config/archetypes.py — these
# are generation shape choices (PRD gives a range/description, not an exact
# number), not new thresholds requiring sign-off.
ARCHETYPE_PROFILES = {
    "A1_TINKERER": {
        "gpu_pool": CONSUMER_GPUS,
        "templates": ["ollama", "text_generation_webui", "stable_diffusion_webui", "base_pytorch_jupyter"],
        "instance_type_p": {"interruptible": 0.85, "on_demand": 0.15, "reserved": 0.0},
        "launch_console_p": 0.90,
        "duration_hours": lambda rng, n: np.minimum(rng.lognormal(np.log(1.5), 0.7, n), 20.0),
        "concurrency": lambda rng, n: np.ones(n, dtype=int),
        "bursts_per_month": lambda rng, n: rng.poisson(5.6, n),  # ~1.3/week, matches S2 benchmark
        "restart_after_interruption_p": 0.40,
        "prefer_verified": False,
        "deposit_amount": lambda rng, n: rng.uniform(5, 40, n),
        "deposit_interval_days": (28, 56),
        "auto_reload_p": 0.05,
    },
    "A2_RESEARCHER": {
        "gpu_pool": ["A100", "H100", "4090"],
        "templates": ["axolotl", "torchtune", "llama_factory", "unsloth", "base_pytorch_jupyter"],
        "instance_type_p": {"interruptible": 0.50, "on_demand": 0.45, "reserved": 0.05},
        "launch_console_p": 0.55,
        "duration_hours": lambda rng, n: rng.uniform(6, 72, n),
        "concurrency": lambda rng, n: rng.integers(1, 5, n),
        "bursts_per_month": lambda rng, n: rng.poisson(1.1, n),  # episodic — idle weeks between runs
        "restart_after_interruption_p": 0.50,
        "prefer_verified": False,
        "deposit_amount": lambda rng, n: rng.uniform(50, 300, n),
        "deposit_interval_days": (30, 90),
        "auto_reload_p": 0.10,
    },
    "A3_BATCH": {
        "gpu_pool": CONSUMER_GPUS,
        "templates": ["whisper", "offline_vllm_batch", "embedding_pipeline", "render_job"],
        "instance_type_p": {"interruptible": 0.90, "on_demand": 0.10, "reserved": 0.0},
        "launch_console_p": 0.10,
        "duration_hours": lambda rng, n: np.minimum(rng.lognormal(np.log(1.5), 0.6, n), 8.0),
        "concurrency": lambda rng, n: rng.integers(5, 51, n),  # defining trait, PRD §4
        "bursts_per_month": lambda rng, n: rng.poisson(1.0, n),  # regular, repeating
        "restart_after_interruption_p": 0.85,  # >80% defining trait
        "prefer_verified": False,
        "deposit_amount": lambda rng, n: rng.uniform(20, 60, n),  # generator flavor only, not a classifier feature
        "deposit_interval_days": (25, 35),
        "auto_reload_p": 0.15,
    },
    "A4_PRODUCTION": {
        "gpu_pool": DATACENTER_GPUS,
        "templates": ["vllm_serving", "tgi", "sglang", "tensorrt_llm"],
        "instance_type_p": {"interruptible": 0.05, "on_demand": 0.60, "reserved": 0.35},
        "launch_console_p": 0.05,
        "duration_hours": lambda rng, n: np.minimum(rng.lognormal(np.log(220), 0.5, n), 700.0),  # >168h
        "concurrency": lambda rng, n: rng.integers(2, 21, n),
        "bursts_per_month": lambda rng, n: rng.poisson(1.0, n),  # flat 24/7 — few long-lived blocks
        "restart_after_interruption_p": 0.90,
        "prefer_verified": True,  # >90% verified-host share, PRD §4
        "deposit_amount": lambda rng, n: rng.uniform(200, 800, n),
        "deposit_interval_days": (14, 21),
        "auto_reload_p": 0.80,
    },
}

SERVERLESS_PRESENT_P = {"A4_PRODUCTION": 0.30}  # satisfies A4's session-length OR criterion for short-session accounts


def _write_templates(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.DataFrame(TEMPLATE_CATALOG, columns=["name", "category", "implied_workload"])
    df.insert(0, "template_id", range(1, len(df) + 1))
    df.to_sql("templates", conn, if_exists="append", index=False)
    return df


def _generate_machines(conn: sqlite3.Connection, rng: np.random.Generator,
                        n_consumer: int = 300, n_datacenter: int = 150) -> pd.DataFrame:
    n_machines = n_consumer + n_datacenter
    gpu_model = np.concatenate([
        rng.choice(CONSUMER_GPUS, size=n_consumer),
        rng.choice(DATACENTER_GPUS, size=n_datacenter),
    ])
    is_datacenter = np.concatenate([np.zeros(n_consumer, dtype=bool), np.ones(n_datacenter, dtype=bool)])
    verified_p = np.where(is_datacenter, 0.60, 0.20)
    verified = rng.random(n_machines) < verified_p
    df = pd.DataFrame({
        "machine_id": range(1, n_machines + 1),
        "gpu_model": gpu_model,
        "gpu_count": np.where(is_datacenter, rng.integers(1, 9, n_machines), rng.integers(1, 3, n_machines)),
        "verified": verified.astype(int),
        "reliability_score": rng.uniform(0.85, 0.999, size=n_machines).round(4),
        "region": rng.choice(["us-east", "us-west", "eu", "apac"], size=n_machines),
        # magnitude discipline (PRD §8): real verified H100 SXM ~$2.89/GPU-hr
        "base_price_per_hour": np.where(
            is_datacenter, rng.uniform(1.20, 3.10, n_machines), rng.uniform(0.15, 0.55, n_machines)
        ).round(3),
    })
    df.to_sql("machines", conn, if_exists="append", index=False)
    return df


def _generate_accounts(
    rng: np.random.Generator, n_accounts: int, archetype_mix: dict
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    window_days = (SIMULATION_END - SIMULATION_START).days
    # Accelerating signups (PRD §8): rng.power skews samples toward 1 (late window).
    signup_frac = rng.power(SIGNUP_GROWTH_POWER, size=n_accounts)
    signup_offsets = (signup_frac * window_days).astype(int)
    signup_dates = [SIMULATION_START + dt.timedelta(days=int(o)) for o in signup_offsets]

    archetype_keys = list(archetype_mix.keys())
    archetype_probs = list(archetype_mix.values())
    signup_archetype = rng.choice(archetype_keys, size=n_accounts, p=archetype_probs)

    accounts = pd.DataFrame({
        "account_id": range(1, n_accounts + 1),
        "signup_date": [d.isoformat() for d in signup_dates],
        "first_deposit_amount": rng.uniform(5, 100, size=n_accounts).round(2),
        "auto_reload_enabled": 0,  # set properly once trajectory/profile is known, see run()
        "churn_date": None,
    })
    ground_truth = pd.DataFrame({"account_id": accounts["account_id"], "archetype": signup_archetype})
    return accounts, ground_truth, signup_archetype


def _simulate_trajectories(
    rng: np.random.Generator, signup_archetype: np.ndarray, signup_dates: list,
    forward_p: float, backward_p: float, churn_p: float,
) -> list:
    """For each account, walk month by month from signup to SIMULATION_END.
    Returns a list (per account) of (month_start_date, true_archetype_or_None)
    tuples; None means churned (no further activity generated)."""
    trajectories = []
    for archetype, signup in zip(signup_archetype, signup_dates):
        current = archetype
        cursor = signup
        months = []
        while cursor < SIMULATION_END:
            if rng.random() < churn_p:
                break
            months.append((cursor, current))
            forward_edges = MIGRATION_GRAPH.get(current, [])
            r = rng.random()
            if forward_edges and r < forward_p:
                current = forward_edges[rng.integers(0, len(forward_edges))]
            elif r < forward_p + backward_p:
                backward_candidates = [k for k, v in MIGRATION_GRAPH.items() if current in v]
                if backward_candidates:
                    current = backward_candidates[rng.integers(0, len(backward_candidates))]
            cursor += dt.timedelta(days=30)
        trajectories.append(months)
    return trajectories


def _generate_rentals_and_deposits(
    rng: np.random.Generator,
    accounts: pd.DataFrame,
    trajectories: list,
    machines: pd.DataFrame,
    templates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    machine_ids_by_gpu: dict = {g: machines.loc[machines["gpu_model"] == g, "machine_id"].to_numpy() for g in CONSUMER_GPUS + DATACENTER_GPUS}
    machine_price = machines.set_index("machine_id")["base_price_per_hour"].to_dict()
    machine_verified_ids = machines.loc[machines["verified"] == 1, "machine_id"].to_numpy()
    template_ids_by_category = templates.set_index("category")["template_id"].to_dict()

    rental_rows = []
    deposit_rows = []
    a3_rentals_so_far: dict = {}
    account_auto_reload: dict = {}

    for account_id, months in zip(accounts["account_id"], trajectories):
        for month_start, true_archetype in months:
            # Feature overlap blends only the discriminating fields (instance
            # type mix, launch method) toward the adjacent archetype — not
            # the whole profile. Blending concurrency/duration too would let
            # a high-concurrency neighbor (e.g. A3's 5-50 vs A1's fixed 1)
            # dominate the account's rental VOLUME from a single blended
            # month, which overstates the intended ~15% ambiguity by an
            # order of magnitude. Caught by validation after the first pass.
            profile = dict(ARCHETYPE_PROFILES[true_archetype])
            if true_archetype in ADJACENT_ARCHETYPE and rng.random() < FEATURE_OVERLAP:
                neighbor = ARCHETYPE_PROFILES[ADJACENT_ARCHETYPE[true_archetype]]
                profile["instance_type_p"] = neighbor["instance_type_p"]
                profile["launch_console_p"] = neighbor["launch_console_p"]
            effective_archetype = true_archetype
            n_bursts = int(profile["bursts_per_month"](rng, 1)[0])
            if n_bursts <= 0:
                continue

            for _ in range(n_bursts):
                concurrency = int(profile["concurrency"](rng, 1)[0])
                if effective_archetype == "A3_BATCH":
                    remaining = A3_MAX_RENTALS_PER_ACCOUNT - a3_rentals_so_far.get(account_id, 0)
                    concurrency = max(0, min(concurrency, remaining))
                    if concurrency == 0:
                        continue
                    a3_rentals_so_far[account_id] = a3_rentals_so_far.get(account_id, 0) + concurrency

                gpu_pool = profile["gpu_pool"]
                candidate_ids = np.concatenate([machine_ids_by_gpu[g] for g in gpu_pool if len(machine_ids_by_gpu[g])])
                if profile["prefer_verified"]:
                    verified_candidates = np.intersect1d(candidate_ids, machine_verified_ids)
                    if len(verified_candidates) > 0 and rng.random() < 0.9:
                        candidate_ids = verified_candidates

                burst_start = dt.datetime.combine(month_start, dt.time(hour=19)) + dt.timedelta(
                    days=float(rng.uniform(0, 28)), hours=float(rng.uniform(0, 5))
                )
                if burst_start.date() >= SIMULATION_END:
                    continue

                duration_hours = profile["duration_hours"](rng, concurrency)
                machine_idx = rng.integers(0, len(candidate_ids), concurrency)
                sel_machine_ids = candidate_ids[machine_idx]
                sel_prices = np.array([machine_price[m] for m in sel_machine_ids])
                template_pool = np.array([template_ids_by_category[c] for c in profile["templates"]])
                sel_template_ids = rng.choice(template_pool, concurrency)

                p = profile["instance_type_p"]
                instance_type = rng.choice(list(p.keys()), size=concurrency, p=list(p.values()))
                launch_method = rng.choice(
                    ["console", "programmatic"], size=concurrency,
                    p=[profile["launch_console_p"], 1 - profile["launch_console_p"]],
                )

                start_ts = np.full(concurrency, burst_start, dtype="datetime64[s]")
                end_ts = start_ts + (duration_hours * np.timedelta64(1, "h")).astype("timedelta64[s]")

                interrupted = (instance_type == "interruptible") & (rng.random(concurrency) < HOST_FAILURE_RATE)
                end_reason = np.full(concurrency, "user", dtype=object)
                if interrupted.any():
                    end_reason[interrupted] = rng.choice(["host_reclaim", "machine_offline"], size=interrupted.sum())

                total_cost = np.round(duration_hours * sel_prices, 2)
                # Burst-level serverless flag (PRD §4 A4 OR-criterion): satisfies
                # classification independently of session length for accounts
                # using serverless endpoints rather than long-lived instances.
                is_serverless = int(rng.random() < SERVERLESS_PRESENT_P.get(true_archetype, 0.0))

                for i in range(concurrency):
                    rental_rows.append((
                        int(account_id), int(sel_machine_ids[i]), int(sel_template_ids[i]),
                        pd.Timestamp(start_ts[i]).isoformat(), pd.Timestamp(end_ts[i]).isoformat(),
                        str(instance_type[i]), str(launch_method[i]), str(end_reason[i]),
                        round(float(duration_hours[i]), 3), float(total_cost[i]), is_serverless,
                    ))
                    if interrupted[i] and rng.random() < profile["restart_after_interruption_p"]:
                        gap_h = float(rng.uniform(0.05, 0.5))
                        restart_dur = float(profile["duration_hours"](rng, 1)[0])
                        restart_start = pd.Timestamp(end_ts[i]) + pd.Timedelta(hours=gap_h)
                        restart_end = restart_start + pd.Timedelta(hours=restart_dur)
                        rental_rows.append((
                            int(account_id), int(sel_machine_ids[i]), int(sel_template_ids[i]),
                            restart_start.isoformat(), restart_end.isoformat(),
                            str(instance_type[i]), str(launch_method[i]), "user",
                            round(restart_dur, 3), round(restart_dur * sel_prices[i], 2), is_serverless,
                        ))

            if profile["auto_reload_p"] > 0 and rng.random() < profile["auto_reload_p"] * 0.3:
                account_auto_reload[account_id] = True

            lo, hi = profile["deposit_interval_days"]
            if rng.random() < 30 / ((lo + hi) / 2):  # ~monthly chance scaled to this archetype's cadence
                amount = float(profile["deposit_amount"](rng, 1)[0])
                deposit_ts = month_start + dt.timedelta(days=int(rng.integers(0, 28)))
                if deposit_ts < SIMULATION_END:
                    deposit_rows.append((int(account_id), deposit_ts.isoformat(), round(amount, 2)))

    rentals = pd.DataFrame(rental_rows, columns=[
        "account_id", "machine_id", "template_id", "start_ts", "end_ts",
        "instance_type", "launch_method", "end_reason", "gpu_hours", "total_cost", "is_serverless",
    ])
    rentals.insert(0, "rental_id", range(1, len(rentals) + 1))

    deposits = pd.DataFrame(deposit_rows, columns=["account_id", "ts", "amount"])
    deposits.insert(0, "deposit_id", range(1, len(deposits) + 1))

    return rentals, deposits, account_auto_reload


def generate_dataset(
    n_accounts: int = N_ACCOUNTS,
    seed: int = RANDOM_SEED,
    archetype_mix: dict | None = None,
    forward_p: float | None = None,
    backward_p: float | None = None,
    churn_p: float | None = None,
    db_path: str = DB_PATH,
) -> dict:
    """Generate the full four-archetype synthetic dataset, write it to
    SQLite, and return a benchmark report. Overrides (V4 sliders, S8) fall
    back to the approved config/generator.py defaults when None — the
    defaults are never silently changed, only the in-memory run."""
    archetype_mix = archetype_mix if archetype_mix is not None else ARCHETYPE_MIX_AT_SIGNUP
    forward_p = forward_p if forward_p is not None else MIGRATION_FORWARD_MONTHLY
    backward_p = backward_p if backward_p is not None else MIGRATION_BACKWARD_MONTHLY
    churn_p = churn_p if churn_p is not None else CHURN_PROBABILITY_MONTHLY

    rng = np.random.default_rng(seed)
    migrate(db_path)
    conn = sqlite3.connect(db_path)

    t0 = time.perf_counter()

    templates = _write_templates(conn)
    machines = _generate_machines(conn, rng)
    accounts, ground_truth, signup_archetype = _generate_accounts(rng, n_accounts, archetype_mix)
    signup_dates = [dt.date.fromisoformat(d) for d in accounts["signup_date"]]

    trajectories = _simulate_trajectories(rng, signup_archetype, signup_dates, forward_p, backward_p, churn_p)

    t_gen_start = time.perf_counter()
    rentals, deposits, account_auto_reload = _generate_rentals_and_deposits(rng, accounts, trajectories, machines, templates)
    t_gen = time.perf_counter() - t_gen_start

    accounts["auto_reload_enabled"] = accounts["account_id"].map(lambda a: int(account_auto_reload.get(a, False)))

    accounts.to_sql("accounts", conn, if_exists="append", index=False)
    ground_truth.to_sql("ground_truth", conn, if_exists="append", index=False)
    rentals.to_sql("rentals", conn, if_exists="append", index=False)
    deposits.to_sql("deposits", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    total_time = time.perf_counter() - t0

    import os
    mix_counts = pd.Series(signup_archetype).value_counts().to_dict()
    return {
        "n_accounts": n_accounts,
        "signup_mix": mix_counts,
        "n_rentals": len(rentals),
        "n_deposits": len(deposits),
        "rentals_per_account_mean": round(len(rentals) / n_accounts, 2),
        "behavior_generation_seconds": round(t_gen, 3),
        "total_pipeline_seconds": round(total_time, 3),
        "rentals_df_memory_mb": round(rentals.memory_usage(deep=True).sum() / 1e6, 3),
        "db_file_size_mb": round(os.path.getsize(db_path) / 1e6, 3),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generate_dataset(), indent=2))
