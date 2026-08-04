"""
SQLite schema + additive migration runner.

Deviates from PRD.md §9 in one place, intentionally: ground_truth_archetype
is NOT a column on accounts. It lives in its own table, per CLAUDE.md's
non-negotiable convention ("Ground-truth archetype labels live in their own
table and are never read by the classifier at inference time"). Keeping it
on accounts would put it one careless join away from leaking into
src/classify.py. accounts holds only fields the classifier is allowed to see.

CLAUDE.md: "Schema changes use migrations that preserve existing rows.
Never drop and regenerate the database to accommodate a schema change."
Each schema version is a numbered, additive migration below.
"""

import os
import sqlite3

DB_PATH = "data/marketplace.db"

MIGRATIONS = [
    # v1: initial schema
    """
    CREATE TABLE IF NOT EXISTS accounts (
        account_id          INTEGER PRIMARY KEY,
        signup_date          TEXT NOT NULL,
        first_deposit_amount REAL,
        auto_reload_enabled  INTEGER NOT NULL DEFAULT 0,
        churn_date           TEXT
    );

    CREATE TABLE IF NOT EXISTS ground_truth (
        account_id  INTEGER PRIMARY KEY REFERENCES accounts(account_id),
        archetype   TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS machines (
        machine_id         INTEGER PRIMARY KEY,
        gpu_model          TEXT NOT NULL,
        gpu_count          INTEGER NOT NULL,
        verified           INTEGER NOT NULL DEFAULT 0,
        reliability_score  REAL,
        region             TEXT,
        base_price_per_hour REAL
    );

    CREATE TABLE IF NOT EXISTS templates (
        template_id      INTEGER PRIMARY KEY,
        name             TEXT NOT NULL,
        category         TEXT NOT NULL,
        implied_workload  TEXT
    );

    CREATE TABLE IF NOT EXISTS rentals (
        rental_id      INTEGER PRIMARY KEY,
        account_id     INTEGER NOT NULL REFERENCES accounts(account_id),
        machine_id     INTEGER NOT NULL REFERENCES machines(machine_id),
        template_id    INTEGER NOT NULL REFERENCES templates(template_id),
        start_ts       TEXT NOT NULL,
        end_ts         TEXT,
        instance_type  TEXT NOT NULL,  -- interruptible | on_demand | reserved
        launch_method  TEXT NOT NULL,  -- console | programmatic
        end_reason     TEXT,           -- user | host_reclaim | machine_offline | setup_failure
        gpu_hours      REAL,
        total_cost     REAL
    );

    CREATE TABLE IF NOT EXISTS deposits (
        deposit_id  INTEGER PRIMARY KEY,
        account_id  INTEGER NOT NULL REFERENCES accounts(account_id),
        ts          TEXT NOT NULL,
        amount      REAL NOT NULL
    );

    -- derived; rebuilt by src/features.py + src/classify.py, never hand-edited
    CREATE TABLE IF NOT EXISTS weekly_profile (
        account_id           INTEGER NOT NULL REFERENCES accounts(account_id),
        week_start           TEXT NOT NULL,
        classified_archetype TEXT,
        rental_count             INTEGER,
        gpu_hours_sum            REAL,
        interruptible_share      REAL,
        on_demand_or_reserved_share REAL,
        median_session_hours     REAL,
        max_concurrent_instances INTEGER,
        verified_host_share      REAL,
        console_launch_share     REAL,
        restart_after_interruption_rate REAL,
        serverless_present       INTEGER,
        dominant_gpu_class       TEXT,
        dominant_template_category TEXT,
        PRIMARY KEY (account_id, week_start)
    );

    CREATE INDEX IF NOT EXISTS idx_rentals_account ON rentals(account_id);
    CREATE INDEX IF NOT EXISTS idx_rentals_machine ON rentals(machine_id);
    CREATE INDEX IF NOT EXISTS idx_deposits_account ON deposits(account_id);
    CREATE INDEX IF NOT EXISTS idx_weekly_profile_account ON weekly_profile(account_id);

    CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
    """,
    # v2: A4's "median session > 168h, OR serverless present" criterion
    # (PRD §4) needs a raw serverless signal to be measurable; additive,
    # existing rows default to 0 rather than losing data.
    """
    ALTER TABLE rentals ADD COLUMN is_serverless INTEGER NOT NULL DEFAULT 0;
    """,
]


def _current_version(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )
    if cur.fetchone() is None:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def migrate(db_path: str = DB_PATH) -> int:
    """Apply any migrations not yet applied. Returns the resulting schema version."""
    # Git doesn't track empty directories, so a fresh clone (Streamlit Cloud,
    # every deploy) has no data/ dir at all — sqlite3.connect() fails outright
    # without this. Caught by a from-scratch clean-clone test at S10; every
    # earlier local test had data/ already on disk from S0's initial mkdir,
    # so this never surfaced until testing the exact deployment path.
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        current = _current_version(conn)
        for version, script in enumerate(MIGRATIONS, start=1):
            if version <= current:
                continue
            conn.executescript(script)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()
        return len(MIGRATIONS)
    finally:
        conn.close()


if __name__ == "__main__":
    final_version = migrate()
    print(f"Schema at version {final_version}: {DB_PATH}")
