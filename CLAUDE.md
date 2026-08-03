# CLAUDE.md

## What this project is

A Streamlit dashboard that segments a GPU marketplace's customer base by inferred workload archetype and measures whether accounts migrate between archetypes over time. Runs on synthetic data.

Full specification is in `PRD.md`. Read it at the start of every session.

**Context that matters:** this is a portfolio artifact for a growth leader at Vast.ai, built in a single day. The repo is public and a technical reader will open it. It is explicitly **not** a findings document — it demonstrates the questions worth asking and a system that answers them once pointed at real data.

## How to work with me

I work in the Claude Code desktop app and I am **not comfortable in the terminal**. When any step requires me to do something myself — creating a GitHub repo, authorizing a service, clicking through a deployment UI — explain it in plain language, one step at a time, and tell me exactly where to click. Do not assume I know git, shell commands, or deployment tooling.

Run commands yourself wherever you can rather than handing me a command to paste.

## Stack

- Python 3.12, SQLite, pandas, Streamlit, Plotly
- Streamlit Community Cloud (~1 GB memory ceiling — design for it)
- Entry point: `streamlit_app.py`
- Run locally: `streamlit run streamlit_app.py`
- Regenerate data: `python -m src.generate`

## Time constraint

This is a one-day build. Prefer the working version over the elegant one. Flag anything that looks like it will take more than about 45 minutes so I can decide whether to cut it.

Follow the slice order in §12 of `PRD.md`. **Deploy a hello-world app (S1) before writing any analysis code** — GitHub repo creation and the Streamlit Community Cloud connection both happen there, while nothing is at stake.

## Working agreement

**Plan before building.** For anything larger than a single-file edit, show the plan and wait.

**Ask, don't assume.** Use the AskUserQuestion tool whenever the PRD is silent or a decision is ambiguous. Batch related questions together. Given the time constraint, ask high-value questions in one round rather than trickling them out.

**Never invent a threshold.** All archetype thresholds and generator parameters come from `PRD.md` or from asking me. If you find yourself choosing a number, stop and ask.

**Say when something is wrong.** If a spec decision is bad, inconsistent, or will produce misleading output, say so before implementing it. Silent compliance with a bad instruction is the failure mode I care most about.

**Flag silent failures.** Call out anywhere bad input could produce plausible-looking output with no error raised.

**Distinguish strategy from implementation.** Some questions look technical but change what the dashboard claims (for example, whether unclassified accounts are excluded from the migration matrix). Flag those as judgment calls rather than deciding them.

## Non-negotiable conventions

- All tunable parameters live in `config/`. No magic numbers anywhere else.
- Ground-truth archetype labels live in their own table and are **never** read by the classifier at inference time. The classifier must rediscover them independently.
- Every view carries a one-line header stating the question it answers.
- The synthetic-data disclosure is visible without scrolling on every view.
- **No UI copy asserts a finding.** The dashboard shows mechanisms and distributions. It never says which segment is better, or what the data "shows." Headlines are questions, not answers.
- Data sourced from Vast's real public API is labeled distinctly from synthetic data.
- Schema changes use migrations that preserve existing rows. Never drop and regenerate the database to accommodate a schema change.
- Commit after each working slice.

## Things to ask me about rather than decide

- Any archetype threshold not written in `PRD.md`
- Chart type for any new view
- Anything that changes the schema
- Anything that adds a dependency
- Any UI copy that characterizes what the data shows
