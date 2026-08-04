# PRD: Vast.ai Workload Archetype & Migration Dashboard

**Author:** Michael Mawn
**Status:** v2, scoped for a one-day build
**Last updated:** 2026-08-03

---

## 1. What this is

A dashboard that segments a GPU marketplace's customer base by **inferred workload archetype** and measures whether accounts move between archetypes over time.

It runs on synthetic data. It is not a findings document and does not assert any conclusion about Vast's business. It is a working instrument built to demonstrate two things: the set of questions worth asking about a marketplace customer base, and a system that answers them the moment it is pointed at real data.

**Every design decision in this document serves that framing.** Where a choice would make the dashboard argue a position, the choice goes the other way.

---

## 2. The problem it addresses

Vast.ai has grown revenue roughly 4x in a year with no formal RevOps function. Signups are growing faster still. When the top of the funnel fills that fast, the middle goes unexamined, because nothing looks broken.

Meanwhile the product surface has moved upmarket: Serverless for zero-ops inference, Clusters for multi-node training, SOC 2. That direction rests on an assumption: that the large population of low-cost experimental users contains a segment that graduates into durable production spend.

The dashboard exists to make that assumption measurable rather than assumed.

### Questions it is built to answer

Not conclusions. Questions.

1. Who are we actually serving, in terms of workload rather than firmographics?
2. How far apart are our account distribution and our revenue distribution?
3. Do accounts migrate from experimental usage into production usage, at what rate, and in what direction?
4. Which migrations happen often enough to be worth designing for?
5. What would have to be true for the answer to flip?

---

## 3. Framing rules (non-negotiable)

- The underlying customer data is synthetic. This is stated in the UI without scrolling, on every view.
- The dashboard displays **mechanisms and distributions, never verdicts.** No UI copy asserts a finding. No headline says which segment is better.
- Every view carries a one-line header naming the question it answers. The dashboard must be readable cold, without narration, because the recipient will open the link again later without the walkthrough.
- Every generated number traces to a visible, adjustable assumption or to Vast's real public API. Nothing else.
- Where data availability is uncertain, the UI says so rather than papering over it.

---

## 4. Core concept: archetypes

An archetype is a behavioral category describing what kind of work an account is doing, inferred entirely from observable platform behavior. It is a proxy for use case. It exists because industry, company size, and funding stage are not observable from marketplace data, while workload shape is.

### Signals used

| Signal | What it indicates |
|---|---|
| Docker image / template launched | Strongest use-case fingerprint |
| Interruptible vs on-demand vs reserved | Tolerance for interruption |
| Session duration distribution | Experiment vs sustained operation |
| Max concurrent instances | Single-workstation use vs pipeline fan-out |
| GPU class mix | Consumer vs datacenter tier |
| Verified-host share of hours | Willingness to pay for reliability |
| Launch method (console vs programmatic) | Human-driven vs system-driven (see §7) |
| Restart-after-interruption rate | Fault-tolerant workload vs churn |
| Time-of-day / day-of-week pattern | Hobby vs operations |
| Deposit size and cadence | Commitment level |

### The four archetypes

Four, not five. A migration matrix with five archetypes plus churn is thirty cells; at 3,000 accounts most cells hold single digits and read as noise. Four plus churn is twenty cells and is legible.

Thresholds below are the starting spec. All live in `config/archetypes.py`. The classifier does not invent its own.

**A1. Tinkerer**
- Consumer GPUs (3090 / 4090 / 5090)
- Interruptible share of hours > 70%
- Median session < 4h
- Max concurrent instances = 1
- Console-dominant launches
- Templates: Ollama, text-generation-webui, Stable Diffusion WebUI, base PyTorch/Jupyter
- Small, infrequent deposits; evening and weekend skew

**A2. Researcher / Fine-tuner**
- A100, H100, sometimes 4090
- Interruptible share 30-70%
- Median session 6-72h
- Max concurrent 1-4
- Templates: Axolotl, torchtune, LLaMA-Factory, Unsloth, base PyTorch
- Lumpy, episodic deposits; multi-day runs separated by idle weeks

**A3. Batch / async processing**
- Cost-optimal GPU mix, consumer-heavy
- Interruptible share > 80%
- Max concurrent 5-50 (**parallel fan-out is the defining trait**)
- Programmatic launches dominant
- **Restart-after-interruption rate > 80%** — they restart rather than churn
- Templates: Whisper / faster-whisper, offline vLLM batch, embedding pipelines, render jobs
- Regular and growing deposits; bursty but repeating

**A4. Production inference**
- A100, H100, B200
- On-demand or reserved share > 80%
- Verified-host share of hours > 90%
- Median session > 168h, or serverless endpoints present
- Max concurrent 2-20, trending up
- Programmatic launches, serverless
- Templates: vLLM serving, TGI, SGLang, TensorRT-LLM
- Large recurring deposits, auto-reload; flat 24/7 utilization

**A0. Unclassified** — insufficient activity. Reported explicitly, never silently dropped.

**Deferred: Creative pipeline** (ComfyUI, video generation, Blender). Cut from v1 because much of that workload is batch generation and overlaps A3 without adding a distinct question. Named here rather than omitted, because the boundary is a judgment call worth surfacing.

---

## 5. Migration

Archetype is computed from a **trailing 30-day behavior window**, recomputed weekly.

An account has **migrated** when its classification changes and the new classification **holds for two or more consecutive weeks**. Single-week flips are noise and are not counted.

The migration matrix is archetype at day 30 versus archetype at day 180, with churn as a terminal column. **Resolved during the build** (this paragraph originally read as a second, competing definition of migration — a snapshot comparison with no role for the two-week hold rule above): each endpoint of the day-30/day-180 comparison uses the classification that has *held* for two or more consecutive weekly recomputations as of that date, not that week's raw label. This makes the hold rule and the snapshot comparison one mechanism instead of two. It under-detects relative to a full event log — an account that goes A1→A3→A1 within the window reads as no migration — which is the accepted tradeoff for a legible two-point comparison. See `src/migrate_analysis.py`.

**Note on interpretation:** the matrix answers "what workload types move toward production usage." It does not answer "what industry moves toward production usage." Industry is not observable from marketplace data. This limitation is stated in the UI, not hidden.

**Note on what this matrix demonstrates:** the migration rate itself is a generator input (`config/generator.py`), not something measured from real behavior — this dataset is synthetic. The matrix demonstrates the *mechanism* that would measure real migration once pointed at real data: cohort eligibility, the hold rule, churn as a terminal state. It is not a finding about how often accounts actually graduate. Stated in the UI (V2), not hidden.

---

## 6. Views (v1 scope)

**V1. Base composition** — *Who are we serving, by workload?*
Accounts by archetype alongside **lifetime rental spend** by archetype (labeled that way, not "revenue" — unambiguous that it's marketplace volume, not Vast's take). The gap between the two distributions is the thing to look at, but no UI copy says which gap is good or bad.

**V2. Migration matrix** — *Do accounts move from experimental to production usage?*
Day 30 archetype vs day 180 archetype. Heatmap, churn as terminal state.

**V3. Classifier accuracy — CUT.** *Is the segmentation trustworthy?*
Originally scoped as a confusion matrix of classified archetype against ground-truth label. Cut during the build for a one-day, single-reader scope (see build plan). The generator still creates realistic overlap between archetypes so classification is a real exercise, not a formality — this dataset just never shows how well the classifier recovers the labels it was built from. On real data, that check would need to be rebuilt before trusting the classifier's output. Moved to the deferred list below rather than silently dropped.

**V4. Assumptions** — *What would have to be true for this to look different?*
Scope cut from "every generator parameter" to three: archetype mix at signup, migration dynamics, and the churn window (see build plan). Each carries the plain-language description and real-world evidence question this section originally specified.

### Deferred, and named as deferred in the UI

- Expansion / net revenue retention by cohort
- Early-signature analysis (what the first 14 days of an eventual A3 or A4 account look like)
- Reliability-tax analysis (retention and spend delta following a host-side failure)
- Classifier accuracy vs. ground truth (originally scoped as V3 above)

These are listed in-product as "next," not omitted silently.

---

## 7. Data availability: confirmed vs assumed

The dashboard depends on the fields below. Honesty about which are verified matters more than pretending completeness.

**Confirmed from Vast's public API documentation:**
machine_id, Docker image, template id / hash / name, instance start and end timestamps, duration, instance state, runtype, GPU model and count, host reliability score, verified status, geolocation, offer pricing. Documented endpoints exist for billing, accounts, and serverless.

**Assumed, likely available, unverified:**
Granular end-reason for a rental (user-destroyed vs outbid vs host reclaim vs container setup failure). Deposit history and auto-reload flag.

**Requires confirmation from Vast:**
Whether launch method is distinguishable server-side. The console, CLI, and SDK all call the same REST API, so separating them requires a logged client identifier. If it is not logged, weaker proxies exist (SSH key attached at creation, tightly sequential create calls, template reuse), but they are inference rather than a field.

**This list is a feature, not a caveat.** It is the first set of questions to ask on day one with real data access, and it is surfaced in the UI.

---

## 8. Synthetic data generator

Ground-truth archetype labels are assigned **first**. Behavior is generated **from** those labels, with deliberate overlap at the archetype boundaries so classification is a real exercise rather than reading back stamped values. The classifier then **rediscovers** them independently. (V3, which would have reported that accuracy, is cut — see §6.)

Ground-truth labels live in their own table and are never read by the classifier at inference time. This structure is not optional — it is what makes the classifier's quality measurable rather than assumed.

**Scale:** ~3,000 accounts, 12 months of simulated activity, signup rate accelerating across the period.

**Tunable parameters, all exposed in V4:**
- Archetype mix at signup
- Migration probability per archetype pair
- Host-side failure rate (verified vs unverified)
- Retention penalty following a failure
- Signup growth rate
- Deposit size distribution per archetype

All parameters live in one config module. No magic numbers elsewhere.

**Magnitude discipline:** generated values must stay inside the envelope implied by Vast's real public numbers (17,000+ GPUs, 1,400+ providers, verified H100 SXM around $2.89/GPU-hour). A number that is obviously off will cost credibility on a detail unrelated to the analysis.

**Stretch goal:** seed the marketplace substrate (GPU classes, price distributions, reliability distributions, verified split, geography) from a one-time broad snapshot of Vast's public API across all GPU classes. Customer behavior stays synthetic; the market they operate in becomes real. Cut this before cutting anything else if time runs short.

---

## 9. Schema

```
accounts        account_id, signup_date, ground_truth_archetype,
                first_deposit_amount, auto_reload_enabled, churn_date

machines        machine_id, gpu_model, gpu_count, verified, reliability_score,
                region, base_price_per_hour

rentals         rental_id, account_id, machine_id, template_id, start_ts, end_ts,
                instance_type (interruptible|on_demand|reserved),
                launch_method (console|programmatic),
                end_reason (user|host_reclaim|machine_offline|setup_failure),
                gpu_hours, total_cost

templates       template_id, name, category, implied_workload

deposits        deposit_id, account_id, ts, amount

weekly_profile  account_id, week_start, classified_archetype, [features]
                [derived; rebuilt by the classifier]
```

---

## 10. Out of scope

Real customer data. Channel attribution. Supply-side recruitment. ML models (the classifier is rule-based and readable on purpose). Authentication. Anything requiring data Vast does not already collect.

---

## 11. Stack and deployment

Python 3.12, SQLite, pandas, Streamlit, Plotly. Deployed to Streamlit Community Cloud from a **public** GitHub repo at a custom subdomain.

Constraints to design within: ~1 GB memory, apps sleep after 12 quiet hours. Slider-triggered regeneration must complete in a few seconds at 3,000 accounts — precompute and cache aggressively.

The repo being public is intentional. `PRD.md` in the repo is part of the deliverable.

---

## 12. Build sequence (one day)

**Do S0 and S1 before writing any analysis code.** Deployment failures at the end of a build day are how these projects die.

- **S0** — Repo scaffold, config module, schema, `requirements.txt`
- **S1** — Hello-world Streamlit app deployed to Community Cloud. Working public URL before anything else exists.
- **S2** — Generator for A1 only, plus one chart, end to end
- **S3** — All four archetypes with ground-truth labels
- **S4** — Rule-based classifier
- **S5** — V1 base composition
- **S6** — V2 migration matrix
- **S7** — V3 classifier accuracy
- **S8** — V4 sliders with live regeneration
- **S9** — Copy pass: question headers, synthetic-data disclosure, deferred-views note, data-availability note
- **S10** — Final deploy and cold-read test
- **S11 (stretch)** — Real API seeding of the marketplace layer

---

## 13. Spec review (time-boxed)

Before S0, one focused round of questioning, capped at 30 minutes. Highest-value questions only, asked in a single batch. Priority areas:

- Overlaps or ambiguities among the four archetype threshold sets
- Whether generate-from-labels-then-classify-back is circular in a way that undermines V3's credibility
- Whether the two-week-hold migration definition under- or over-detects
- What breaks first at 3,000 accounts on a 1 GB instance
- The single weakest assumption in this document
