"""
Classifier thresholds and scoring weights. Transcribed from PRD.md §4.

FIREWALL: this module must never be imported by config/generator.py or by
anything that touches the ground_truth table. src/classify.py is the only
consumer. This boundary is what makes the classifier's independence a
checked fact rather than a promised convention — see src/tests for the
import-boundary test.

Scoring (decided in spec review, finalized at S4): each archetype accumulates
one WEIGHT per satisfied criterion, then that sum is NORMALIZED by the
archetype's own max-possible weight sum (a 0-1 "fit fraction") before
comparing across archetypes. Raw sums would structurally favor archetypes
with more criteria (A4 has 8, A1 has 7-ish counting the split template
weight, A2/A3 have 6) regardless of true fit — normalizing removes that
bias. Highest fit fraction wins. An account below MIN_FIT_FRACTION, or tied
at the top between two archetypes, is reported as A0.

MIN_FIT_FRACTION=0.5 (must satisfy at least half an archetype's weighted
criteria) is a provisional value — user was unavailable to confirm at S4;
logged in the build plan for review. Weights are uniform (1.0) except
A3's max-concurrency criterion (1.5x, PRD §4 calls it out as the single
defining trait), which was already approved.

Trajectory criteria from PRD §4 ("growing deposits", "trending up") are
deliberately EXCLUDED here — unmeasurable within a 30-day window without a
slope threshold the PRD doesn't specify, and inclusion would make migration
into A4 partly definitional. They remain generator flavor only
(config/generator.py), not classifier inputs.
"""

from config.settings import A0_MIN_RENTALS, A0_MIN_GPU_HOURS

CONSUMER_GPUS = {"3090", "4090", "5090"}
DATACENTER_GPUS = {"A100", "H100", "B200"}

# Minimum fit fraction (0-1, own-archetype-normalized) to classify at all.
# Below this -> A0. Provisional (user unavailable at S4) — see docstring.
MIN_FIT_FRACTION = 0.5

# Default criterion weight. Uniform until S4 review.
DEFAULT_WEIGHT = 1.0

# A0 floor: minimum activity to be classified at all (vs A0).
A0_FLOOR = {
    "min_rentals": A0_MIN_RENTALS,
    "min_gpu_hours": A0_MIN_GPU_HOURS,
}

# Separate, higher floor before any RATIO feature is trusted as evidence.
# Caught post-S10: A0_MIN_RENTALS was lowered to 2 (see settings.py) so a
# single/double-session Researcher could still classify off non-ratio
# signals. But at exactly 2 rentals, interruptible_share (etc.) can only be
# 0%, 50%, or 100% — reusing the SAME floor for "classify at all" and "trust
# this ratio" meant a 2-rental account landing on two verified hosts read as
# a clean 100% verified-host signal, exactly the false-confidence failure
# the floor exists to prevent. Below this threshold, ratio criteria are
# excluded from scoring entirely (see src/classify.py) rather than counted
# at low-denominator, high-noise values — classification then rests on
# gpu_class, session length, concurrency, and template alone.
MIN_RENTALS_FOR_RATIO_TRUST = 5
RATIO_FEATURES = {
    "interruptible_share",
    "on_demand_or_reserved_share",
    "verified_host_share",
    "console_launch_share",
    "restart_after_interruption_rate",
}

# Each criterion: (feature_name, comparator, threshold_value, weight)
# comparator is one of: "gt", "gte", "lt", "lte", "in", "between"
# "base_pytorch_jupyter" templates are listed under both A1 and A2 in the PRD
# and are treated as a weak neutral signal — included at reduced weight on
# both, not as a discriminator between them.

ARCHETYPES = {
    "A1_TINKERER": {
        "label": "Tinkerer",
        "criteria": [
            ("gpu_class", "in", CONSUMER_GPUS, DEFAULT_WEIGHT),
            ("interruptible_share", "gt", 0.70, DEFAULT_WEIGHT),
            ("median_session_hours", "lt", 4, DEFAULT_WEIGHT),
            ("max_concurrent_instances", "eq", 1, DEFAULT_WEIGHT),
            ("console_launch_share", "gt", 0.5, DEFAULT_WEIGHT),  # depends on launch_method — see NOTE below
            ("template_category", "in",
             {"ollama", "text_generation_webui", "stable_diffusion_webui"}, DEFAULT_WEIGHT),
            ("template_category", "eq", "base_pytorch_jupyter", DEFAULT_WEIGHT * 0.5),
        ],
    },
    "A2_RESEARCHER": {
        "label": "Researcher / Fine-tuner",
        "criteria": [
            ("gpu_class", "in", {"A100", "H100", "4090"}, DEFAULT_WEIGHT),
            ("interruptible_share", "between", (0.30, 0.70), DEFAULT_WEIGHT),
            ("median_session_hours", "between", (6, 72), DEFAULT_WEIGHT),
            ("max_concurrent_instances", "between", (1, 4), DEFAULT_WEIGHT),
            ("template_category", "in",
             {"axolotl", "torchtune", "llama_factory", "unsloth"}, DEFAULT_WEIGHT),
            ("template_category", "eq", "base_pytorch_jupyter", DEFAULT_WEIGHT * 0.5),
        ],
    },
    "A3_BATCH": {
        "label": "Batch / async processing",
        "criteria": [
            ("gpu_class", "in", CONSUMER_GPUS, DEFAULT_WEIGHT),
            ("interruptible_share", "gt", 0.80, DEFAULT_WEIGHT),
            ("max_concurrent_instances", "between", (5, 50), DEFAULT_WEIGHT * 1.5),  # defining trait, PRD §4
            ("console_launch_share", "lt", 0.5, DEFAULT_WEIGHT),  # depends on launch_method — see NOTE below
            ("restart_after_interruption_rate", "gt", 0.80, DEFAULT_WEIGHT),
            ("template_category", "in",
             {"whisper", "offline_vllm_batch", "embedding_pipeline", "render_job"}, DEFAULT_WEIGHT),
        ],
    },
    "A4_PRODUCTION": {
        "label": "Production inference",
        "criteria": [
            ("gpu_class", "in", DATACENTER_GPUS, DEFAULT_WEIGHT),
            ("on_demand_or_reserved_share", "gt", 0.80, DEFAULT_WEIGHT),
            ("verified_host_share", "gt", 0.90, DEFAULT_WEIGHT),
            ("median_session_hours", "gt", 168, DEFAULT_WEIGHT),  # OR serverless_present, PRD §4
            ("serverless_present", "eq", True, DEFAULT_WEIGHT),
            ("max_concurrent_instances", "between", (2, 20), DEFAULT_WEIGHT),
            ("console_launch_share", "lt", 0.5, DEFAULT_WEIGHT),  # depends on launch_method — see NOTE below
            ("template_category", "in",
             {"vllm_serving", "tgi", "sglang", "tensorrt_llm"}, DEFAULT_WEIGHT),
        ],
    },
}

# NOTE (PRD §7, surfaced in UI per build-plan decision): launch_method
# distinguishes console vs programmatic calls to the same REST API. Whether
# it is logged and distinguishable server-side "requires confirmation from
# Vast." A3 and A4 both depend on it here. Any view showing A3 or A4 must
# name this dependency rather than let the reader assume the field is solid.
LAUNCH_METHOD_DEPENDENT_ARCHETYPES = ("A3_BATCH", "A4_PRODUCTION")
