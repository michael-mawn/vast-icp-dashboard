"""
Feature-floor and A0 behavior checks for src/classify.py. Plain assert
script; run with `python -m tests.test_classifier`.
"""

import pandas as pd

from src.classify import classify_row


def _row(**overrides) -> pd.Series:
    base = {
        "rental_count": 10,
        "gpu_hours_sum": 50.0,
        "interruptible_share": 1.0,
        "on_demand_or_reserved_share": 0.0,
        "median_session_hours": 1.0,
        "max_concurrent_instances": 1,
        "verified_host_share": 1.0,
        "console_launch_share": 1.0,
        "restart_after_interruption_rate": 0.0,
        "serverless_present": 0,
        "dominant_gpu_class": "4090",
        "dominant_template_category": "ollama",
    }
    base.update(overrides)
    return pd.Series(base)


def test_below_a0_floor_never_classifies():
    row = _row(rental_count=1, gpu_hours_sum=100.0, verified_host_share=1.0,
               dominant_gpu_class="H100", on_demand_or_reserved_share=1.0)
    assert classify_row(row) == "A0_UNCLASSIFIED", (
        "An account below the A0 rental-count floor classified despite a "
        "100% verified-host share on a 1-rental denominator — exactly the "
        "silent-failure mode the floor exists to prevent."
    )


def test_below_gpu_hours_floor_never_classifies():
    row = _row(rental_count=5, gpu_hours_sum=1.0)
    assert classify_row(row) == "A0_UNCLASSIFIED"


def test_clean_a1_profile_classifies_a1():
    row = _row(dominant_gpu_class="4090", interruptible_share=0.9, median_session_hours=1.5,
               max_concurrent_instances=1, console_launch_share=0.9,
               dominant_template_category="ollama", rental_count=8, gpu_hours_sum=12.0)
    assert classify_row(row) == "A1_TINKERER"


def test_clean_a4_profile_classifies_a4():
    row = _row(dominant_gpu_class="H100", on_demand_or_reserved_share=0.95, verified_host_share=0.95,
               median_session_hours=200.0, max_concurrent_instances=8, console_launch_share=0.05,
               dominant_template_category="vllm_serving", interruptible_share=0.05,
               rental_count=10, gpu_hours_sum=2000.0)
    assert classify_row(row) == "A4_PRODUCTION"


def test_low_denominator_ratios_dont_drive_classification():
    """The specific failure the two-floor split fixes: 3 rentals clears the
    A0_MIN_RENTALS floor (2), but sits below MIN_RENTALS_FOR_RATIO_TRUST
    (5) — a consumer-GPU, short-session account that happens to land 100%
    verified/on-demand on just 3 rentals must NOT classify as A4 purely
    from those ratios. Every A4-defining non-ratio signal (GPU class,
    session length) is deliberately absent here."""
    row = _row(
        rental_count=3, gpu_hours_sum=6.0,
        dominant_gpu_class="4090", dominant_template_category="ollama",
        median_session_hours=1.5, max_concurrent_instances=1,
        on_demand_or_reserved_share=1.0, verified_host_share=1.0,  # A4-shaped ratios, by chance
        interruptible_share=0.0, console_launch_share=0.0,
    )
    result = classify_row(row)
    assert result != "A4_PRODUCTION", (
        f"A 3-rental account with A4-shaped ratios but consumer GPU/short "
        f"sessions classified as {result} — ratio features are driving the "
        "outcome below the trust floor, exactly the failure being fixed."
    )


if __name__ == "__main__":
    test_below_a0_floor_never_classifies()
    test_below_gpu_hours_floor_never_classifies()
    test_clean_a1_profile_classifies_a1()
    test_clean_a4_profile_classifies_a4()
    test_low_denominator_ratios_dont_drive_classification()
    print("PASS: A0 floor, ratio-trust floor, and clean-profile classification checks.")
