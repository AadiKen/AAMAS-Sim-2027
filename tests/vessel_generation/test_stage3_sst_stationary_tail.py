"""Regression checks for the read-only stationary-tail search."""

import numpy as np

from tools.analyze_stage3_sst_stationary_tail import (
    earliest_continuous_tail,
    summarize_window,
)


def test_search_rejects_temporary_crossing_before_later_failure() -> None:
    records = [
        {"start_s": 55.0, "end_s": 63.0, "important_channels_accepted": True},
        {"start_s": 55.25, "end_s": 63.25, "important_channels_accepted": False},
        {"start_s": 55.5, "end_s": 63.5, "important_channels_accepted": True},
        {"start_s": 56.0, "end_s": 64.0, "important_channels_accepted": False},
        {"start_s": 56.25, "end_s": 64.25, "important_channels_accepted": True},
        {"start_s": 60.0, "end_s": 80.0, "important_channels_accepted": True},
    ]
    found = earliest_continuous_tail(records, candidate_start_s=55., resolution_s=.25)
    assert found["start_s"] == 56.25
    assert found["later_refailure"] is False


def test_stationary_tail_summary_uses_unchanged_production_gate() -> None:
    time = np.arange(0., 80.0001, .05)
    force = 23.6 + .01 * np.sin(2 * np.pi * time / 2.5)
    heave = 1794. + .1 * np.sin(2 * np.pi * time / 2.5)
    pitch = 480. + .05 * np.sin(2 * np.pi * time / 2.5)
    history = np.column_stack((time, force, np.zeros_like(time), heave,
                               np.zeros_like(time), pitch, np.zeros_like(time)))
    result = summarize_window(history, 80., 20.)
    assert result["important_channels_accepted"]
    assert all(result["channels"][name]["normalized_slope"] < .03
               for name in ("Fx", "Fz", "My"))
