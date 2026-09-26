"""Minimal, inactive virtual-PMM experiment planner.

This module selects dimensionless captive states. It performs no CFD, alters
no production provider, and has no benchmark coefficients as inputs.
"""
from .planner import CaptiveState, initial_states, select_next_state

__all__ = ["CaptiveState", "initial_states", "select_next_state"]
