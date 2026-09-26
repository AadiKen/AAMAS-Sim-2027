"""Experimental Spec A maneuvering coefficient pipeline."""

from .fit import CoefficientSurface, fit_cases
from .matrix import CaseState, froude_gate, case_matrix

__all__ = ["CoefficientSurface", "fit_cases", "CaseState", "froude_gate", "case_matrix"]
