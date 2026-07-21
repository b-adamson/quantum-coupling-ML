"""
Lindblad open-system latching (JAX).

This subpackage intentionally contains **no CTMC backend**.

Public entrypoint:
  - `markov_jumps(model, vg)` from `ground_state.py`

Low-level building blocks live in `markov_jumps.py` (transition graphs, rates,
Lindblad scan integrators).
"""

from .ground_state import markov_jumps

from .markov_jumps import (
    LindbladParams,
    build_transition_graph,
    cached_transition_graph_for_full_basis,
    compute_edge_rates_over_grid_jax,
    psi_to_charge_expectation,
    rho_to_charge_expectation,
    simulate_lindblad_scan_jax,
    simulate_mcwf_scan_jax,
    simulate_eigenjump_scan_jax,
    simulate_eigenjump_scan_jax_fast,
)

__all__ = [
    "LindbladParams",
    "build_transition_graph",
    "cached_transition_graph_for_full_basis",
    "compute_edge_rates_over_grid_jax",
    "markov_jumps",
    "psi_to_charge_expectation",
    "rho_to_charge_expectation",
    "simulate_lindblad_scan_jax",
    "simulate_mcwf_scan_jax",
    "simulate_eigenjump_scan_jax",
    "simulate_eigenjump_scan_jax_fast",
]
