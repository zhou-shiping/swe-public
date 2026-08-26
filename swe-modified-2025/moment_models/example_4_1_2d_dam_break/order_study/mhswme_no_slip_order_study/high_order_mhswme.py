#!/usr/bin/env python3
"""Vectorized one-dimensional MHSWME solver for arbitrary retained order.

The homogeneous operator and initial data are identical to the classical
HSWME retained-order study.  The only model change is the MHSWME source:
the stiff wall-traction term is replaced by the effective mean-velocity
source while the Legendre vertical-diffusion terms are unchanged.  Both
models use the same backward-Euler source solver.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
HSWME_STUDY_DIR = HERE.parent / "hswme_no_slip_order_study"
if str(HSWME_STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(HSWME_STUDY_DIR))

from high_order_hswme import (  # noqa: E402
    ImplicitSourceSolver,
    SolverResult,
    initial_state,
    legendre_derivative_inner_products,
    maximum_wavespeed,
    roe_increment,
)


def effective_gamma_bar(
    height: np.ndarray,
    epsilon: float,
    gamma: float,
    inverse_reynolds_0: float,
) -> np.ndarray:
    """Return the effective MHSWME wall-friction coefficient."""
    return gamma / (
        epsilon + height * gamma / (3.0 * inverse_reynolds_0)
    )


def modified_source(
    state: np.ndarray,
    epsilon: float,
    gamma: float,
    inverse_reynolds_0: float,
) -> np.ndarray:
    """Evaluate the general-order MHSWME source in conserved variables."""
    if state.ndim != 2 or state.shape[1] < 3:
        raise ValueError("state must have shape (cells, order + 2), order >= 1")
    height = state[:, 0]
    if np.any(height <= 0.0):
        raise FloatingPointError("MHSWME source received non-positive height")

    order = state.shape[1] - 2
    mean_velocity = state[:, 1] / height
    moments = state[:, 2:] / height[:, None]
    derivative_coupling = (
        moments @ legendre_derivative_inner_products(order).T
    )
    gamma_bar = effective_gamma_bar(
        height, epsilon, gamma, inverse_reynolds_0
    )
    factors = 2.0 * np.arange(1, order + 1, dtype=float) + 1.0

    source = np.zeros_like(state)
    source[:, 1] = -gamma_bar * mean_velocity
    source[:, 2:] = -factors[None, :] * (
        (gamma_bar * mean_velocity)[:, None]
        + inverse_reynolds_0
        * derivative_coupling
        / height[:, None]
    )
    return source


def solve_mhswme(
    *,
    order: int,
    nx: int,
    cfl: float,
    output_times: Iterable[float],
    velocity_scale: float = 100.0,
    gravity_number: float = 9.81 * 1.5 / 100.0**2,
    gamma: float = 1.0e10 / (1000.0 * 100.0),
    epsilon: float = 1.5 / 100.0,
    inverse_reynolds_0: float = (1.0e-6 / (100.0 * 1.5)) / (1.5 / 100.0),
    progress: Callable[[int, float, float, float], None] | None = None,
) -> SolverResult:
    """Advance one no-slip MHSWME order and save requested snapshots."""
    if not 0.0 < cfl <= 1.0:
        raise ValueError("cfl must lie in (0,1]")
    targets = np.asarray(sorted(set(float(t) for t in output_times)), dtype=float)
    if targets.size == 0 or targets[0] <= 0.0:
        raise ValueError("output_times must contain positive values")

    state = initial_state(nx, order, velocity_scale)
    dx = 1.0 / nx
    final_time = float(targets[-1])
    source_solver = ImplicitSourceSolver(order)
    snapshots = [state.copy()]
    saved_times = [0.0]
    dt_history: list[float] = []
    current_time = 0.0
    target_index = 0
    step = 0

    while current_time < final_time:
        max_speed = float(np.max(maximum_wavespeed(state, gravity_number)))
        dt = min(cfl * dx / max_speed, final_time - current_time)
        target_hit = False
        if target_index < len(targets):
            remaining = float(targets[target_index] - current_time)
            if dt >= remaining:
                dt = remaining
                target_hit = True
        if dt <= 0.0:
            raise FloatingPointError("non-positive numerical time step")

        increment = roe_increment(state, gravity_number, dx, dt)
        state[1:-1, 0] += increment[1:-1, 0]
        if np.any(state[1:-1, 0] <= 0.0):
            raise FloatingPointError("non-positive water height after Roe step")
        height = state[1:-1, 0]
        explicit_rhs = state[1:-1, 1:] + increment[1:-1, 1:]
        gamma_bar = effective_gamma_bar(
            height, epsilon, gamma, inverse_reynolds_0
        )
        state[1:-1, 1:] = source_solver.solve(
            explicit_rhs,
            height,
            dt,
            gamma_bar,
            inverse_reynolds_0,
            wall_couples_moments=False,
        )
        state[0] = state[1]
        state[-1] = state[-2]
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("non-finite MHSWME state")

        current_time += dt
        step += 1
        dt_history.append(dt)
        if target_hit:
            current_time = float(targets[target_index])
            snapshots.append(state.copy())
            saved_times.append(current_time)
            target_index += 1
        if progress is not None and (step % 100 == 0 or target_hit):
            progress(step, current_time, dt, max_speed)

    return SolverResult(
        states=np.asarray(snapshots, dtype=float),
        saved_times=np.asarray(saved_times, dtype=float),
        dt_history=np.asarray(dt_history, dtype=float),
    )
