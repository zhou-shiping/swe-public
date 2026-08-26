#!/usr/bin/env python3
"""Vectorized one-dimensional HSWME solver for arbitrary retained order.

The implementation follows the original no-slip HSWME discretization used by
``src/moment_1d/SWE_1D/MSWME.py``:

* arithmetic Roe states and the same local Lax--Friedrichs dissipation;
* first-order explicit treatment of the homogeneous HSWME operator; and
* backward Euler for the stiff wall-friction and vertical-viscosity source.

Only the classical (unmodified) HSWME is implemented here.  The general-order
formulas reduce exactly to the existing N=1, 2, and 3 formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class SolverResult:
    """Snapshots and time-step history returned by :func:`solve_hswme`."""

    states: np.ndarray
    saved_times: np.ndarray
    dt_history: np.ndarray


def legendre_derivative_inner_products(order: int) -> np.ndarray:
    """Return ``C_ij = int_0^1 phi_i' phi_j' dzeta`` for ``1 <= i,j <= N``.

    For ``phi_i(zeta)=P_i(1-2*zeta)``, the closed form is

    ``C_ij = 2*m*(m+1)`` when ``i+j`` is even, and zero otherwise, where
    ``m=min(i,j)``.
    """
    if order < 1:
        return np.empty((0, 0), dtype=float)
    indices = np.arange(1, order + 1, dtype=int)
    i, j = np.meshgrid(indices, indices, indexing="ij")
    m = np.minimum(i, j).astype(float)
    return np.where((i + j) % 2 == 0, 2.0 * m * (m + 1.0), 0.0)


def initial_state(
    nx: int,
    order: int,
    velocity_scale: float,
    left_height: float = 1.0,
    right_height: float = 2.0 / 3.0,
    amplitude_m_per_s: float = 0.25,
) -> np.ndarray:
    """Initialize the quadratic no-slip dam-break profile."""
    if nx < 10:
        raise ValueError("nx must be at least 10")
    if order < 1:
        raise ValueError("order must be positive")
    state = np.zeros((nx, order + 2), dtype=float)
    state[:, 0] = right_height
    state[: nx // 2, 0] = left_height
    moment_values = np.zeros(order, dtype=float)
    moment_values[0] = -amplitude_m_per_s / (2.0 * velocity_scale)
    if order >= 2:
        moment_values[1] = -amplitude_m_per_s / (6.0 * velocity_scale)
    mean_velocity = 2.0 * amplitude_m_per_s / (3.0 * velocity_scale)
    state[:, 1] = state[:, 0] * mean_velocity
    state[:, 2:] = state[:, [0]] * moment_values
    return state


def maximum_wavespeed(state: np.ndarray, gravity_number: float) -> np.ndarray:
    """Return the HSWME wavespeed used by the existing Roe scheme."""
    height = state[..., 0]
    if np.any(height <= 0.0):
        raise FloatingPointError("HSWME produced non-positive water height")
    mean_velocity = state[..., 1] / height
    alpha_1 = state[..., 2] / height
    return np.abs(mean_velocity) + np.sqrt(
        gravity_number * height + alpha_1**2
    )


def apply_hswme_jacobian(
    roe_state: np.ndarray, jump: np.ndarray, gravity_number: float
) -> np.ndarray:
    """Apply the general-order one-dimensional HSWME Jacobian to cell jumps."""
    if roe_state.shape != jump.shape or roe_state.ndim != 2:
        raise ValueError("roe_state and jump must be two-dimensional and equal-sized")
    order = roe_state.shape[1] - 2
    if order < 1:
        raise ValueError("at least one moment is required")

    height = roe_state[:, 0]
    mean_velocity = roe_state[:, 1] / height
    alpha_1 = roe_state[:, 2] / height
    product = np.zeros_like(jump)

    product[:, 0] = jump[:, 1]
    product[:, 1] = (
        (-mean_velocity**2 - alpha_1**2 / 3.0 + gravity_number * height)
        * jump[:, 0]
        + 2.0 * mean_velocity * jump[:, 1]
        + (2.0 / 3.0) * alpha_1 * jump[:, 2]
    )
    product[:, 2] = (
        -2.0 * mean_velocity * alpha_1 * jump[:, 0]
        + 2.0 * alpha_1 * jump[:, 1]
        + mean_velocity * jump[:, 2]
    )
    if order >= 2:
        product[:, 2] += (3.0 / 5.0) * alpha_1 * jump[:, 3]
        product[:, 3] = (
            -(2.0 / 3.0) * alpha_1**2 * jump[:, 0]
            + (1.0 / 3.0) * alpha_1 * jump[:, 2]
            + mean_velocity * jump[:, 3]
        )
    if order >= 3:
        product[:, 3] += (4.0 / 7.0) * alpha_1 * jump[:, 4]
        for moment in range(3, order + 1):
            column = moment + 1
            lower = (moment - 1.0) / (2.0 * moment - 1.0)
            product[:, column] = (
                lower * alpha_1 * jump[:, column - 1]
                + mean_velocity * jump[:, column]
            )
            if moment < order:
                upper = (moment + 2.0) / (2.0 * moment + 3.0)
                product[:, column] += (
                    upper * alpha_1 * jump[:, column + 1]
                )
    return product


def roe_increment(
    state: np.ndarray, gravity_number: float, dx: float, dt: float
) -> np.ndarray:
    """Return the vectorized increment of the existing first-order Roe scheme."""
    jump = state[1:] - state[:-1]
    roe_state = 0.5 * (state[1:] + state[:-1])
    jacobian_jump = apply_hswme_jacobian(roe_state, jump, gravity_number)
    wavespeed = maximum_wavespeed(roe_state, gravity_number)
    increment = np.zeros_like(state)
    coefficient = -dt / (2.0 * dx)
    increment[1:-1] = coefficient * (
        jacobian_jump[:-1]
        + jacobian_jump[1:]
        + wavespeed[:-1, None] * jump[:-1]
        - wavespeed[1:, None] * jump[1:]
    )
    return increment


class ImplicitSourceSolver:
    """Fast backward-Euler solve for the general-order HSWME/MHSWME source."""

    def __init__(self, order: int) -> None:
        if order < 1:
            raise ValueError("order must be positive")
        self.order = order
        self.factors = 2.0 * np.arange(1, order + 1, dtype=float) + 1.0
        self.sqrt_factors = np.sqrt(self.factors)
        c_matrix = legendre_derivative_inner_products(order)
        symmetric = (
            self.sqrt_factors[:, None]
            * c_matrix
            * self.sqrt_factors[None, :]
        )
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(symmetric)

    def _solve_viscous_block(
        self, right_hand_side: np.ndarray, b_values: np.ndarray
    ) -> np.ndarray:
        scaled = right_hand_side / self.sqrt_factors
        modal = scaled @ self.eigenvectors
        modal /= 1.0 + b_values[:, None] * self.eigenvalues[None, :]
        return (modal @ self.eigenvectors.T) * self.sqrt_factors

    def solve(
        self,
        right_hand_side: np.ndarray,
        height: np.ndarray,
        dt: float,
        friction: float | np.ndarray,
        inverse_reynolds_0: float,
        wall_couples_moments: bool = True,
    ) -> np.ndarray:
        """Solve the coupled wall-friction/viscosity update in every wet cell."""
        if right_hand_side.shape[1] != self.order + 1:
            raise ValueError("unexpected source right-hand-side shape")
        a_values = dt * friction / height
        b_values = dt * inverse_reynolds_0 / height**2

        base_solution = np.empty_like(right_hand_side)
        base_solution[:, 0] = right_hand_side[:, 0]
        base_solution[:, 1:] = self._solve_viscous_block(
            right_hand_side[:, 1:], b_values
        )

        source_direction = np.empty_like(right_hand_side)
        source_direction[:, 0] = 1.0
        repeated_factors = np.broadcast_to(
            self.factors, (len(height), self.order)
        )
        source_direction[:, 1:] = self._solve_viscous_block(
            repeated_factors, b_values
        )

        if wall_couples_moments:
            coupled_base = np.sum(base_solution, axis=1)
            coupled_direction = np.sum(source_direction, axis=1)
        else:
            coupled_base = base_solution[:, 0]
            coupled_direction = source_direction[:, 0]
        denominator = 1.0 + a_values * coupled_direction
        correction = a_values * coupled_base / denominator
        return base_solution - correction[:, None] * source_direction


def solve_hswme(
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
    """Advance one no-slip HSWME order and save the requested snapshots."""
    if not 0.0 < cfl <= 1.0:
        raise ValueError("cfl must lie in (0,1]")
    targets = np.asarray(sorted(set(float(t) for t in output_times)), dtype=float)
    if targets.size == 0 or targets[0] <= 0.0:
        raise ValueError("output_times must contain positive values")

    state = initial_state(nx, order, velocity_scale)
    dx = 1.0 / nx
    final_time = float(targets[-1])
    friction = gamma / epsilon
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
        explicit_rhs = state[1:-1, 1:] + increment[1:-1, 1:]
        state[1:-1, 1:] = source_solver.solve(
            explicit_rhs,
            state[1:-1, 0],
            dt,
            friction,
            inverse_reynolds_0,
        )
        state[0] = state[1]
        state[-1] = state[-2]
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("non-finite HSWME state")

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
