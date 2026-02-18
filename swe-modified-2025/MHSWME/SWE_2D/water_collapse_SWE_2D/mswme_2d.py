"""
Modified Hyperbolic Shallow Water Moment Expansion (MHSWME) Solver — 2D

Authors
-------
Shiping Zhou <zhouship@msu.edu>
    Michigan State University, CMSE, USA

License
-------
MIT License. See LICENSE file for details.

Date
----
2026-02-17

This module implements a 2D shallow water moment expansion system solver
using the Roe scheme for spatial discretization and explicit time integration.

The system of equations solved:
    h_t + (hu_m)_x + (hv_m)_y = 0
    (hu_m)_t + (hu_m^2 + 1/3*h*alpha_1^2 + G/2*h^2)_x = -a0*(u_m + sum alpha_j)
    (hv_m)_t + (hv_m^2 + 1/3*h*beta_1^2  + G/2*h^2)_y = -a0*(v_m + sum beta_j)
    (h*alpha_1)_t + (2*h*u_m*alpha_1)_x = -3*a0*(u_m + (1 + 4*R_0/h/epsilon/a0)*alpha_1)
    (h*beta_1)_t  + (2*h*v_m*beta_1)_y  = -3*a0*(v_m + (1 + 4*R_0/h/epsilon/a0)*beta_1)

State vector:
    U = [h, hu_m, hv_m, h*alpha_1, h*beta_1, h*alpha_2, h*beta_2, ...]^T

where:
    h         : water height
    u_m, v_m  : mean velocities in x and y
    alpha_i   : x-direction velocity moments
    beta_i    : y-direction velocity moments
    G         : gravitational parameter
    a0        : friction coefficient
    R_0       : viscosity parameter
    epsilon   : aspect ratio H1/L1
"""

import os
import numpy as np


def initialize_state(nx, ny, dx, dy, h_center, h_rest, var_num, u_scale, h_scale,
                     init_velocity):
    """
    Initialize the state vector for the 2D circular dam break problem.

    Parameters
    ----------
    nx : int
        Number of computational cells in x
    ny : int
        Number of computational cells in y
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
    h_center : float
        Initial water height inside the dam (dimensionless)
    h_rest : float
        Initial water height outside the dam (dimensionless)
    var_num : int
        Number of variables:
        3: SWE       (h, hu, hv)
        5: SWME1     (h, hu, hv, h*alpha_1, h*beta_1)
        7: SWME2     (h, hu, hv, h*alpha_1, h*beta_1, h*alpha_2, h*beta_2)
    u_scale : float
        Velocity scale U1 [m/s]
    h_scale : float
        Height scale H1 [m]
    init_velocity : str
        'zero'    — zero initial velocity and moment fields
        'nonzero' — non-trivial initial velocity and moment fields

    Returns
    -------
    ndarray
        State vector U of shape (nx, ny, var_num)
    """
    state = np.zeros((nx, ny, var_num))

    x_center = 0.5
    y_center = 0.5
    radius = 0.15

    for i in range(nx):
        xi = i * dx
        for j in range(ny):
            yj = j * dy
            if (xi - x_center) ** 2 + (yj - y_center) ** 2 <= radius ** 2:
                state[i, j, 0] = h_center
                if init_velocity == 'nonzero':
                    state[i, j, 1] = 0.125 * state[i, j, 0] ** 2 * h_scale / u_scale  # hu
                    state[i, j, 2] = 0.125 * state[i, j, 0] ** 2 * h_scale / u_scale  # hv
                    if var_num >= 5:
                        state[i, j, 3] = -0.125 * state[i, j, 0] ** 2 * h_scale / u_scale  # h*alpha_1
                        state[i, j, 4] = -0.125 * state[i, j, 0] ** 2 * h_scale / u_scale  # h*beta_1
                    if var_num == 7:
                        state[i, j, 5] = 0.0  # h*alpha_2
                        state[i, j, 6] = 0.0  # h*beta_2
            else:
                state[i, j, 0] = h_rest

    return state


def compute_jacobian(state, gravity, model_type='HSWME'):
    """
    Compute the Jacobian matrices in x- and y-directions at a single point.

    Parameters
    ----------
    state : ndarray
        State vector [h, hu, hv, h*alpha_1, h*beta_1, ...] at a single point
    gravity : float
        Dimensionless gravitational parameter G
    model_type : str, optional
        Model type ('HSWME' for higher-order SWME), default is 'HSWME'

    Returns
    -------
    matrix_x : ndarray
        Jacobian matrix in x-direction, shape (var_num, var_num)
    matrix_y : ndarray
        Jacobian matrix in y-direction, shape (var_num, var_num)
    """
    var_num = len(state)
    order_moment = (var_num - 3) / 2
    matrix_x = np.zeros((var_num, var_num))
    matrix_y = np.zeros((var_num, var_num))

    h, hu, hv = state[0], state[1], state[2]
    u, v = hu / h, hv / h

    # Base shallow water Jacobian — x-direction
    matrix_x[0, 1] = 1.0
    matrix_x[1, 0] = -u ** 2 + gravity * h
    matrix_x[1, 1] = 2.0 * u
    matrix_x[2, 0] = -u * v
    matrix_x[2, 1] = v
    matrix_x[2, 2] = u

    # Base shallow water Jacobian — y-direction
    matrix_y[0, 2] = 1.0
    matrix_y[1, 0] = -u * v
    matrix_y[1, 1] = v
    matrix_y[1, 2] = u
    matrix_y[2, 0] = -v ** 2 + gravity * h
    matrix_y[2, 2] = 2.0 * v

    # First-order moment terms
    if order_moment >= 1:
        alpha_1 = state[3] / h
        beta_1 = state[4] / h

        matrix_x[1, 0] += -alpha_1 ** 2 / 3.0
        matrix_x[1, 3] += 2.0 / 3.0 * alpha_1
        matrix_x[2, 0] += -1.0 / 3.0 * alpha_1 * beta_1
        matrix_x[2, 3] += 1.0 / 3.0 * beta_1
        matrix_x[2, 4] += 1.0 / 3.0 * alpha_1
        matrix_x[3, 0] += -2.0 * u * alpha_1
        matrix_x[3, 1] += 2.0 * alpha_1
        matrix_x[3, 3] += u
        matrix_x[4, 0] += -u * beta_1 - v * alpha_1
        matrix_x[4, 1] += beta_1
        matrix_x[4, 2] += alpha_1
        matrix_x[4, 4] += u

        matrix_y[1, 0] += -1.0 / 3.0 * alpha_1 * beta_1
        matrix_y[1, 3] += 1.0 / 3.0 * beta_1
        matrix_y[1, 4] += 1.0 / 3.0 * alpha_1
        matrix_y[2, 0] += -1.0 / 3.0 * beta_1 ** 2
        matrix_y[2, 4] += 2.0 / 3.0 * beta_1
        matrix_y[3, 0] += -u * beta_1 - v * alpha_1
        matrix_y[3, 1] += beta_1
        matrix_y[3, 2] += alpha_1
        matrix_y[3, 3] += v
        matrix_y[4, 0] += -2.0 * v * beta_1
        matrix_y[4, 2] += 2.0 * beta_1
        matrix_y[4, 4] += v

    # Second-order moment terms
    if order_moment == 2 and model_type == 'HSWME':
        alpha_1 = state[3] / h
        beta_1 = state[4] / h

        matrix_x[3, 5] += 3.0 / 5.0 * alpha_1
        matrix_x[4, 5] += 1.0 / 5.0 * beta_1
        matrix_x[4, 6] += 2.0 / 5.0 * alpha_1
        matrix_x[5, 0] += -2.0 / 3.0 * alpha_1 ** 2
        matrix_x[5, 3] += 1.0 / 3.0 * alpha_1
        matrix_x[5, 5] += u
        matrix_x[6, 0] += -2.0 / 3.0 * alpha_1 * beta_1
        matrix_x[6, 3] += -1.0 / 3.0 * beta_1
        matrix_x[6, 4] += 2.0 / 3.0 * alpha_1
        matrix_x[6, 6] += u

        matrix_y[3, 5] += 2.0 / 5.0 * beta_1
        matrix_y[3, 6] += 1.0 / 5.0 * alpha_1
        matrix_y[4, 6] += 3.0 / 5.0 * beta_1
        matrix_y[5, 0] += -2.0 / 3.0 * alpha_1 * beta_1
        matrix_y[5, 3] += 2.0 / 3.0 * beta_1
        matrix_y[5, 4] += -1.0 / 3.0 * alpha_1
        matrix_y[5, 5] += v
        matrix_y[6, 0] += -2.0 / 3.0 * beta_1 ** 2
        matrix_y[6, 4] += 1.0 / 3.0 * beta_1
        matrix_y[6, 6] += v

    return matrix_x, matrix_y


def compute_source_term(state, epsilon, alpha0, r0, model_type, nx, ny):
    """
    Compute the source term for friction and viscosity effects.

    Parameters
    ----------
    state : ndarray
        State vector of shape (nx, ny, var_num)
    epsilon : float
        Aspect ratio H1/L1
    alpha0 : float
        Friction coefficient (dimensionless)
    r0 : float
        Viscosity parameter (dimensionless)
    model_type : str
        'original' or 'modified' formulation
    nx : int
        Number of grid cells in x
    ny : int
        Number of grid cells in y

    Returns
    -------
    ndarray
        Source term array of shape (nx, ny, var_num)
    """
    var_num = state.shape[2]
    order_moment = int((var_num - 3) / 2)
    source = np.zeros((nx, ny, var_num))

    h = state[:, :, 0]
    u = state[:, :, 1] / h
    v = state[:, :, 2] / h

    if model_type == 'original':
        coe0 = alpha0
        coe11 = 1.0 + 4.0 * r0 / h / epsilon / alpha0
        coe12 = 1.0
        coe21 = 1.0
        coe22 = 1.0 + 12.0 * r0 / h / epsilon / alpha0
    elif model_type == 'modified':
        alpha_bar = alpha0 / (1.0 + h * epsilon * alpha0 / 2.0 / r0)
        coe0 = alpha_bar
        coe11 = 4.0 * r0 / h / epsilon / alpha_bar
        coe12 = 0.0
        coe21 = 0.0
        coe22 = 12.0 * r0 / h / epsilon / alpha_bar

    source[:, :, 1] = -coe0 * u
    source[:, :, 2] = -coe0 * v

    if order_moment == 1:
        alpha1 = state[:, :, 3] / h
        beta1 = state[:, :, 4] / h
        source[:, :, 3] = -3.0 * coe0 * (u + coe11 * alpha1)
        source[:, :, 4] = -3.0 * coe0 * (v + coe11 * beta1)

    elif order_moment == 2:
        alpha1 = state[:, :, 3] / h
        beta1 = state[:, :, 4] / h
        alpha2 = state[:, :, 5] / h
        beta2 = state[:, :, 6] / h
        source[:, :, 3] = -3.0 * coe0 * (u + coe11 * alpha1 + coe12 * alpha2)
        source[:, :, 4] = -3.0 * coe0 * (v + coe11 * beta1 + coe12 * beta2)
        source[:, :, 5] = -5.0 * coe0 * (u + coe21 * alpha1 + coe22 * alpha2)
        source[:, :, 6] = -5.0 * coe0 * (v + coe21 * beta1 + coe22 * beta2)

    return source


def compute_max_eigenvalue(state, gravity):
    """
    Compute the maximum eigenvalues in x- and y-directions at a single point.

    Parameters
    ----------
    state : ndarray
        State vector [h, hu, hv, h*alpha_1, h*beta_1, ...]
    gravity : float
        Dimensionless gravitational parameter G

    Returns
    -------
    max_eigenvalue_x : float
        Maximum wave speed in x-direction
    max_eigenvalue_y : float
        Maximum wave speed in y-direction

    Raises
    ------
    ValueError
        If water height is non-positive
    """
    var_num = len(state)
    order_moment = int((var_num - 3) / 2)

    h = state[0]
    if h <= 0:
        raise ValueError(f"Water height h must be positive, got {h}")

    u = state[1] / h
    v = state[2] / h
    alpha_1 = state[3] / h if order_moment > 0 else 0.0
    beta_1 = state[4] / h if order_moment > 0 else 0.0

    max_eigenvalue_x = np.abs(u) + np.sqrt(gravity * h + alpha_1 ** 2)
    max_eigenvalue_y = np.abs(v) + np.sqrt(gravity * h + beta_1 ** 2)

    return max_eigenvalue_x, max_eigenvalue_y


def roe_scheme_2d(state, gravity, nx, ny, dx, dy, dt):
    """
    Apply the 2D Roe numerical scheme for spatial discretization.

    Computes the right-hand side contributions from x- and y-direction
    fluxes using Roe averages at cell interfaces.

    Parameters
    ----------
    state : ndarray
        Current state vector of shape (nx, ny, var_num)
    gravity : float
        Dimensionless gravitational parameter G
    nx : int
        Number of grid cells in x
    ny : int
        Number of grid cells in y
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
    dt : float
        Time step size

    Returns
    -------
    ndarray
        Right-hand side array of shape (nx, ny, var_num)
    """
    var_num = state.shape[2]
    rhs = np.zeros_like(state)

    # --- Roe averages and max wave speeds at interfaces ---
    u_roe_x = np.zeros_like(state)
    u_roe_y = np.zeros_like(state)
    max_wavespeed_x = np.zeros((nx, ny))
    max_wavespeed_y = np.zeros((nx, ny))

    for i in range(nx):
        i_plus = min(i + 1, nx - 1)
        for j in range(ny):
            j_plus = min(j + 1, ny - 1)
            u_roe_x[i, j, :] = 0.5 * (state[i, j, :] + state[i_plus, j, :])
            u_roe_y[i, j, :] = 0.5 * (state[i, j, :] + state[i, j_plus, :])
            max_wavespeed_x[i, j], _ = compute_max_eigenvalue(u_roe_x[i, j, :], gravity)
            _, max_wavespeed_y[i, j] = compute_max_eigenvalue(u_roe_y[i, j, :], gravity)

    # --- Jacobians at x-interfaces ---
    jacobians_x = []
    for i in range(nx):
        for j in range(ny):
            jac, _ = compute_jacobian(u_roe_x[i, j, :], gravity)
            jacobians_x.append(jac)

    # --- x-direction RHS contributions ---
    for i in range(1, nx - 1):
        i_minus = i - 1
        i_plus = i + 1
        for j in range(1, ny - 1):
            for m in range(var_num):
                matrix_contrib = 0.0
                for n in range(var_num):
                    term_left = jacobians_x[i_minus * ny + j][m][n] * (state[i, j, n] - state[i_minus, j, n])
                    term_right = jacobians_x[i * ny + j][m][n] * (state[i_plus, j, n] - state[i, j, n])
                    matrix_contrib += -dt / (2.0 * dx) * (term_left + term_right)

                delta_left = state[i, j, m] - state[i_minus, j, m]
                delta_right = state[i_plus, j, m] - state[i, j, m]
                wave_contrib = -dt / (2.0 * dx) * (
                    max_wavespeed_x[i_minus, j] * delta_left - max_wavespeed_x[i, j] * delta_right
                )
                rhs[i, j, m] += matrix_contrib + wave_contrib

    # --- Jacobians at y-interfaces ---
    jacobians_y = []
    for i in range(nx):
        for j in range(ny):
            _, jac = compute_jacobian(u_roe_y[i, j, :], gravity)
            jacobians_y.append(jac)

    # --- y-direction RHS contributions ---
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            j_minus = j - 1
            j_plus = j + 1
            for m in range(var_num):
                matrix_contrib = 0.0
                for n in range(var_num):
                    term_left = jacobians_y[i * ny + j_minus][m][n] * (state[i, j, n] - state[i, j_minus, n])
                    term_right = jacobians_y[i * ny + j][m][n] * (state[i, j_plus, n] - state[i, j, n])
                    matrix_contrib += -dt / (2.0 * dy) * (term_left + term_right)

                delta_left = state[i, j, m] - state[i, j_minus, m]
                delta_right = state[i, j_plus, m] - state[i, j, m]
                wave_contrib = -dt / (2.0 * dy) * (
                    max_wavespeed_y[i, j_minus] * delta_left - max_wavespeed_y[i, j] * delta_right
                )
                rhs[i, j, m] += matrix_contrib + wave_contrib

    return rhs


def swme_solver(u_scale, h_scale, h_center, h_rest, length, gravity, nx, ny, cfl,
                time_end, alpha0, r0, epsilon, model_type, var_num, t_target, init_case):
    """
    Main solver for the 2D Shallow Water Moment Expansion equations.

    Parameters
    ----------
    u_scale : float
        Velocity scale U1 [m/s]
    h_scale : float
        Height scale H1 [m]
    h_center : float
        Initial dimensionless water height inside the dam
    h_rest : float
        Initial dimensionless water height outside the dam
    length : float
        Domain length L (assumed square: L x L) [dimensionless]
    gravity : float
        Dimensionless gravitational parameter G
    nx : int
        Number of grid cells in x
    ny : int
        Number of grid cells in y
    cfl : float
        CFL number for time step calculation
    time_end : float
        End time for simulation [dimensionless]
    alpha0 : float
        Friction coefficient [dimensionless]
    r0 : float
        Viscosity parameter [dimensionless]
    epsilon : float
        Aspect ratio H1/L1
    model_type : str
        'original' or 'modified' formulation
    var_num : int
        Number of variables (3, 5, or 7)
    t_target : list
        Target time instances for which snapshots are saved
    init_case : str
        Initial velocity condition ('zero' or 'nonzero')

    Returns
    -------
    tuple
        (U_history, dt_history) — list of state snapshots and list of time steps
    """
    dx = length / nx
    dy = length / ny

    state = initialize_state(nx, ny, dx, dy, h_center, h_rest, var_num,
                             u_scale, h_scale, init_velocity=init_case)

    u_history = [state.copy()]
    dt_history = [0.0]

    t_curr = 0.0
    time_step = 0
    target_idx = 0

    print("=" * 70)
    print(f"Starting 2D simulation: model={model_type}, var_num={var_num}")
    print(f"Domain: nx={nx}, ny={ny}, dx={dx:.4e}, dy={dy:.4e}, CFL={cfl}")
    print(f"Parameters: alpha0={alpha0:.4e}, R0={r0:.4e}, epsilon={epsilon:.4e}")
    print("=" * 70)

    while t_curr < time_end:
        # --- CFL-based time step ---
        wavespeed_x = np.zeros((nx, ny))
        wavespeed_y = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                wavespeed_x[i, j], wavespeed_y[i, j] = compute_max_eigenvalue(state[i, j, :], gravity)

        max_wavespeed_x = np.max(wavespeed_x)
        max_wavespeed_y = np.max(wavespeed_y)
        dt = cfl / (max_wavespeed_x / dx + max_wavespeed_y / dy)
        dt = min(dt, time_end - t_curr)

        # Snap to target output times
        if target_idx < len(t_target) and dt > t_target[target_idx] - t_curr:
            dt = t_target[target_idx] - t_curr
            t_curr = t_target[target_idx]
            target_idx += 1
        else:
            t_curr += dt

        time_step += 1
        dt_history.append(dt)

        # --- Spatial discretization and source term ---
        rhs = roe_scheme_2d(state, gravity, nx, ny, dx, dy, dt)
        source = compute_source_term(state, epsilon, alpha0, r0, model_type, nx, ny)

        # --- Time integration ---
        if var_num == 3 and model_type == 'original':
            h = state[:, :, 0].copy()
            state[:, :, 0] += rhs[:, :, 0]
            state[:, :, 1] = (state[:, :, 1] + rhs[:, :, 1]) / (1.0 + dt * alpha0 / h)
            state[:, :, 2] = (state[:, :, 2] + rhs[:, :, 2]) / (1.0 + dt * alpha0 / h)

        elif var_num == 5 and model_type == 'original':
            h = state[:, :, 0].copy()
            state[:, :, 0] += rhs[:, :, 0]
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    a = dt * alpha0 / h[i, j]
                    b = r0 / h[i, j] / epsilon / alpha0
                    A = np.array([[1 + a,     0,         a,               0        ],
                                  [0,         1 + a,     0,               a        ],
                                  [3 * a,     0,         1 + 3*a*(1+4*b), 0        ],
                                  [0,         3 * a,     0,               1 + 3*a*(1+4*b)]])
                    f = (state[i, j, 1:] + rhs[i, j, 1:]).reshape(var_num - 1, 1)
                    uh = np.linalg.solve(A, f)
                    state[i, j, 1:5] = uh[:4].squeeze()

        elif var_num == 7 and model_type == 'original':
            h = state[:, :, 0].copy()
            state[:, :, 0] += rhs[:, :, 0]
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    a = dt * alpha0 / h[i, j]
                    b = r0 / h[i, j] / epsilon / alpha0
                    A = np.array([[1 + a,   0,       a,               0,               a,               0              ],
                                  [0,       1 + a,   0,               a,               0,               a              ],
                                  [3 * a,   0,       1+3*a*(1+4*b),   0,               3 * a,           0              ],
                                  [0,       3 * a,   0,               1+3*a*(1+4*b),   0,               3 * a          ],
                                  [5 * a,   0,       5 * a,           0,               1+5*a*(1+12*b),  0              ],
                                  [0,       5 * a,   0,               5 * a,           0,               1+5*a*(1+12*b)]])
                    f = (state[i, j, 1:] + rhs[i, j, 1:]).reshape(var_num - 1, 1)
                    uh = np.linalg.solve(A, f)
                    state[i, j, 1:7] = uh[:6].squeeze()

        elif model_type == 'modified':
            state += rhs + source * dt

        # --- Zero-gradient boundary conditions ---
        state[0, :, :] = state[1, :, :]
        state[-1, :, :] = state[-2, :, :]
        state[:, 0, :] = state[:, 1, :]
        state[:, -1, :] = state[:, -2, :]

        u_history.append(state.copy())

        if time_step % 50 == 0:
            max_u = np.max(np.abs(state[:, :, 1] / state[:, :, 0]))
            print(f"Step {time_step:6d} | t={t_curr:8.4f} | dt={dt:8.4e} | "
                  f"max_wave_x={max_wavespeed_x:8.4e} | max_u={max_u:8.4e}")
            if var_num >= 5:
                max_alpha1 = np.max(np.abs(state[:, :, 3] / state[:, :, 0]))
                print(f"             | max_alpha1={max_alpha1:8.4e}")

    print("-" * 70)
    print(f"Simulation completed: {time_step} time steps")
    print(f"Final time: {t_curr:.4f}, Final dt: {dt:.4e}")
    print(f"Final max velocity: {np.max(np.abs(state[:, :, 1] / state[:, :, 0])):.4e}")

    return u_history, dt_history


def save_data(model_type, nx, k_coe, var_num, u_history, dt_history, dx, time_end,
              length, init_case, output_dir='Data'):
    """
    Save simulation results to a NumPy binary file.

    Parameters
    ----------
    model_type : str
        'original' or 'modified'
    nx : int
        Number of grid cells in x (used in filename)
    k_coe : float
        Friction coefficient value
    var_num : int
        Number of variables (3, 5, or 7)
    u_history : list
        History of state snapshots
    dt_history : list
        History of time steps
    dx : float
        Grid spacing
    time_end : float
        End time
    length : float
        Domain length
    init_case : str
        Initial condition label ('zero' or 'nonzero')
    output_dir : str, optional
        Output directory name, default is 'Data'
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    moment_num = int((var_num - 3) / 2)
    model_names = {
        'original': f"HSWME{moment_num}",
        'modified': f"M-HSWME{moment_num}",
    }
    model_name = model_names.get(model_type, 'unknown')
    filename = f"{init_case}_{model_name}_N{nx}_k{k_coe:.1e}.npy"
    filepath = os.path.join(output_dir, filename)

    data_dict = {
        'U_history': u_history,
        'dt_history': dt_history,
        'dx': dx,
        'time_end': time_end,
        'L': length,
        'k_coe': k_coe,
    }
    np.save(filepath, data_dict)
    print(f"Data saved to {filepath}")


def load_data(filepath):
    """
    Load simulation results from a NumPy binary file.

    Parameters
    ----------
    filepath : str
        Path to the .npy file

    Returns
    -------
    dict
        Dictionary with keys: 'U_history', 'dt_history', 'dx', 'time_end', 'L', 'k_coe'
    """
    return np.load(filepath, allow_pickle=True).item()


if __name__ == "__main__":
    print("SWME 2D Solver Module")
    print("This module should be imported and used from a separate script.")
    print("See example usage in test_run.py")
