"""
Modified Hyperbolic Shallow Water Moment Expansion (MHSWME) Solver 1D

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

This module implements a 1D shallow water moment expansion system solver
using the Roe scheme for spatial discretization and explicit time integration.

The system of equations solved:
    h_t + (hu_m)_x = 0
    (hu_m)_t + (hu_m^2 + 1/3*h*alpha_1^2 + G/2*h^2)_x = -a0*(u_m + sum alpha_j)
    (h*alpha_1)_t + (2*h*u_m*alpha_1)_x = u_m*(h*alpha_1)_x - 3*a0*(u_m + (1+4*R_0/h/epsilon/a0)*alpha_1)

where:
    h: water height
    u_m: mean velocity
    alpha_i: velocity moments
    G: gravitational parameter
    a0: friction coefficient
    R_0: viscosity parameter
    epsilon: scale parameter
"""

import os
import numpy as np


def initialize_state(nx, h_left, h_right, var_num, case, u_scale, h_scale):
    """
    Initialize the state vector for the dam break problem.
    
    Parameters
    ----------
    nx : int
        Number of computational cells
    h_left : float
        Initial water height on the left side of the dam (dimensionless)
    h_right : float
        Initial water height on the right side of the dam (dimensionless)
    var_num : int
        Number of variables (2 + order of moment expansion)
        2: Shallow Water Equations (h, hu)
        3: First-order moment expansion (h, hu, h*alpha_1)
        4: Second-order moment expansion (h, hu, h*alpha_1, h*alpha_2)
    case : str
        Initial condition type ('init' for default initialization)
    u_scale : float
        Velocity scale for dimensional conversion
    h_scale : float
        Height scale for dimensional conversion
    
    Returns
    -------
    ndarray
        State vector U of shape (nx, var_num)
        U = [h, hu, h*alpha_1, ..., h*alpha_N]^T
    """
    state = np.zeros((nx, var_num))
    
    # Initialize water height with dam break configuration
    state[:, 0] = h_right
    state[:nx // 2, 0] = h_left
    
    # Initialize velocity and moment fields based on model order
    if var_num == 2:
        state[:, 1] = 0.125 * state[:, 0]**2 * h_scale / u_scale
    elif var_num == 3 and case == 'init':
        state[:, 1] = 0.125 * state[:, 0]**2 * h_scale / u_scale
        state[:, 2] = -0.125 * state[:, 0]**2 * h_scale / u_scale
    elif var_num == 4 and case == 'init':
        state[:, 1] = 0.125 * state[:, 0]**2 * h_scale / u_scale
        state[:, 2] = -0.125 * state[:, 0]**2 * h_scale / u_scale
        state[:, 3] = 0.0
    
    return state


def compute_jacobian(state, gravity, model_type='HSWME'):
    """
    Compute the Jacobian matrix for the flux function in the x-direction.
    
    Parameters
    ----------
    state : ndarray
        State vector [h, hu, h*alpha_1, ...] at a single point
    gravity : float
        Gravitational parameter G
    model_type : str, optional
        Model type ('HSWME' for higher-order SWME), default is 'HSWME'
    
    Returns
    -------
    ndarray
        Jacobian matrix of shape (var_num, var_num)
    """
    var_num = len(state)
    order_moment = var_num - 2
    jacobian = np.zeros((var_num, var_num))
    
    # Extract primary variables
    h = state[0]
    hu = state[1]
    u = hu / h
    
    # Base shallow water equations Jacobian
    jacobian[0, 0] = 0.0
    jacobian[0, 1] = 1.0
    jacobian[1, 0] = -u**2 + gravity * h
    jacobian[1, 1] = 2.0 * u
    
    # First-order moment terms
    if order_moment >= 1:
        h_alpha_1 = state[2]
        alpha_1 = h_alpha_1 / h
        
        jacobian[1, 0] += -alpha_1**2 / 3.0
        jacobian[1, 2] += 2.0 * alpha_1 / 3.0
        jacobian[2, 0] += -2.0 * u * alpha_1
        jacobian[2, 1] += 2.0 * alpha_1
        jacobian[2, 2] += u
    
    # Second-order moment terms
    if order_moment == 2 and model_type == 'HSWME':
        h_alpha_1 = state[2]
        alpha_1 = h_alpha_1 / h
        
        jacobian[2, 3] += 3.0 * alpha_1 / 5.0
        jacobian[3, 0] += -2.0 * alpha_1**2 / 3.0
        jacobian[3, 2] += alpha_1 / 3.0
        jacobian[3, 3] += u
    
    return jacobian


def compute_source_term(state, epsilon, alpha0, r0, model_type, nx):
    """
    Compute the source term for friction and viscosity effects.
    
    Parameters
    ----------
    state : ndarray
        State vector of shape (nx, var_num)
    epsilon : float
        Scale parameter H1/L1
    alpha0 : float
        Friction coefficient
    r0 : float
        Viscosity parameter
    model_type : str
        'original' or 'modified' formulation
    nx : int
        Number of grid cells
    
    Returns
    -------
    ndarray
        Source term array of shape (nx, var_num)
    """
    var_num = state.shape[1]
    order_moment = var_num - 2
    source = np.zeros((nx, var_num))
    
    # Extract primary variables
    h = state[:, 0]
    hu = state[:, 1]
    u = hu / h
    
    if model_type == 'original':
        # Sum of all moment coefficients
        alpha_sum = np.sum(state[:, 2:] / h[:, np.newaxis], axis=1) if order_moment > 0 else 0.0
        coe0 = alpha0
        coe11 = 1.0 + 4.0 * r0 / h / epsilon / alpha0
        coe12 = 1.0
        coe21 = 1.0
        coe22 = 1.0 + 12.0 * r0 / h / epsilon / alpha0
    elif model_type == 'modified':
        alpha_sum = 0.0
        alpha_bar = alpha0 / (1 + h * epsilon * alpha0 / 2 / r0)
        coe0 = alpha_bar
        coe11 = 4.0 * r0 / h / epsilon / alpha_bar
        coe12 = 0.0
        coe21 = 0.0
        coe22 = 12.0 * r0 / h / epsilon / alpha_bar
    
    # Momentum equation source term
    source[:, 1] = -coe0 * (u + alpha_sum)
    
    # Moment equation source terms
    if order_moment == 1:
        alpha1 = state[:, 2] / h
        source[:, 2] = -3.0 * coe0 * (u + coe11 * alpha1)
    elif order_moment == 2:
        alpha1 = state[:, 2] / h
        alpha2 = state[:, 3] / h
        source[:, 2] = -3.0 * coe0 * (u + coe11 * alpha1 + coe12 * alpha2)
        source[:, 3] = -5.0 * coe0 * (u + coe21 * alpha1 + coe22 * alpha2)
    
    return source


def compute_max_eigenvalue(state, gravity):
    """
    Compute the maximum eigenvalue at a single point.
    
    The maximum eigenvalue determines the wave speed and is used
    for CFL condition and numerical stability.
    
    Parameters
    ----------
    state : ndarray
        State vector [h, hu, h*alpha_1, ...]
    gravity : float
        Gravitational parameter G
    
    Returns
    -------
    float
        Maximum eigenvalue (wave speed)
    
    Raises
    ------
    ValueError
        If water height is non-positive
    """
    var_num = len(state)
    order_moment = var_num - 2
    
    h = state[0]
    if np.any(h <= 0):
        raise ValueError(f"Water height h must be positive, got {h}")
    
    u = state[1] / h
    alpha_1 = 0.0
    
    if order_moment > 0:
        alpha_1 = state[2] / h
    
    max_eigenvalue = np.abs(u) + np.sqrt(gravity * h + alpha_1**2)
    return max_eigenvalue


def roe_scheme_1d(state, gravity, nx, dx, dt):
    """
    Apply the Roe numerical scheme for spatial discretization.
    
    The Roe scheme is a finite volume method that computes the
    numerical flux using Roe averages at cell interfaces.
    
    Parameters
    ----------
    state : ndarray
        Current state vector of shape (nx, var_num)
    gravity : float
        Gravitational parameter G
    nx : int
        Number of grid cells
    dx : float
        Grid spacing
    dt : float
        Time step
    
    Returns
    -------
    ndarray
        Right-hand side (RHS) of the discretized equations
    """
    nx, var_num = state.shape
    rhs = np.zeros_like(state)
    
    # Compute Roe averages at each interface
    u_roe = np.zeros_like(state)
    max_wavespeed = np.zeros(nx)
    
    for i in range(nx):
        i_plus = min(i + 1, nx - 1)  # Boundary condition
        u_roe[i, :] = 0.5 * (state[i, :] + state[i_plus, :])
        max_wavespeed[i] = compute_max_eigenvalue(u_roe[i, :], gravity)
    
    # Compute Jacobian matrices at Roe average states
    jacobians = []
    for i in range(nx):
        j = compute_jacobian(u_roe[i, :], gravity)
        jacobians.append(j)
    
    # Update RHS using Roe scheme (interior points only)
    for i in range(1, nx - 1):
        i_minus = i - 1
        i_plus = i + 1
        
        for j in range(var_num):
            matrix_contrib = 0.0
            for k in range(var_num):
                # Left interface contribution (i-1/2)
                term_left = jacobians[i_minus][j][k] * (state[i][k] - state[i_minus][k])
                # Right interface contribution (i+1/2)
                term_right = jacobians[i][j][k] * (state[i_plus][k] - state[i][k])
                matrix_contrib += -dt / (2 * dx) * (term_left + term_right)
            
            # Numerical dissipation
            delta_left = state[i][j] - state[i_minus][j]
            delta_right = state[i_plus][j] - state[i][j]
            wave_contrib = -dt / (2 * dx) * (max_wavespeed[i_minus] * delta_left 
                                             - max_wavespeed[i] * delta_right)
            
            rhs[i][j] = matrix_contrib + wave_contrib
    
    return rhs


def swme_solver(u_scale, h_scale, h_left, h_right, length, gravity, nx, cfl,
                time_end, alpha0, r0, epsilon, model_type, var_num, t_target,
                initial_case):
    """
    Main solver for the Shallow Water Moment Expansion equations.
    
    Parameters
    ----------
    u_scale : float
        Velocity scale U1 [m/s]
    h_scale : float
        Height scale H1 [m]
    h_left : float
        Initial dimensionless water height on left side of dam
    h_right : float
        Initial dimensionless water height on right side of dam
    length : float
        Domain length L [dimensionless]
    gravity : float
        Dimensionless gravitational parameter G
    nx : int
        Number of grid cells
    cfl : float
        CFL number for time step calculation
    time_end : float
        End time for simulation [dimensionless]
    alpha0 : float
        Friction coefficient [dimensionless]
    r0 : float
        Viscosity parameter [dimensionless]
    epsilon : float
        Scale parameter H1/L1
    model_type : str
        'original' or 'modified' formulation
    var_num : int
        Number of variables (2, 3, or 4)
    t_target : list
        Target time instances for output
    initial_case : str
        Initial condition type ('init')
    
    Returns
    -------
    tuple
        (U_history, dt_history) containing the state history and time steps
    """
    dx = length / nx
    state = initialize_state(nx, h_left, h_right, var_num, initial_case,
                            u_scale, h_scale)
    
    u_history = [state.copy()]
    dt_history = []
    
    t_curr = 0.0
    time_step = 0
    target_idx = 0
    
    print(f"Starting simulation: model={model_type}, var_num={var_num}")
    print(f"Domain: nx={nx}, dx={dx:.4e}, CFL={cfl}")
    print(f"Parameters: alpha0={alpha0:.4e}, R0={r0:.4e}, epsilon={epsilon:.4e}")
    print("-" * 70)
    
    while t_curr < time_end:
        # Compute maximum wave speed for CFL condition
        wavespeed = np.zeros(nx)
        for i in range(nx):
            wavespeed[i] = compute_max_eigenvalue(state[i, :], gravity)
        
        max_wavespeed = np.max(wavespeed)
        dt = cfl * dx / max_wavespeed
        dt = min(dt, time_end - t_curr)
        
        # Ensure target times are captured
        if target_idx < len(t_target) and dt > t_target[target_idx] - t_curr:
            dt = t_target[target_idx] - t_curr
            t_curr = t_target[target_idx]
            target_idx += 1
        else:
            t_curr += dt
        
        time_step += 1
        dt_history.append(dt)
        
        # Compute RHS and source term
        rhs = roe_scheme_1d(state, gravity, nx, dx, dt)
        source = compute_source_term(state, epsilon, alpha0, r0, model_type, nx)
        
        # Time integration with implicit treatment of source terms
        if var_num == 2 and model_type == 'original':
            state[:, 0] += rhs[:, 0]
            state[:, 1] = (state[:, 1] + rhs[:, 1]) / (1 + dt * alpha0 / state[:, 0])
        
        elif var_num == 3 and model_type == 'original':
            h = state[:, 0].copy()
            for i in range(1, nx - 1):
                state[i, 0] += rhs[i, 0] + source[i, 0] * dt
                
                # Implicit solve for momentum and first moment
                a11 = 1 + dt * alpha0 / h[i]
                a12 = dt * alpha0 / h[i]
                a21 = 3 * dt * alpha0 / h[i]
                a22 = 1 + dt * alpha0 / h[i] * (3 + 12 * r0 / h[i] / epsilon / alpha0)
                det = a22 * a11 - a12 * a21
                
                f1 = state[i, 1] + rhs[i, 1]
                f2 = state[i, 2] + rhs[i, 2]
                state[i, 1] = (f1 * a22 - f2 * a12) / det
                state[i, 2] = (a11 * f2 - a21 * f1) / det
        
        elif var_num == 4 and model_type == 'original':
            h = state[:, 0].copy()
            for i in range(1, nx - 1):
                state[i, 0] += rhs[i, 0] + source[i, 0] * dt
                
                # Implicit solve for momentum and moments
                a = dt * alpha0 / h[i]
                b = r0 / h[i] / epsilon / alpha0
                matrix_a = np.array([[1 + a, a, a],
                                    [3 * a, 1 + 3 * (1 + 4 * b) * a, 3 * a],
                                    [5 * a, 5 * a, 1 + 5 * (1 + 12 * b) * a]])
                f = (state[i, 1:] + rhs[i, 1:]).reshape(3, 1)
                uh = np.linalg.solve(matrix_a, f)
                state[i, 1:4] = uh.flatten()
        
        elif model_type == 'modified':
            state += rhs + source * dt
        
        # Apply boundary conditions (zero gradient)
        state[0, :] = state[1, :]
        state[-1, :] = state[-2, :]
        
        u_history.append(state.copy())
        
        # Print progress
        if time_step % 100 == 0:
            max_velocity = np.max(np.abs(state[:, 1] / state[:, 0]))
            print(f"Step {time_step:6d} | t={t_curr:8.4f} | dt={dt:8.4e} | "
                  f"max_wave={max_wavespeed:8.4e} | max_u={max_velocity:8.4e}")
            if var_num >= 3:
                max_alpha1 = np.max(np.abs(state[:, 2] / state[:, 0]))
                print(f"             | max_alpha1={max_alpha1:8.4e}")
    
    # Final summary
    print("-" * 70)
    print(f"Simulation completed: {time_step} time steps")
    print(f"Final time: {t_curr:.4f}, Final dt: {dt:.4e}")
    print(f"Final max velocity: {np.max(np.abs(state[:, 1] / state[:, 0])):.4e}")
    
    return u_history, dt_history


def save_data(model_type, k_coe, var_num, u_history, dt_history, dx,
              time_end, length, output_dir='Data'):
    """
    Save simulation results to a NumPy binary file.
    
    Parameters
    ----------
    model_type : str
        'original' or 'modified'
    k_coe : float
        Friction coefficient value
    var_num : int
        Number of variables (2, 3, or 4)
    u_history : list
        History of state vectors
    dt_history : list
        History of time steps
    dx : float
        Grid spacing
    time_end : float
        End time
    length : float
        Domain length
    output_dir : str, optional
        Output directory name, default is 'Data'
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine filename based on model type and variable number
    model_names = {
        (2, 'original'): 'SWE',
        (2, 'modified'): 'M-SWE',
        (3, 'original'): 'SWME',
        (3, 'modified'): 'M-SWME',
        (4, 'original'): 'SWME2',
        (4, 'modified'): 'M-SWME2'
    }
    
    model_name = model_names.get((var_num, model_type), 'unknown')
    filename = f"{model_name}_data_k{k_coe}.npy"
    filepath = os.path.join(output_dir, filename)
    
    # Save data
    data_dict = {
        'U_history': u_history,
        'dt_history': dt_history,
        'dx': dx,
        'time_end': time_end,
        'L': length,
        'k_coe': k_coe
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
        Dictionary containing simulation data with keys:
        'U_history', 'dt_history', 'dx', 'time_end', 'L', 'k_coe'
    """
    data = np.load(filepath, allow_pickle=True).item()
    return data


if __name__ == "__main__":
    print("SWME Solver Module")
    print("This module should be imported and used from a separate script.")
    print("See example usage in test_run.py")
