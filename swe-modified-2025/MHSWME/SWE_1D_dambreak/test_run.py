"""
Test script for the Shallow Water Moment Expansion (SWME) solver.

This script demonstrates the usage of the SWME solver for simulating
dam break problems with different model configurations.
"""

import numpy as np
from mswme import (
    initialize_state,
    compute_jacobian,
    compute_source_term,
    compute_max_eigenvalue,
    roe_scheme_1d,
    swme_solver,
    save_data
)


def run_dam_break_simulation():
    """
    Run dam break simulations with various model configurations.
    
    This function sets up and executes simulations comparing:
    - Shallow Water Equations (SWE) vs Moment Expansion (SWME)
    - Original vs Modified formulations
    - Different orders of moment expansion
    """
    # Physical constants
    g = 9.81  # Gravity [m/s^2]
    nu = 1e-6  # Kinematic viscosity [m^2/s]
    rho = 1000  # Water density [kg/m^3]
    
    # Domain configuration
    L = 1.0  # Domain length [dimensionless]
    H_left = 1.0  # Initial water height on left side of dam [dimensionless]
    H_right = 2.0 / 3.0  # Initial water height on right side of dam [dimensionless]
    
    # Numerical parameters
    Nx = 4000  # Number of grid cells
    dx = L / Nx  # Cell width [m]
    CFL = 0.7  # Courant-Friedrichs-Lewy number
    
    # Characteristic scales
    L1 = 100  # Length scale [m]
    H1 = 1.5  # Height scale [m]
    U1 = L1  # Velocity scale [m/s]
    
    # Dimensionless parameters
    epsilon = H1 / L1  # Aspect ratio
    G = g * H1 / U1**2  # Dimensionless gravity
    R0 = epsilon * (nu / U1 / H1)  # Dimensionless viscosity
    
    # Simulation time
    time_end = 3.0  # End time [dimensionless]
    t_target = [1.0 * i + 1 for i in range(int(time_end))]  # Target output times
    
    # Friction coefficient values to test
    k_coefficients = [1e10]
    
    # Model configurations to test
    var_nums = [2, 3, 4]  # Number of variables (2: SWE, 3: SWME1, 4: SWME2)
    model_types = ['original', 'modified']
    
    print("=" * 70)
    print("Dam Break Simulation Suite")
    print("=" * 70)
    print(f"Domain: L={L}, Nx={Nx}, dx={dx:.6f}")
    print(f"Initial conditions: H_left={H_left}, H_right={H_right}")
    print(f"Scales: L1={L1} m, H1={H1} m, U1={U1} m/s")
    print(f"Dimensionless: epsilon={epsilon:.6e}, G={G:.6e}, R0={R0:.6e}")
    print(f"Time: t_end={time_end}, CFL={CFL}")
    print("=" * 70)
    
    # Run simulations for each configuration
    for k_coe in k_coefficients:
        alpha0 = L1 / rho / U1 / H1 * k_coe
        print(f"\nFriction coefficient k={k_coe:.2e}, alpha0={alpha0:.6e}")
        print("-" * 70)
        
        for var_num in var_nums:
            for model_type in model_types:
                print(f"\nConfiguration: var_num={var_num}, model={model_type}")
                
                # Run simulation
                U_history, dt_history = swme_solver(
                    u_scale=U1,
                    h_scale=H1,
                    h_left=H_left,
                    h_right=H_right,
                    length=L,
                    gravity=G,
                    nx=Nx,
                    cfl=CFL,
                    time_end=time_end,
                    alpha0=alpha0,
                    r0=R0,
                    epsilon=epsilon,
                    model_type=model_type,
                    var_num=var_num,
                    t_target=t_target,
                    initial_case='init'
                )
                
                # Save results
                save_data(
                    model_type=model_type,
                    k_coe=k_coe,
                    var_num=var_num,
                    u_history=U_history,
                    dt_history=dt_history,
                    dx=dx,
                    time_end=time_end,
                    length=L
                )
                
                # Print summary statistics
                final_state = U_history[-1]
                max_velocity = np.max(np.abs(final_state[:, 1] / final_state[:, 0]))
                print(f"Final max velocity: {max_velocity:.6e}")
                
                if var_num >= 3:
                    max_alpha1 = np.max(np.abs(final_state[:, 2] / final_state[:, 0]))
                    print(f"Final max alpha1: {max_alpha1:.6e}")
    
    print("\n" + "=" * 70)
    print("All simulations completed successfully!")
    print("=" * 70)


def analyze_saved_data(filepath):
    """
    Load and analyze saved simulation data.
    
    Parameters
    ----------
    filepath : str
        Path to the .npy file containing simulation results
    """
    from mswme import load_data
    
    data = load_data(filepath)
    
    print(f"Loaded data from: {filepath}")
    print(f"Domain length: {data['L']}")
    print(f"Grid spacing: {data['dx']:.6e}")
    print(f"End time: {data['time_end']}")
    print(f"Friction coefficient: {data['k_coe']:.2e}")
    print(f"Number of time steps: {len(data['dt_history'])}")
    print(f"Number of snapshots: {len(data['U_history'])}")
    
    # Compute statistics
    time_history = np.cumsum(data['dt_history'])
    final_state = data['U_history'][-1]
    
    print(f"\nFinal state statistics:")
    print(f"  Max height: {np.max(final_state[:, 0]):.6f}")
    print(f"  Min height: {np.min(final_state[:, 0]):.6f}")
    print(f"  Max velocity: {np.max(np.abs(final_state[:, 1] / final_state[:, 0])):.6e}")
    
    if final_state.shape[1] >= 3:
        print(f"  Max alpha1: {np.max(np.abs(final_state[:, 2] / final_state[:, 0])):.6e}")


if __name__ == "__main__":
    # Run the dam break simulation suite
    run_dam_break_simulation()
    
    # Example: Analyze saved data (uncomment to use)
    # analyze_saved_data('Data/SWME_data_k1.00e+10.npy')
