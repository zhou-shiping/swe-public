from pathlib import Path

import numpy as np


MOMENT_DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "raw"
    / "2d_dam_break_quadratic_velocity"
    / "moment_models"
)

"""Solve the one-dimensional shallow-water moment systems.

The notation follows the manuscript:

    eps = Hchar/Lchar,
    iRe = nu/(Uchar*Hchar),
    iRe0 = iRe/eps,
    gamma = kappa/(rho*Uchar).

The original source uses the stiff coefficient gamma/eps. The modified
source uses gamma_bar = gamma/(eps + h*gamma/(2*iRe0)), which approaches
2*iRe0/h in the no-slip limit. For the scaled Legendre basis used here,
C_11 = 4, C_22 = 12, and C_12 = C_21 = 0.
"""

# Initial conditions [2/5]
def Initial(Nx, H_left, H_right, var_num, case, Uchar, S=0.25):
    """Initialize ``(h, h*u_m, h*alpha_1, ...)`` for the dam-break test.

    The no-slip manuscript profile is
    ``u(zeta) = S*(2*zeta - zeta**2)``. Its dimensionless moments are
    ``u_m = 2*S/(3*Uchar)``, ``alpha_1 = -S/(2*Uchar)``, and
    ``alpha_2 = -S/(6*Uchar)``.

    The perfect-slip OpenFOAM profile is
    ``u(zeta) = A*(3*zeta**2 - 2*zeta**3)`` with ``A=1/3 m/s``. Its
    projection onto the retained shifted-Legendre basis has
    ``u_m=A/(2*Uchar)``, ``alpha_1=-3*A/(5*Uchar)``, ``alpha_2=0``, and
    ``alpha_3=A/(10*Uchar)``. Thus, the N=3 reconstruction is exact.

    Args:
        Nx (integer): number of computional domain cells
        H_left (float): _description_
        H_right (_type_): _description_
        var_num (integer): 2 + order_moment

    Returns:
        _type_: _description_
    """
    U = np.zeros((Nx, var_num))

    # initial water height
    U[:, 0] = H_right
    U[:Nx // 2, 0] = H_left

    if case in ('init', 'quadratic_no_slip'):
        U[:, 1] = U[:, 0] * (2.0 * S / (3.0 * Uchar))
        if var_num >= 3:
            U[:, 2] = U[:, 0] * (-S / (2.0 * Uchar))
        if var_num >= 4:
            U[:, 3] = U[:, 0] * (-S / (6.0 * Uchar))
        if var_num >= 5:
            U[:, 4] = 0.0
    elif case == 'cubic_perfect_slip':
        amplitude = 1.0 / 3.0
        U[:, 1] = U[:, 0] * (amplitude / (2.0 * Uchar))
        if var_num >= 3:
            U[:, 2] = U[:, 0] * (-3.0 * amplitude / (5.0 * Uchar))
        if var_num >= 4:
            U[:, 3] = 0.0
        if var_num >= 5:
            U[:, 4] = U[:, 0] * (amplitude / (10.0 * Uchar))
    elif case not in ('zero', None):
        raise ValueError(
            "case must be 'quadratic_no_slip', 'cubic_perfect_slip', "
            "'init', 'zero', or None"
        )

    return U

# Calculate the Jacobian matrix [2/4]
def Jacobian_matrix(U, G, model_A='HSWME'):
    """
    Compute the Jacobian matrix in the x-direction for the 1D case.

    Parameters:
    U (list or numpy array): Conserved variables [h, hu, halpha_1, ...]

    Returns:
    matrix_x (numpy array): Jacobian matrix
    """
    #print(U.shape)
    var_num = len(U)  # Determine number of variables dynamically
    order_moment = var_num - 2 # U = (h,hu,halpha_1,...,halpha_N)^{T}
    matrix_x = np.zeros((var_num, var_num))  # Initialize as zero matrix

    # Extract primary variables
    h = U[0]
    hu = U[1]
    u = hu / h

    # If order_number is 0, reduces to the shallow water equation
    # Jacobian matrix in x-direction
    matrix_x[0, 0] = 0.0
    matrix_x[0, 1] = 1.0
    matrix_x[1, 0] = - u ** 2 + G * h
    matrix_x[1, 1] = 2.0 * u

    if order_moment >= 1:
        # Higher-order moment terms
        h_alpha_1 = U[2]
        alpha_1 = h_alpha_1 / h

        # Additional terms for shallow water moment equations
        # pF/pU - Q
        matrix_x[1, 0] += - alpha_1 ** 2 / 3.0
        matrix_x[1, 2] += 2.0 * alpha_1 / 3.0
        matrix_x[2, 0] += - 2.0 * u * alpha_1
        matrix_x[2, 1] += 2.0 * alpha_1
        matrix_x[2, 2] += u

    if order_moment >= 2 and model_A == 'HSWME':
        h_alpha_1 = U[2]
        alpha_1 = h_alpha_1 / h
        #matrix_x[2, 2] += - u
        matrix_x[2, 3] += 3.0*alpha_1/5.0
        matrix_x[3, 0] += - 2.0*alpha_1**2/3.0
        matrix_x[3, 2] += alpha_1/3.0
        matrix_x[3, 3] += u

    if order_moment == 3 and model_A == 'HSWME':
        matrix_x[3, 4] += 4.0*alpha_1/7.0
        matrix_x[4, 3] += 2.0*alpha_1/5.0
        matrix_x[4, 4] += u
    #print("jacobi at each point: ", matrix_x)
    return matrix_x


def effective_gamma_bar(h, eps, gamma, iRe0):
    """Return the effective friction coefficient in manuscript Eq. (45)."""
    return gamma / (eps + h * gamma / (3.0 * iRe0))


def Source(U, eps, gamma, iRe0, model_type, Nx):
    """Evaluate the original or modified manuscript source term."""
    var_num = U.shape[1]
    order_moment = var_num - 2
    SourceU = np.zeros((Nx, var_num))

    h = U[:, 0]
    u = U[:, 1] / h
    friction = gamma / eps
    alpha = U[:, 2:] / h[:, np.newaxis] if order_moment > 0 else None
    alpha_sum = np.sum(alpha, axis=1) if order_moment > 0 else 0.0

    if model_type == 'original':
        SourceU[:, 1] = -friction * (u + alpha_sum)

        if order_moment == 1:
            alpha1 = alpha[:, 0]
            SourceU[:, 2] = -3.0 * (
                friction * (u + alpha1) + 4.0 * iRe0 * alpha1 / h
            )
        elif order_moment == 2:
            alpha1 = alpha[:, 0]
            alpha2 = alpha[:, 1]
            SourceU[:, 2] = -3.0 * (
                friction * (u + alpha1 + alpha2) + 4.0 * iRe0 * alpha1 / h
            )
            SourceU[:, 3] = -5.0 * (
                friction * (u + alpha1 + alpha2) + 12.0 * iRe0 * alpha2 / h
            )
        elif order_moment == 3:
            alpha1 = alpha[:, 0]
            alpha2 = alpha[:, 1]
            alpha3 = alpha[:, 2]
            wall_velocity = u + alpha1 + alpha2 + alpha3
            SourceU[:, 2] = -3.0 * (
                friction * wall_velocity
                + 4.0 * iRe0 * (alpha1 + alpha3) / h
            )
            SourceU[:, 3] = -5.0 * (
                friction * wall_velocity + 12.0 * iRe0 * alpha2 / h
            )
            SourceU[:, 4] = -7.0 * (
                friction * wall_velocity
                + iRe0 * (4.0 * alpha1 + 24.0 * alpha3) / h
            )
    elif model_type == 'modified':
        gamma_bar = effective_gamma_bar(h, eps, gamma, iRe0)
        SourceU[:, 1] = -gamma_bar * u

        if order_moment == 1:
            alpha1 = alpha[:, 0]
            SourceU[:, 2] = -3.0 * (
                gamma_bar * u + 4.0 * iRe0 * alpha1 / h
            )
        elif order_moment == 2:
            alpha1 = alpha[:, 0]
            alpha2 = alpha[:, 1]
            SourceU[:, 2] = -3.0 * (
                gamma_bar * u + 4.0 * iRe0 * alpha1 / h
            )
            SourceU[:, 3] = -5.0 * (
                gamma_bar * u + 12.0 * iRe0 * alpha2 / h
            )
        elif order_moment == 3:
            alpha1 = alpha[:, 0]
            alpha2 = alpha[:, 1]
            alpha3 = alpha[:, 2]
            SourceU[:, 2] = -3.0 * (
                gamma_bar * u
                + 4.0 * iRe0 * (alpha1 + alpha3) / h
            )
            SourceU[:, 3] = -5.0 * (
                gamma_bar * u + 12.0 * iRe0 * alpha2 / h
            )
            SourceU[:, 4] = -7.0 * (
                gamma_bar * u
                + iRe0 * (4.0 * alpha1 + 24.0 * alpha3) / h
            )
    else:
        raise ValueError("model_type must be either 'original' or 'modified'.")

    return SourceU


def _modified_backward_euler_step(U, RHS, dt, eps, gamma, iRe0):
    """Apply transport followed by backward Euler for the modified source."""
    transported = U + RHS
    interior = transported[1:-1]
    height = interior[:, 0]
    if np.any(height <= 0.0):
        raise ValueError("the backward-Euler source step requires positive depth")

    order_moment = U.shape[1] - 2
    block_size = order_moment + 1
    matrices = np.broadcast_to(
        np.eye(block_size), (height.size, block_size, block_size)
    ).copy()

    gamma_bar = effective_gamma_bar(height, eps, gamma, iRe0)
    wall_scale = dt * gamma_bar / height
    matrices[:, 0, 0] += wall_scale

    if order_moment > 0:
        indices = np.arange(1, order_moment + 1, dtype=int)
        row_indices, column_indices = np.meshgrid(
            indices, indices, indexing="ij"
        )
        minimum_indices = np.minimum(row_indices, column_indices).astype(float)
        derivative_inner_products = np.where(
            (row_indices + column_indices) % 2 == 0,
            2.0 * minimum_indices * (minimum_indices + 1.0),
            0.0,
        )
        viscous_scale = dt * iRe0 / height**2
        for moment_index in range(order_moment):
            row = moment_index + 1
            factor = 2.0 * row + 1.0
            matrices[:, row, 0] += factor * wall_scale
            matrices[:, row, 1:] += (
                factor
                * viscous_scale[:, np.newaxis]
                * derivative_inner_products[moment_index, :]
            )

    U[:, 0] = transported[:, 0]
    U[1:-1, 1:] = np.linalg.solve(
        matrices, interior[:, 1:, np.newaxis]
    )[:, :, 0]
    return U

# Compute the maximum eigenvalues
def MaxEigenValues(U, G):
    """
    Compute the maximum eigenvalue in x-direction (pointwise).

    Args:
        U (_type_): _description_
        G (float): inverse Froude number.

    Returns:

    """
    var_nums = len(U)
    order_moment = var_nums - 2 # U = [h,hu,halpha1,...,halphaN]

    h = U[0]
    if np.any(h <= 0):
        raise ValueError(f"Fluid height h must be positive, got {h}")

    u = U[1]/h
    alpha_1 = np.zeros_like(h)

    if order_moment > 0:
        alpha_1 = U[2]/h # The third element is halpha1

    max_eigenvalue_x = np.abs(u) + np.sqrt(G*h + alpha_1**2)
    return max_eigenvalue_x


# Evaluate Roe scheme
def RoeScheme_1d(U, G, Nx, dx, dt):
    """_summary_

    Args:
        U (_type_): _description_
        RHS (_type_): _description_
        Nx (_type_): _description_
        dx (_type_): _description_
        dt (_type_): _description_
    """
    Nx, var_nums = U.shape
    RHS = np.zeros_like(U)

    # Compute Roe averages for each interface
    U_roe_x = np.zeros_like(U)
    max_wavespeed_x = np.zeros(Nx)
    # Compute the Roe averages in the x direction
    for i in range(Nx):
        iplus = i+1 if i+1 < Nx else Nx-1 # boundary condition
        U_roe_x[i,:] = 0.5*(U[i,:] + U[iplus,:])

        # Compute max eigenvalues for each interface
        max_wavespeed_x[i] = MaxEigenValues(U_roe_x[i,:], G)
    #print("max wave speed shape: ", max_wavespeed_x.shape)
    # Compute the Roe matrix in x direction
    # Roe matrix has shape Nx * var_num * var_num
    # each J has shape num_bar * var_num
    Jacobi =[]
    #print("Inverse Froude number G: ", G)
    for i in range(0,Nx):
        J = Jacobian_matrix(U_roe_x[i,:], G)
        #print("Jacobi shape: \n", J.shape)
        Jacobi.append(J)
    #print("U_roe_x for compute Jacobi: ", U_roe_x)
    #print("Jacobi one step: ", Jacobi)
    # Update RHS based on Roe Scheme
    for i in range(1,Nx-1):
        # only compute interior points
        iminus = i-1
        iplus  = i+1
        #iminus = i-1 if i-1 > 0 else 0
        #iplus = i+1 if i+1 < Nx-1 else Nx-1

        for j in range(var_nums):
            matrix_contrib = 0.0
            for k in range(var_nums):
                # Contribution from the left interface (i-1/2)
                term_left = Jacobi[iminus][j][k]*(U[i][k]-U[iminus][k])
                # Contribution from the right interface (i+1/2)
                term_right = Jacobi[i][j][k]*(U[iplus][k]-U[i][k])
                #print("left shape: ", term_left.shape)
                #print("right shape: ", term_right.shape)
                matrix_contrib += -dt/(2*dx)*(term_left + term_right)

            # Contribution from wave speeds
            delta_left = U[i][j] - U[iminus][j]
            delta_right = U[iplus][j] - U[i][j]
            wave_contrib = - dt/(2*dx)*(max_wavespeed_x[iminus]*delta_left \
                                        - max_wavespeed_x[i]*delta_right)
            #print("matrix contrib shape: ", matrix_contrib.shape)
            #print("wave contrib shape: ", wave_contrib.shape)
            RHS[i][j] = matrix_contrib + wave_contrib

    return RHS

# Main solver
def SWME_Solver(Uchar, H_left, H_right, L, G, Nx, CFL, time_end,
                gamma, iRe0, eps, model_type, var_num, t_target,
                non_zero_initial, store_every_step=True):
    """Advance a manuscript SWE/HSWME or modified model in time.

    The original and modified source terms are treated with backward Euler.

    If ``store_every_step`` is false, ``U_history`` contains only the initial
    state and states at requested target times (plus the final state when it
    is not already a target). ``dt_history`` always contains every numerical
    time step, so callers using sparse storage should save target times
    explicitly rather than reconstructing them from ``dt_history``.
    """
    dx = L/Nx # mesh size

    # Initialization
    U = Initial(Nx, H_left, H_right, var_num, non_zero_initial, Uchar)

    U_history = [U.copy()]
    dt_history = []

    # time integration
    t_curr = 0.0
    time_step = 0

    # target time instances
    #t_target = [1.0, 2.0, time_end]
    target_idx = 0
    while t_curr < time_end:
        wavespeed_x = np.zeros((Nx,1))
        for i in range(Nx):
            wavespeed_x[i] = MaxEigenValues(U[i,:], G)

        max_wavespeed_x = np.max(wavespeed_x)
        # time step
        dt = CFL*dx/max_wavespeed_x
        dt = min(dt, time_end-t_curr)

        # ensure given time isntances picked, for better data illustration
        target_hit = False
        if target_idx < len(t_target) and dt >= t_target[target_idx] - t_curr:
            dt = t_target[target_idx] - t_curr
            t_curr = t_target[target_idx]
            time_step += 1
            target_idx += 1
            target_hit = True
        else:
            t_curr += dt
            time_step += 1

        dt_history.append(dt)

        RHS = RoeScheme_1d(U, G, Nx, dx, dt)
        #if time_step == 0:
        #    print("max wave speed: ", max_wavespeed_x)
        #    break
        SourceU = Source(U, eps, gamma, iRe0, model_type, Nx)
        friction = gamma / eps
        if var_num == 2 and model_type == 'original':
            U[:,0] += RHS[:,0]
            U[:,1] = (U[:,1]+RHS[:,1])/(1+dt*friction/U[:,0])
        elif var_num == 3 and model_type == "original":
            # update hu and halpha1
            h = U[:,0]
            for i in range(1,Nx-1):
                U[i,0] += RHS[i,0]+(SourceU[i,0])*dt

                a11 = 1 + dt * friction / h[i]
                a12 = dt * friction / h[i]
                a21 = 3.0 * dt * friction / h[i]
                a22 = (
                    1.0 + 3.0 * dt * friction / h[i]
                    + 12.0 * dt * iRe0 / h[i]**2
                )
                det = a22*a11 - a12*a21

                f1 = U[i,1] + RHS[i,1]
                f2 = U[i,2] + RHS[i,2]
                U[i,1] = (f1*a22-f2*a12)/det
                U[i,2] = (a11*f2-a21*f1)/det
        elif var_num == 4 and model_type == 'original':
            h = U[:,0]
            for i in range(1,Nx-1):
                U[i,0] += RHS[i,0]+SourceU[i,0]*dt

                a = dt * friction / h[i]
                b = dt * iRe0 / h[i]**2
                A = np.array([[1+a, a, a],
                              [3*a, 1+3*a+12*b, 3*a],
                              [5*a, 5*a, 1+5*a+60*b]])
                f = np.array([U[i,1:]+RHS[i,1:]]).reshape(3,1)
                #print(A.shape, f.shape)
                uh = np.linalg.solve(A, f)
                U[i,1] = uh[0]
                U[i,2] = uh[1]
                U[i,3] = uh[2]

        elif var_num == 5 and model_type == 'original':
            h = U[:,0]
            derivative_inner_products = np.array(
                [[4.0, 0.0, 4.0],
                 [0.0, 12.0, 0.0],
                 [4.0, 0.0, 24.0]]
            )
            for i in range(1,Nx-1):
                U[i,0] += RHS[i,0]+SourceU[i,0]*dt

                a = dt * friction / h[i]
                b = dt * iRe0 / h[i]**2
                A = np.eye(4)
                A[0,:] += a
                for moment_index in range(3):
                    factor = 2.0 * (moment_index + 1) + 1.0
                    A[moment_index + 1,:] += factor * a
                    A[moment_index + 1,1:] += (
                        factor * b * derivative_inner_products[moment_index,:]
                    )
                f = U[i,1:] + RHS[i,1:]
                U[i,1:] = np.linalg.solve(A, f)

        elif model_type == 'modified':
            _modified_backward_euler_step(U, RHS, dt, eps, gamma, iRe0)


        # Apply boundary condition
        U[0,:]  = U[1,:]
        U[-1,:] = U[-2,:]

        if (
            store_every_step
            or target_hit
            or np.isclose(t_curr, time_end, rtol=0.0, atol=1.0e-14)
        ):
            U_history.append(U.copy())

        if time_step%100 == 0:
            print('-------------------------------------------------------------')
            print(f"Max wavespeed in x direction: {max_wavespeed_x:8.2e}; \n"
                    f"Current time: {t_curr:8.2e}; dt: {dt:8.2e}; time step: {time_step}\n"
                    f"velocity u: {np.max(np.abs(U[:,1]/U[:,0]))}")
            if var_num == 3:
                print(f"alpha1 {np.max(np.abs(U[:,2]/U[:,0]))}")
            #print(f"velocity: {U[:,1]}")
        #if time_step == 5:
        #    break
        

    #print("U history length: ", len(U_history))
    #print(" U shape : ", U_history[0].shape)
    print(f"Max wavespeed in x direction: {max_wavespeed_x:8.2e}; \n"
                    f"Current time: {t_curr:8.2e}; dt: {dt:8.2e}; time step: {time_step}\n"
                    f"velocity u: {np.max(np.abs(U[:,1]/U[:,0]))}")
    #if var_num == 3:
    #    print(f"alpha1 {np.max(np.abs(U[:,2]/U[:,0]))}")
    return U_history, dt_history


# Function to save data
def save_data(model_type, kappa, var_num, U_history, dt_history, dx, time_end, L):
    """Save one model result in the repository's raw moment-model data folder."""
    model_names = {
        (2, 'original'): "SWE",
        (2, 'modified'): "M-SWE",
        (3, 'original'): "SWME",
        (3, 'modified'): "M-SWME",
        (4, 'original'): "SWME2",
        (4, 'modified'): "M-SWME2",
        (5, 'original'): "SWME3",
        (5, 'modified'): "M-SWME3",
    }
    try:
        model_name = model_names[(var_num, model_type)]
    except KeyError as exc:
        raise ValueError(
            "Expected var_num in {2, 3, 4, 5} and model_type in "
            "{'original', 'modified'}."
        ) from exc

    MOMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    kappa_label = f"{kappa:.1e}"
    file_path = MOMENT_DATA_DIR / f"{model_name}_data_kappa{kappa_label}.npy"
    np.save(file_path, {'U_history': U_history, 'dt_history': dt_history,
                        'dx': dx, 'time_end': time_end, 'L': L,
                        'kappa': kappa})
    print(f"Data saved in {file_path}")
    
