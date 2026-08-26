import numpy as np
import os
"""Solve the two-dimensional shallow-water moment systems.

The dimensionless source parameters follow the manuscript notation:

    epsilon = Hchar/Lchar,
    iRe = nu/(Uchar*Hchar) = epsilon*iRe0,
    gamma = kappa/(rho*Uchar).

The classical source uses the wall coefficient ``gamma/epsilon``.  The
modified source uses

    gamma_bar = gamma/(epsilon*(1 + h*gamma/(3*iRe))).

For the scaled Legendre basis retained here, ``C_11=4``, ``C_22=12``,
and ``C_12=C_21=0``.  Both the original and modified linear sources are
advanced with the same backward-Euler solve.
"""

def _linear_profile_state(height, var_num, U1, H1, velocity_slope):
    """Return the conserved state for ``u(z)=v(z)=velocity_slope*z``."""
    state = np.zeros(np.shape(height) + (var_num,))
    state[..., 0] = height

    # With z=H1*h*zeta and phi_1=1-2*zeta,
    # S*z = S*H1*h*(1-phi_1)/2.  Hence
    # u_m=v_m=S*H1*h/(2*U1) and alpha_1=beta_1=-u_m.
    mean_discharge = 0.5 * velocity_slope * H1 / U1 * np.asarray(height)**2
    state[..., 1] = mean_discharge
    state[..., 2] = mean_discharge
    if var_num >= 5:
        state[..., 3] = -mean_discharge
        state[..., 4] = -mean_discharge
    return state


# Initial conditions [3/4/25]
def Initial(
    Nx, Ny, dx, dy, H_center, H_rest, var_num, U1, H1,
    init_velocity, velocity_slope=0.25,
):
    """Initialize the cylindrical height jump and prescribed velocity profile.

    The linear-profile case uses ``u(0,x,y,z)=v(0,x,y,z)=S*z`` throughout
    the water domain, including the ambient-depth region outside the initial
    cylinder.  The profile is represented exactly for every model with at
    least one retained moment.
    """
    if var_num not in (3, 5, 7):
        raise ValueError("var_num must be 3, 5, or 7")

    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    inside_cylinder = (
        (x_grid - 0.5)**2 + (y_grid - 0.5)**2 <= 0.15**2
    )
    height = np.where(inside_cylinder, H_center, H_rest)

    if init_velocity in ('linear', 'nonzero'):
        return _linear_profile_state(
            height, var_num, U1, H1, velocity_slope
        )
    if init_velocity == 'zero':
        U = np.zeros((Nx, Ny, var_num))
        U[:, :, 0] = height
        return U
    raise ValueError("init_velocity must be 'linear' or 'zero'")

# Calculate the Jacobian matrix [2/4]
def Jacobian_matrix(U, g, model_A='HSWME'):
    """
    Compute the Jacobian matrix in the x-direction for the 1D case.

    Parameters:
    U (list or numpy array): Conserved variables [h, hu, halpha_1, ...]

    Returns:
    matrix_x (numpy array): Jacobian matrix
    """
    #print('check U shape in computing Jacobian: ', U.shape)
    var_num = len(U)  # Determine number of variables dynamically
    order_moment = (var_num - 3)/2 # U = (h,hu,hv,halpha_1,hbeta_1,...,halpha_N, hbeta_N)^{T}
    matrix_x = np.zeros((var_num, var_num))  # Initialize as zero matrix
    matrix_y = np.zeros((var_num, var_num))

    # Extract primary variables
    h, hu, hv = U[0], U[1], U[2]
    u, v = hu/h, hv/h

    # If order_number is 0, reduces to the shallow water equation
    # Jacobian matrix in x-direction
    matrix_x[0, 1] = 1.0
    matrix_x[1, 0] = - u**2 + g*h
    matrix_x[1, 1] = 2.0*u
    matrix_x[2, 0] = - u*v
    matrix_x[2, 1] = v
    matrix_x[2, 2] = u

    # Jacobian matrix in y-direction
    matrix_y[0, 2] = 1.0
    matrix_y[1, 0] = -u*v
    matrix_y[1, 1] = v
    matrix_y[1, 2] = u
    matrix_y[2, 0] = -v**2 + g*h
    matrix_y[2, 2] = 2.0*v

    if order_moment >= 1:
        # Higher-order moment terms
        h_alpha_1, h_beta_1 = U[3], U[4]
        alpha_1, beta_1 = h_alpha_1/h, h_beta_1/h

        # Additional terms for shallow water moment equations
        # pF/pU - Q
        matrix_x[1, 0] += -alpha_1**2/3.0
        matrix_x[1, 3] += 2.0/3.0*alpha_1
        matrix_x[2, 0] += -1.0/3.0*alpha_1*beta_1
        matrix_x[2, 3] += 1.0/3.0*beta_1
        matrix_x[2, 4] += 1.0/3.0*alpha_1
        matrix_x[3, 0] += -2.0*u*alpha_1
        matrix_x[3, 1] += 2.0*alpha_1
        matrix_x[3, 3] += u
        matrix_x[4, 0] += -u*beta_1 - v*alpha_1
        matrix_x[4, 1] += beta_1
        matrix_x[4, 2] += alpha_1
        matrix_x[4, 4] += u

        matrix_y[1, 0] += -1.0/3.0*alpha_1*beta_1
        matrix_y[1, 3] += 1.0/3.0*beta_1
        matrix_y[1, 4] += 1.0/3.0*alpha_1
        matrix_y[2, 0] += -1.0/3.0*beta_1**2
        matrix_y[2, 4] += 2.0/3.0*beta_1
        matrix_y[3, 0] += -u*beta_1-v*alpha_1
        matrix_y[3, 1] += beta_1
        matrix_y[3, 2] += alpha_1
        matrix_y[3, 3] += v
        matrix_y[4, 0] += -2.0*v*beta_1
        matrix_y[4, 2] += 2.0*beta_1
        matrix_y[4, 4] += v


    if order_moment == 2 and model_A == 'HSWME':
        h_alpha_1, h_beta_1 = U[3], U[4]
        alpha_1, beta_1 = h_alpha_1/h, h_beta_1/h
        #h_alpha_2, h_beta_2 = U[5], U[6]
        #alpha_2, beta_2 = h_alpha_2/h, h_beta_2/h

        matrix_x[3, 5] += 3.0/5.0*alpha_1
        matrix_x[4, 5] += 1.0/5.0*beta_1
        matrix_x[4, 6] += 2.0/5.0*alpha_1
        matrix_x[5, 0] += -2.0/3.0*alpha_1**2
        matrix_x[5, 3] += 1.0/3.0*alpha_1
        matrix_x[5, 5] += u
        matrix_x[6, 0] += -2.0/3.0*alpha_1*beta_1
        matrix_x[6, 3] += -1.0/3.0*beta_1
        matrix_x[6, 4] += 2.0/3.0*alpha_1
        matrix_x[6, 6] += u

        matrix_y[3, 5] += 2.0/5.0*beta_1
        matrix_y[3, 6] += 1.0/5.0*alpha_1
        matrix_y[4, 6] += 3.0/5.0*beta_1
        matrix_y[5, 0] += -2.0/3.0*alpha_1*beta_1
        matrix_y[5, 3] += 2.0/3.0*beta_1
        matrix_y[5, 4] += -1.0/3.0*alpha_1
        matrix_y[5, 5] += v
        matrix_y[6, 0] += -2.0/3.0*beta_1**2
        matrix_y[6, 4] += 1.0/3.0*beta_1
        matrix_y[6, 6] += v

    return matrix_x, matrix_y


def effective_gamma_bar(h, epsilon, gamma, iRe):
    """Return the effective wall coefficient from manuscript Eq. (45)."""
    return gamma / (epsilon * (1.0 + h * gamma / (3.0 * iRe)))


def _derivative_inner_products(order_moment):
    """Return ``C_ij`` for the supported scaled-Legendre orders."""
    if order_moment == 0:
        return np.zeros((0, 0))
    if order_moment == 1:
        return np.array([[4.0]])
    if order_moment == 2:
        return np.array([[4.0, 0.0], [0.0, 12.0]])
    raise ValueError("only moment orders N=0, 1, and 2 are supported")


# Calculate the source term
def Source(U, epsilon, gamma, iRe, model_type='modified'):
    """Evaluate the original or modified manuscript source term."""
    Nx, Ny, var_num = U.shape
    order_moment = int((var_num - 3) / 2)
    if var_num != 3 + 2 * order_moment:
        raise ValueError("state must contain paired x- and y-moment variables")

    derivative_inner_products = _derivative_inner_products(order_moment)
    SourceU = np.zeros((Nx, Ny, var_num))

    h = U[:, :, 0]
    u = U[:, :, 1] / h
    v = U[:, :, 2] / h
    alpha = U[:, :, 3::2] / h[:, :, np.newaxis]
    beta = U[:, :, 4::2] / h[:, :, np.newaxis]

    if model_type == 'original':
        wall_coefficient = gamma / epsilon
        wall_u = u + np.sum(alpha, axis=2)
        wall_v = v + np.sum(beta, axis=2)
        SourceU[:, :, 1] = -wall_coefficient * wall_u
        SourceU[:, :, 2] = -wall_coefficient * wall_v
    elif model_type == 'modified':
        wall_coefficient = effective_gamma_bar(h, epsilon, gamma, iRe)
        wall_u = u
        wall_v = v
        SourceU[:, :, 1] = -wall_coefficient * u
        SourceU[:, :, 2] = -wall_coefficient * v
    else:
        raise ValueError("model_type must be either 'original' or 'modified'")

    if order_moment > 0:
        viscous_alpha = np.einsum(
            'ij,xyj->xyi', derivative_inner_products, alpha
        )
        viscous_beta = np.einsum(
            'ij,xyj->xyi', derivative_inner_products, beta
        )
        viscous_coefficient = iRe / (epsilon * h)
        for moment_index in range(order_moment):
            factor = 2.0 * (moment_index + 1) + 1.0
            SourceU[:, :, 3 + 2 * moment_index] = -factor * (
                wall_coefficient * wall_u
                + viscous_coefficient * viscous_alpha[:, :, moment_index]
            )
            SourceU[:, :, 4 + 2 * moment_index] = -factor * (
                wall_coefficient * wall_v
                + viscous_coefficient * viscous_beta[:, :, moment_index]
            )

    return SourceU


def _implicit_source_matrices(
    height, dt, epsilon, gamma, iRe, order_moment, model_type
):
    """Build the backward-Euler source matrices for a batch of cells."""
    derivative_inner_products = _derivative_inner_products(order_moment)
    block_size = order_moment + 1
    matrices = np.broadcast_to(
        np.eye(block_size), (height.size, block_size, block_size)
    ).copy()

    if model_type == 'original':
        wall_coefficient = np.full_like(height, gamma / epsilon)
        wall_couples_moments = True
    elif model_type == 'modified':
        wall_coefficient = effective_gamma_bar(height, epsilon, gamma, iRe)
        wall_couples_moments = False
    else:
        raise ValueError("model_type must be either 'original' or 'modified'")

    wall_scale = dt * wall_coefficient / height
    viscous_scale = dt * iRe / (epsilon * height**2)

    if wall_couples_moments:
        matrices[:, 0, :] += wall_scale[:, np.newaxis]
    else:
        matrices[:, 0, 0] += wall_scale

    for moment_index in range(order_moment):
        row = moment_index + 1
        factor = 2.0 * row + 1.0
        matrices[:, row, 0] += factor * wall_scale
        if wall_couples_moments:
            matrices[:, row, 1:] += factor * wall_scale[:, np.newaxis]
        matrices[:, row, 1:] += (
            factor
            * viscous_scale[:, np.newaxis]
            * derivative_inner_products[moment_index, :]
        )

    return matrices


def implicit_source_step(U, RHS, dt, epsilon, gamma, iRe, model_type):
    """Apply transport followed by one backward-Euler source substep."""
    Nx, Ny, var_num = U.shape
    order_moment = int((var_num - 3) / 2)
    if var_num != 3 + 2 * order_moment:
        raise ValueError("state must contain paired x- and y-moment variables")

    transported = U + RHS
    interior = transported[1:-1, 1:-1, :]
    height = interior[:, :, 0].reshape(-1)
    if np.any(height <= 0.0):
        raise ValueError("the implicit source step requires positive water depth")

    matrices = _implicit_source_matrices(
        height, dt, epsilon, gamma, iRe, order_moment, model_type
    )
    x_indices = np.arange(1, var_num, 2)
    y_indices = np.arange(2, var_num, 2)
    right_hand_side = np.stack(
        (
            interior[:, :, x_indices].reshape(-1, order_moment + 1),
            interior[:, :, y_indices].reshape(-1, order_moment + 1),
        ),
        axis=2,
    )
    solved = np.linalg.solve(matrices, right_hand_side)

    U[:, :, 0] = transported[:, :, 0]
    U[1:-1, 1:-1, x_indices] = solved[:, :, 0].reshape(
        Nx - 2, Ny - 2, order_moment + 1
    )
    U[1:-1, 1:-1, y_indices] = solved[:, :, 1].reshape(
        Nx - 2, Ny - 2, order_moment + 1
    )
    return U


def apply_boundary_conditions(
    U, H_rest, U1, H1, init_case, velocity_slope=0.25
):
    """Apply the boundary conditions for the selected collapse problem."""
    if init_case in ('linear', 'nonzero'):
        # Outflow at x=1 and y=1.
        U[-1, :, :] = U[-2, :, :]
        U[:, -1, :] = U[:, -2, :]

        # Fixed inflow at x=0 and y=0: dimensional depth H1*H_rest=1 m
        # and u(z)=v(z)=S*z.  Apply these last so the two inflow corners are
        # not overwritten by an outflow copy.
        inflow_state = _linear_profile_state(
            H_rest, U.shape[2], U1, H1, velocity_slope
        )
        U[0, :, :] = inflow_state
        U[:, 0, :] = inflow_state
    elif init_case == 'zero':
        U[0, :, :] = U[1, :, :]
        U[-1, :, :] = U[-2, :, :]
        U[:, 0, :] = U[:, 1, :]
        U[:, -1, :] = U[:, -2, :]
    else:
        raise ValueError("init_case must be 'linear' or 'zero'")
    return U


# Compute the maximum eigenvalues
def MaxEigenValues(U, g):
    """
    Compute the maximum eigenvalue in x-direction (pointwise).

    Args:
        U (_type_): _description_
        g (_type_): _description_

    Returns:

    """
    #print("check U shape: ", U.shape)
    var_nums = len(U)
    order_moment = int((var_nums - 3)/2) # U = [h,hu,halpha1,...,halphaN]

    h = U[0]
    if h.any() <= 0:
        raise ValueError(f"Fluid height h mush be positive, got {h}")

    u, v = U[1]/h, U[2]/h
    alpha_1 = np.zeros_like(h)
    beta_1 = np.zeros_like(h)

    if order_moment > 0:
        alpha_1 = U[3]/h # The third element is halpha1
        beta_1 = U[4]/h

    max_eigenvalue_x = np.abs(u) + np.sqrt(g*h + alpha_1**2)
    max_eigenvalue_y = np.abs(v) + np.sqrt(g*h + beta_1**2)

    return max_eigenvalue_x, max_eigenvalue_y


# Evaluate Roe scheme
def RoeScheme_2d(U, g, Nx, Ny, dx, dy, dt):
    """_summary_

    Args:
        U (_type_): _description_
        RHS (_type_): _description_
        Nx (_type_): _description_
        dx (_type_): _description_
        dt (_type_): _description_
    """
    Nx, Ny, var_nums = U.shape
    RHS = np.zeros_like(U)

    # Compute Roe averages for each interface
    U_roe_x = np.zeros_like(U)
    U_roe_y = np.zeros_like(U)
    max_wavespeed_x = np.zeros((Nx, Ny))
    max_wavespeed_y = np.zeros((Nx, Ny))
    # Compute the Roe averages in the x direction
    for i in range(Nx):
        iplus = i+1 if i+1 < Nx else Nx-1 # boundary condition
        for j in range(Ny):
            U_roe_x[i,j,:] = 0.5*(U[i,j,:] + U[iplus,j,:])
            max_wavespeed_x[i,j], _ = MaxEigenValues(U_roe_x[i,j,:], g) # omit the second output

            jplus = j+1 if j+1 < Ny else Ny-1
            U_roe_y[i,j,:] = 0.5*(U[i,j,:]+U[i,jplus,:])
            _, max_wavespeed_y[i,j] = MaxEigenValues(U_roe_y[i,j,:], g)
    #print("Check if U_roe_x is symmetric: ")
    #print("h: ",  np.array_equal(U_roe_x[:,:,0], U_roe_x[:,:,0].T))
    #print("hu: ", np.array_equal(U_roe_x[:,:,1], U_roe_x[:,:,1].T))
    #print("hv: ", np.array_equal(U_roe_x[:,:,2], U_roe_x[:,:,2].T))
    #print("Check if U_roe_y is symmetric: ")
    #print("h: ",  np.array_equal(U_roe_y[:,:,0], U_roe_y[:,:,0].T))
    #print("hu: ", np.array_equal(U_roe_y[:,:,1], U_roe_y[:,:,1].T))
    #print("hv: ", np.array_equal(U_roe_y[:,:,2], U_roe_y[:,:,2].T))
    #U_roe_x and max_wavespeed_x are symmetric w.r.t. y=0
    #U_roe_y and max_wavespeed_y are symmetric w.r.t. x=0
    #print("max wave speed shape: ", max_wavespeed_x.shape)
    # Compute the Roe matrix in x direction
    # Roe matrix has shape Nx * Ny * var_num * var_num
    # each J has shape num_bar * var_num
    Jacobi_x =[]
    #print("Coe g (G): ", g)
    for i in range(0, Nx):
        for j in range(0, Ny):
            J, _ = Jacobian_matrix(U_roe_x[i,j,:], g)
            #print("Jacobi shape: \n", J.shape)
            Jacobi_x.append(J)
    # check the Jacobian matrix for each mesh node
    #print(f"start of Jacobian matrix ------------------------------------")
    #for j in range(Ny):
    #    print(f"Jacobian matrix at {0,j}-th node: \n", Jacobi_x[j])
    #print(f"end of Jacobian matrix --------------------------------------")
    
    # Update RHS based on Roe Scheme
    for i in range(1,Nx-1):
        # only compute interior points
        iminus = i-1
        iplus  = i+1
        #iminus = i-1 if i-1 > 0 else 0
        #iplus = i+1 if i+1 < Nx-1 else Nx-1
        for j in range(1,Nx-1):
            for m in range(var_nums):
                matrix_contrib = 0.0
                for n in range(var_nums):
                    #print("Check Jacobi_x shape: ", len(Jacobi_x[1]))
                    # Contribution from the left interface (i-1/2)
                    term_left = Jacobi_x[iminus*Ny+j][m][n]*(U[i][j][n]-U[iminus][j][n])
                    # Contribution from the right interface (i+1/2)
                    term_right = Jacobi_x[i*Ny+j][m][n]*(U[iplus][j][n]-U[i][j][n])
                    #print("left shape: ", term_left.shape)
                    #print("right shape: ", term_right.shape)
                    matrix_contrib += -dt/(2*dx)*(term_left + term_right)

                # Contribution from wave speeds
                delta_left = U[i][j][m] - U[iminus][j][m]
                delta_right = U[iplus][j][m] - U[i][j][m]
                #print("Check max_wavespeed_x shape: ", max_wavespeed_x.shape)
                #print("Check: delta_left, delta_right: ", delta_left, delta_right)
                wave_contrib = - dt/(2*dx)*(max_wavespeed_x[iminus][j]*delta_left \
                                        - max_wavespeed_x[i][j]*delta_right)
                #print("matrix contrib shape: ", matrix_contrib.shape)
                #print("wave contrib shape: ", wave_contrib.shape)
                RHS[i][j][m] += matrix_contrib + wave_contrib
    Jacobi_y =[]
    #print("Coe g (G): ", g)
    for i in range(0, Nx):
        for j in range(0, Ny):
            _, J = Jacobian_matrix(U_roe_y[i,j,:], g)
            #print("Jacobi shape: \n", J.shape)
            Jacobi_y.append(J)
    # Update RHS based on Roe Scheme
    for i in range(1,Nx-1):
        # only compute interior points
        #iminus = i-1
        #iplus  = i+1
        #iminus = i-1 if i-1 > 0 else 0
        #iplus = i+1 if i+1 < Nx-1 else Nx-1
        for j in range(1,Nx-1):
            jminus = j-1
            jplus  = j+1
            for m in range(var_nums):
                matrix_contrib = 0.0
                for n in range(var_nums):
                    # Contribution from the left interface (i-1/2)
                    term_left = Jacobi_y[i*Ny+jminus][m][n]*(U[i][j][n]-U[i][jminus][n])
                    # Contribution from the right interface (i+1/2)
                    term_right = Jacobi_y[i*Ny+j][m][n]*(U[i][jplus][n]-U[i][j][n])
                    #print("left shape: ", term_left.shape)
                    #print("right shape: ", term_right.shape)
                    matrix_contrib += -dt/(2*dy)*(term_left + term_right)

                # Contribution from wave speeds
                delta_left = U[i][j][m] - U[i][jminus][m]
                delta_right = U[i][jplus][m] - U[i][j][m]
                wave_contrib = - dt/(2*dy)*(max_wavespeed_y[i][jminus]*delta_left \
                                        - max_wavespeed_y[i][j]*delta_right)
                #print("matrix contrib shape: ", matrix_contrib.shape)
                #print("wave contrib shape: ", wave_contrib.shape)
                RHS[i][j][m] += matrix_contrib + wave_contrib

    return RHS

# Main solver
def SWME_Solver(U1, H1, H_center, H_rest, L, G, Nx, Ny, CFL, time_end,
                gamma, iRe, epsilon, model_type, var_num, t_target, init_case,
                velocity_slope=0.25):
    """Advance an original or modified 2D moment model in time.

    The transport step is explicit.  For every model and retained order, the
    linear wall-friction and vertical-viscosity source is advanced with the
    common backward-Euler solver implemented in :func:`implicit_source_step`.
    Only states at ``t_target`` are retained in ``U_history``.  The returned
    ``dt_history`` contains increments between those target times, so
    ``np.cumsum(dt_history)`` gives the saved-time array.
    """
    dx = L/Nx # mesh size
    dy = L/Ny
    # Initialization
    U = Initial(
        Nx, Ny, dx, dy, H_center, H_rest, var_num, U1, H1,
        init_velocity=init_case, velocity_slope=velocity_slope,
    )

    #print("Check if initialization is symmetric: ")
    #print("h: ",  np.array_equal(U[:,:,0], U[:,:,0].T))
    #print("hu: ", np.array_equal(U[:,:,1], U[:,:,1].T))
    #print("hv: ", np.array_equal(U[:,:,2], U[:,:,2].T))


    target_times = np.asarray(t_target, dtype=float)
    if target_times.ndim != 1 or target_times.size == 0:
        raise ValueError("t_target must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(target_times)):
        raise ValueError("t_target must contain only finite values")
    if np.any(np.diff(target_times) <= 0.0):
        raise ValueError("t_target must be strictly increasing")
    if target_times[0] < 0.0 or target_times[-1] > time_end:
        raise ValueError("t_target values must lie in [0, time_end]")

    U_history = []
    saved_times = []

    # time integration
    T0 = 0.0
    t_curr = 0.0
    time_step = 0

    target_idx = 0
    if target_times[0] == 0.0:
        U_history.append(U.copy())
        saved_times.append(0.0)
        target_idx = 1

    while t_curr < time_end:
        wavespeed_x = np.zeros((Nx,Ny))
        wavespeed_y = np.zeros((Nx,Ny))
        for i in range(Nx):
            for j in range(Ny):
                wavespeed_x[i,j], wavespeed_y[i,j] = MaxEigenValues(U[i,j,:], G)

        max_wavespeed_x = np.max(wavespeed_x)
        max_wavespeed_y = np.max(wavespeed_y)
        # time step
        dt = CFL/(max_wavespeed_x/dx + max_wavespeed_y/dy)
        dt = min(dt, time_end-t_curr)

        # Shorten the adaptive step when necessary so target times are reached
        # exactly.  Do not retain intermediate adaptive steps.
        reaches_target = False
        if target_idx < target_times.size:
            time_to_target = target_times[target_idx] - t_curr
            if dt >= time_to_target:
                dt = time_to_target
                reaches_target = True

        t_curr += dt
        time_step += 1
        print(f"t: {t_curr:.3f}")

        RHS = RoeScheme_2d(U, G, Nx, Ny, dx, dy, dt) #RoeScheme_1d(U, G, Nx, dx, dt)
        #print("Check if RHS is symmetric: ")
        #print("h: ",  np.array_equal(RHS[:,:,0], RHS[:,:,0].T))
        #print("hu: ", np.array_equal(RHS[:,:,1], RHS[:,:,1].T))
        #print("hv: ", np.array_equal(RHS[:,:,2], RHS[:,:,2].T))
        '''
        # [3/11/25] RHS not symmetric!!!
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        h_history = RHS[:,:,0]
        print("plot h shape: ", h_history.shape)
        L1 = U1
        x = np.linspace(0, L1*L, Nx)
        y = np.linspace(0, L1*L, Ny)
        X, Y = np.meshgrid(x, y)
        ax.plot_wireframe(X, Y, H1*h_history)
        plt.title(f"RHS t={t_curr}")
        #plt.show()
        '''
        #if time_step == 0:
        #    print("max wave speed: ", max_wavespeed_x)
        #    break
        implicit_source_step(
            U, RHS, dt, epsilon, gamma, iRe, model_type
        )
        #print("h: ",  np.array_equal(SourceU[:,:,0], SourceU[:,:,0].T))
        #print("hu: ", np.array_equal(SourceU[:,:,1], SourceU[:,:,1].T))
        #print("hv: ", np.array_equal(SourceU[:,:,2], SourceU[:,:,2].T))
        apply_boundary_conditions(
            U, H_rest, U1, H1, init_case, velocity_slope
        )
        #print(f"water height slice at t={t_curr}: \n", U[0,:,0])
        #print(f"water height slice at t={t_curr}: \n", U[1,:,0])
        #print(f"water height slice at t={t_curr}: \n", U[2,:,0])
        #print(f"water height slice at t={t_curr}: \n", U[3,:,0])
        #print(f"water height slice at t={t_curr}: \n", U[4,:,0])
        '''
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        h_history = U[:,:,0]
        print(f"t: {(t_curr):.2f}, dt: {dt:.6f}")
        L1 = U1
        x = np.linspace(0, L1*L, Nx)
        y = np.linspace(0, L1*L, Ny)
        X, Y = np.meshgrid(x, y)
        #ax.plot_wireframe(X, Y, H1*h_history)
        ax.plot_surface(X, Y, H1*h_history, cmap='viridis')
        plt.title(f"water height at t={(t_curr):.2f}")
        #
        plt.figure(figsize=(10, 5))
        h_history = U[int(Nx/2),:,0]
        print("plot h shape: ", h_history.shape)
        plt.plot(np.linspace(0, 100, Ny), H1*h_history,
                    linewidth=3, label=f't = {t_curr:.2f} s')
        plt.grid(True)
        plt.xlabel('Distance [m]')
        plt.ylabel('Water Height [m]')
        plt.legend()
        plt.show()
        '''
        if reaches_target:
            t_curr = float(target_times[target_idx])
            U_history.append(U.copy())
            saved_times.append(t_curr)
            target_idx += 1

        if time_step%50 == 0:
            print('-------------------------------------------------------------')
            print(f"Max wavespeed in x direction: {max_wavespeed_x:8.2e}; \n"
                    f"Current time: {t_curr:8.2e}; dt: {dt:8.2e}; time step: {time_step}\n"
                    f"velocity u: {np.max(np.abs(U[:,:,1]/U[:,:,0]))}")
            #if var_num == 3:
            #    print(f"alpha1 {np.max(np.abs(U[:,2]/U[:,0]))}")
            #print(f"velocity: {U[:,1]}")
        #if time_step == 5:
        #    break
        

    #print("U history length: ", len(U_history))
    #print(" U shape : ", U_history[0].shape)
    print(f"Max wavespeed in x direction: {max_wavespeed_x:8.2e}; \n"
                    f"Current time: {t_curr:8.2e}; dt: {dt:8.2e}; time step: {time_step}\n"
                    f"velocity u: {np.max(np.abs(U[:,:,1]/U[:,:,0]))}")
    #if var_num == 3:
    #    print(f"alpha1 {np.max(np.abs(U[:,2]/U[:,0]))}")
    if target_idx != target_times.size:
        raise RuntimeError("the solver did not reach every requested target time")

    dt_history = np.diff(np.concatenate(([0.0], np.asarray(saved_times))))
    return U_history, dt_history


# Function to save data
def save_data(model_type, Nx, k_coe, var_num, U_history, dt_history, dx, time_end, L, init_case):
    folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    moment_num = int((var_num-3)/2)
    if model_type == 'original':
        file_name = f"HSWME{moment_num}_N{Nx}_k{k_coe:.1e}.npy"
    elif model_type == 'modified':
        file_name = f"M-HSWME{moment_num}_N{Nx}_k{k_coe:.1e}.npy"

    file_path = os.path.join(folder_name, init_case+file_name)
    time_history = np.cumsum(dt_history)
    if len(U_history) != len(time_history):
        raise ValueError("U_history and dt_history must have the same length")
    np.save(file_path, {'U_history': U_history, 'dt_history': dt_history,
                        'time_history': time_history, 'dx': dx,
                        'time_end': time_end, 'L': L, 'k_coe':k_coe})
    print(f"Data saved in {file_path}")
    
