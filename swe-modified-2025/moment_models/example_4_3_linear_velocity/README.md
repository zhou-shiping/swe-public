# Example 4.3 Moment Models

Created: 2026-08-26
Repository-relative folder: `swe-modified-2025/moment_models/example_4_3_linear_velocity/`

Purpose: generate two-dimensional original and modified moment-model solutions for water collapse with the linear initial velocity $u=v=S z$.

The default driver uses `nx=ny=400`, CFL `0.7`, output times `0, 1, 2, 3`, state sizes `7, 5, 3`, and $S=0.25\,\mathrm{s}^{-1}$.
Run `python3 generate_data.py --help` to select the mesh, saved times, model types, velocity slope, or output folder.

Status on 2026-08-26: prepared and syntax-checked, but the production cases were not rerun during packaging.
Available artifacts are the solver and generator only; no plots, data, or results are included.
Remaining work is to execute the desired matrix and validate the generated data independently.
