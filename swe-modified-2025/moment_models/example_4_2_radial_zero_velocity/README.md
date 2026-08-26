# Example 4.2 Moment Models

Created: 2026-08-26
Repository-relative folder: `swe-modified-2025/moment_models/example_4_2_radial_zero_velocity/`

Purpose: generate two-dimensional original and modified moment-model solutions for radial water collapse with zero initial velocity.

The default driver uses `nx=ny=400`, CFL `0.7`, output times `0, 1, 2, 3`, and state sizes `3, 5, 7`.
Run `python3 generate_data.py --help` to select the mesh, saved times, model types, or output folder.

Status on 2026-08-26: prepared and syntax-checked, but the production cases were not rerun during packaging.
Available artifacts are the solver and generator only; no plots, data, or results are included.
Remaining work is to execute the desired matrix and validate the generated data independently.
