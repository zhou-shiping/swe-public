# OpenFOAM Data Generators

Created: 2026-08-26
Repository-relative folder: `swe-modified-2025/openfoam/`

This tree contains only active OpenFOAM 12 simulation inputs and the scripts needed to build, initialize, run, and clean each case.
Each resolution is a self-contained case with `0/`, `constant/`, `system/`, `Allrun`, and `Allclean`.

Run a case from an initialized OpenFOAM environment with `./Allrun`.
No Slurm scripts, reconstructed export step, analysis code, plotting code, or simulation output is included.

Status on 2026-08-26: source cases are prepared and structurally checked; production simulations were not rerun as part of packaging.
