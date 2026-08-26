# Example 4.1: Two-Dimensional Dam Break

Created: 2026-08-26
Repository-relative folder: `swe-modified-2025/openfoam/example_4_1_2d_dam_break/`

Purpose: generate the OpenFOAM reference solutions for the no-slip quadratic-velocity and perfect-slip cubic-velocity dam-break cases.

The active resolution cases are `Ny=160`, `Ny=320`, and `Ny=640` for both bottom conditions.
Each case uses the copied OpenFOAM initial-field templates and dictionaries, runs mesh generation and field initialization, and executes the configured solver.

Enter one resolution folder and run `./Allrun`; use `./Allclean` to remove locally generated case output.

Status on 2026-08-26: prepared and source-checked, but not rerun during packaging.
Available artifacts are generator inputs and run/cleanup scripts only; there are no data or results in this folder.
Remaining work is to execute the selected resolutions in a compatible OpenFOAM 12 environment and validate the resulting data separately.
