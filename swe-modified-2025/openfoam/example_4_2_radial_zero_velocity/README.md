# Example 4.2: Radial Water Collapse with Zero Initial Velocity

Created: 2026-08-26
Repository-relative folder: `swe-modified-2025/openfoam/example_4_2_radial_zero_velocity/`

Purpose: generate the OpenFOAM reference solutions for the radial water-collapse case with zero initial velocity.

The active cases are `N=200`, `N=400`, and `N=800`.
Each case contains the mesh, phase, tracer, physical-property, numerical-scheme, solver-control, and initial-condition dictionaries required for generation.

Enter one resolution folder and run `./Allrun`; use `./Allclean` to remove locally generated case output.

Status on 2026-08-26: prepared and source-checked, but not rerun during packaging.
Available artifacts are generator inputs and run/cleanup scripts only; there are no data or results in this folder.
Remaining work is to execute and validate the desired resolutions in a compatible OpenFOAM 12 environment.
