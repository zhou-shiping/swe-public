# Source Manifest

Manifest date: 2026-08-26
Package path: `swe-modified-2025/`

Only active data-generation sources were admitted to this package.

## OpenFOAM mappings

| Package folder | Project source |
|---|---|
| `openfoam/example_4_1_2d_dam_break/no_slip/` | `src/openfoam/2d_dam_break_quadratic_velocity/` |
| `openfoam/example_4_1_2d_dam_break/perfect_slip/` | `src/openfoam/2d_dam_break_quadratic_velocity_perfect_slip/` |
| `openfoam/example_4_2_radial_zero_velocity/` | `src/openfoam/3d_radial_water_collapse_zero_velocity/` |
| `openfoam/example_4_3_linear_velocity/` | `src/openfoam/3d_water_collapse_linear_velocity/` |

For every retained OpenFOAM case, the allowlist contains `Allrun`, `Allclean`, the five source templates in `0/`, and the top-level dictionary files in `constant/` and `system/`.
Generated fields, meshes, time directories, processor directories, logs, exported tables, and postprocessing outputs were not copied.
The six Example 4.1 `Allrun` copies omit `reconstructPar` because that step was documented solely as preparation for export and plotting.

## Moment-model mappings

| Package file or folder | Project source |
|---|---|
| `moment_models/example_4_1_2d_dam_break/low_order/MSWME.py` | `src/moment_1d/SWE_1D/MSWME.py` |
| `moment_models/example_4_1_2d_dam_break/order_study/hswme_no_slip_order_study/` | `examples/example_4_1_2d_dam_break/hswme_no_slip_order_study/` |
| `moment_models/example_4_1_2d_dam_break/order_study/mhswme_no_slip_order_study/` | `examples/example_4_1_2d_dam_break/mhswme_no_slip_order_study/` |
| `moment_models/example_4_2_radial_zero_velocity/MSWME.py` | `examples/example_4_2_3d_radial_water_collapse/water_collapse_SWE_2D/MSWME.py` |
| `moment_models/example_4_3_linear_velocity/MSWME.py` | `examples/example_4_3_3d_water_collapse/water_collapse_SWE_2D/MSWME.py` |

The copied numerical kernels retain their active solver and data-writing code.
Plot functions, plotting imports, inert plotting blocks, comparison metadata, and analysis drivers were removed from the package copies.
The `generate_data.py`, `generate_all.sh`, and retained-order `run_orders.py` files are generator-only entry points.

## Exclusion rules

The package contains no `notes/` directory, `*.sb` file, existing data file, plot, exported table, postprocessing utility, or archived source tree.
