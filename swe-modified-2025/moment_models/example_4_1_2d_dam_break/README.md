# Example 4.1 Moment Models

Created: 2026-08-26
Repository-relative folder: `swe-modified-2025/moment_models/example_4_1_2d_dam_break/`

Purpose: generate the one-dimensional depth-averaged SWE, HSWME, MSWE, and MHSWME data for the dam-break study.

The `low_order/` driver generates the perfect-slip and no-slip data for retained orders through $N=3$.
The `order_study/` drivers generate the no-slip classical and modified retained-order sweeps using the active arbitrary-order kernels.

Run `low_order/generate_all.sh` for the low-order matrix and `order_study/generate_all.sh` for both retained-order sweeps.
Both launchers accept the command-line options supported by their Python drivers.

Status on 2026-08-26: prepared and source-checked, but production data were not regenerated during packaging.
Available artifacts are solver kernels and data-generation drivers; no data, analysis, or results are included.
Remaining work is to run the requested matrix and validate its numerical outputs separately.
