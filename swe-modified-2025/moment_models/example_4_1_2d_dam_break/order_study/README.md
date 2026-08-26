# Example 4.1 Retained-Order Generators

Created: 2026-08-26
Repository-relative folder: `swe-modified-2025/moment_models/example_4_1_2d_dam_break/order_study/`

Purpose: generate no-slip HSWME and MHSWME solutions across the retained-order set used by the active study.

Run `./generate_all.sh` to execute both generator families with common options, such as `--nx`, `--orders`, `--times`, and `--overwrite`.
The default production grid is `nx=4000`, and each family writes its own ignored `data/` folder.

Status on 2026-08-26: generator code is prepared and syntax-checked; no retained-order production run was performed during packaging.
Available artifacts are the numerical kernels, drivers, and metadata writers only.
Remaining work is to run the requested orders and verify the generated solutions.
