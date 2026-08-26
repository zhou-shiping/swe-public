# SWE Modified Models: Generator-Only Source Package

Created: 2026-08-26
Repository-relative folder: `swe-modified-2025/`

This folder is the public, generator-only source package for the modified shallow-water work.
It contains the active OpenFOAM case definitions and moment-model solvers needed to generate numerical data.

## Included

- OpenFOAM 12 case dictionaries and solver launch/cleanup scripts for Examples 4.1--4.3.
- Low-order SWE, HSWME, MSWE, and MHSWME data generators for Example 4.1.
- Arbitrary retained-order HSWME and MHSWME data generators for the Example 4.1 no-slip study.
- Two-dimensional moment-model data generators for Examples 4.2 and 4.3.

## Deliberately excluded

- Existing numerical data, figures, tables, logs, and result summaries.
- Plotting, comparison, export, reconstruction-for-export, analysis, and postprocessing code.
- Slurm `*.sb` files and other MSU HPCC launch files.
- The project-private `notes/` folder.
- Archived, superseded, exploratory, or unused source variants.

## Structure

```text
swe-modified-2025/
├── README.md
├── SOURCE_MANIFEST.md
├── environment/
├── openfoam/
│   ├── example_4_1_2d_dam_break/
│   ├── example_4_2_radial_zero_velocity/
│   └── example_4_3_linear_velocity/
└── moment_models/
    ├── example_4_1_2d_dam_break/
    ├── example_4_2_radial_zero_velocity/
    └── example_4_3_linear_velocity/
```

## Running the generators

Load a compatible OpenFOAM 12 environment before running an OpenFOAM case's `./Allrun` script.
For the moment models, install the dependency listed in `environment/requirements.txt`, enter the relevant example folder, and run its `generate_data.py` or `generate_all.sh` entry point.

Generated data remain local and are ignored by this package's `.gitignore`.

## Status

Status on 2026-08-26: prepared and structurally verified as a generator-only source package.
The source package has not been used to rerun the full production simulations during packaging, so no new scientific-result claim is made here.
The remaining work is to review the package in the destination Git repository and run the desired production cases in the appropriate OpenFOAM or Python environment.
