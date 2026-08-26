# Repository Structure

```text
.
├── swe-modified-2025/
│   ├── environment/
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── moment_models/
│   │   ├── example_4_1_2d_dam_break/
│   │   ├── example_4_2_radial_zero_velocity/
│   │   └── example_4_3_linear_velocity/
│   ├── openfoam/
│   │   ├── example_4_1_2d_dam_break/
│   │   ├── example_4_2_radial_zero_velocity/
│   │   └── example_4_3_linear_velocity/
│   ├── .gitignore
│   ├── README.md
│   └── SOURCE_MANIFEST.md
├── .gitignore
├── LICENSE
└── README.md
```

# Works related
1.  Moment-enhanced shallow water equations for non-slip boundary conditions

    **Paper DOI:** [10.48550/arXiv.2506.14785](https://doi.org/10.48550/arXiv.2506.14785)  
    **Authors:** Shiping Zhou, Juntao Huang, Andrew J. Christlieb  
    **Publication:** under review

    [![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://doi.org/10.48550/arXiv.2506.14785)
    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18684508.svg)](https://doi.org/10.5281/zenodo.18684508)
    [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


    ### Abstract

    The shallow water equations often assume a constant velocity profile along the vertical axis. However, this assumption does not hold in many practical applications. To better approximate the vertical velocity distribution, models such as the shallow water moment expansion models have been proposed. Nevertheless, under non-slip bottom boundary conditions, both the standard shallow water equation and its moment-enhanced models struggle to accurately capture the vertical velocity profile due to the stiff source terms. In this work, we propose modified shallow water equations and corresponding moment-enhanced models that perform well under both non-slip and slip boundary conditions. The primary difference between the modified and original models lies in the treatment of the source term, which allows our modified moment expansion models to be readily generalized, while maintaining compatibility with our previous analysis on the hyperbolicity of the model. To assess the performance of both the standard and modified moment expansion models, we conduct a comprehensive numerical comparison with the incompressible Navier--Stokes equations -- a comparison that is absent from existing literature.

2. To be updated.


## Package Scope

The `swe-modified-2025/` package contains active data-generation source code only.
It includes OpenFOAM case definitions and Python moment-model generators for Examples 4.1--4.3.
Existing numerical data, figures, result summaries, plotting, export, postprocessing, Slurm files, and private notes are not included.

See [`swe-modified-2025/README.md`](swe-modified-2025/README.md) for the package overview and [`swe-modified-2025/SOURCE_MANIFEST.md`](swe-modified-2025/SOURCE_MANIFEST.md) for the source mapping.

## How to Use

### OpenFOAM generators

The case dictionaries use OpenFOAM 12.
Load a compatible OpenFOAM environment, enter the desired resolution folder under `swe-modified-2025/openfoam/`, and run:

```bash
./Allclean
./Allrun
```

Each case script builds the mesh, initializes the fields, and runs the configured solver.

### Moment-model generators

Install the Python dependency with:

```bash
python3 -m pip install -r swe-modified-2025/environment/requirements.txt
```

The main generator entry points are:

```text
swe-modified-2025/moment_models/example_4_1_2d_dam_break/low_order/generate_all.sh
swe-modified-2025/moment_models/example_4_1_2d_dam_break/order_study/generate_all.sh
swe-modified-2025/moment_models/example_4_2_radial_zero_velocity/generate_data.py
swe-modified-2025/moment_models/example_4_3_linear_velocity/generate_data.py
```

Run any Python generator with `--help` to see its mesh, model, output-time, and output-directory options.
Generated data remain local and are excluded by the package `.gitignore`.

## Citation

```bibtex
@article{Zhou2025,
  title = {Moment-enhanced shallow water equations for non-slip boundary conditions},
  author = {Zhou, S., Huang, J., and Christlieb, A.J.},
  year = {2025},
  journal = {arXiv},
  doi = {10.48550/arXiv.2506.14785}
}
```
