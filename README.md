# Repository Structure

```
├── swe-modified-2025
│   ├── OpenFOAM-tests/                   #
│   │   ├── Fig1_Openfoam_accuracy_code   #
│   │   ├── 3D-initial-velocity-zero      #
│   │   └── 3D-initial-velocity-nonzero   #
│   ├── MHSWME/                           #
│   │   ├── SWE_1D_dambreak               #
│   │   └── SWE_2D_dambreak               #
│   └── README.md                         #
├── Paper 2: to be updated
│   └── README.md                         #
└── README.md                             # This file
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


## How to Use
### Running OpenFoam on Linux or Windows with WSL2
#### 1. Install OpenFOAM
Install OpenFOAM by following the official OpenFOAM guide. This project was tested using OpenFOAM v12. Make sure the OpenFOAM environment is sourced before running the case.

#### 2. Run the OpenFOAM Simulation
  Open a terminal in the case directory. It is recommended to clean all preivous results before starting a new simulation.

  ```bash
  chmod +x Allclean
  ./Allclean

  blockMesh
  setFields
  foamRun
  ```
  This will:
   - remove old simulation data
   - generate the computational mesh
   - initialize the fields, and
   - run the OpenFOAM solver
#### 3. Run on an HPC Cluster
For high-resolution simulations, the case can be run on an HPC cluster. In this case, you may need to follow the cluster's job submission rules and prepare a batch script accordingly.

In the job script, specify:
- the number of nodes,
- the number of CPUs per node,
- the memory requested per CPU, and
- the wall-clock time limit

This setup is especially convenient when running **parallel simulations** for large-scale or high-resolution cases.

## swe-modified-2025/MHSWME/SWE_1D_dambreak
Numerical solver for the 1D dam break problem using the SWE, HSWME, MSWE, and MSHWMEs.

To run the dam break simulation suite with the default configuration:
```bash
python test_run.py
```
This will run all combinations of model type (`original`, `modified`) and expansion order (`var_num` = 2, 3, 4) and save the results as `.npy` files under a `Data/` directory that is created automatically.

To use the solver in your own script, import from `mswme` directly:

```python
from mswme import swme_solver, save_data, load_data

U_history, dt_history = swme_solver(
    u_scale=100,
    h_scale=1.5,
    h_left=1.0,
    h_right=2/3,
    length=1.0,
    gravity=G,
    nx=4000,
    cfl=0.7,
    time_end=3.0,
    alpha0=alpha0,
    r0=R0,
    epsilon=epsilon,
    model_type='modified',   # 'original' or 'modified'
    var_num=3,               # 2: SWE, 3: SWME1, 4: SWME2
    t_target=[1.0, 2.0, 3.0],
    initial_case='init'
)
```

Saved `.npy` files can be loaded back with:

```python
from mswme import load_data
data = load_data('Data/M-SWME_data_k10000000000.0.npy')
# Keys: 'U_history', 'dt_history', 'dx', 'time_end', 'L', 'k_coe'
```

### Output
Results are stored as NumPy `.npy` dictionaries in the `Data/` directory, named by model and friction coefficient, e.g. `M-SWME_data_k1e+10.npy`. Each file contains the full time history of the state vector (`U_history`), the time step history (`dt_history`), grid spacing (`dx`), end time, domain length, and friction coefficient.

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

