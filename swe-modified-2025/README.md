# Moment-enhanced shallow water equations for non-slip boundary conditions

**Paper DOI:** [10.48550/arXiv.2506.14785](https://doi.org/10.48550/arXiv.2506.14785)  
**Authors:** Shiping Zhou, Juntao Huang, Andrew J. Christlieb  
**Publication:** under review

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://doi.org/10.48550/arXiv.2506.14785)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


## Abstract

The shallow water equations often assume a constant velocity profile along the vertical axis. However, this assumption does not hold in many practical applications. To better approximate the vertical velocity distribution, models such as the shallow water moment expansion models have been proposed. Nevertheless, under non-slip bottom boundary conditions, both the standard shallow water equation and its moment-enhanced models struggle to accurately capture the vertical velocity profile due to the stiff source terms. In this work, we propose modified shallow water equations and corresponding moment-enhanced models that perform well under both non-slip and slip boundary conditions. The primary difference between the modified and original models lies in the treatment of the source term, which allows our modified moment expansion models to be readily generalized, while maintaining compatibility with our previous analysis on the hyperbolicity of the model. To assess the performance of both the standard and modified moment expansion models, we conduct a comprehensive numerical comparison with the incompressible Navier--Stokes equations -- a comparison that is absent from existing literature.


## Repository Structure

```
├── OpenFOAM-tests/                   #
|   ├── Fig1_Openfoam_accuracy_code   #
|   ├── 3D-initial-velocity-zero      #
|   ├── 3D-initial-velocity-nonzero   #
├── MHSWME/                           #
|   ├── SWE_1D_dambreak               #
|   ├── SWE_2D_dambreak               #
└── README.md                         # This file
```
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

