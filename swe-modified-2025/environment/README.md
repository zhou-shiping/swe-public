# Environment

Created: 2026-08-26
Repository-relative folder: `swe-modified-2025/environment/`

The moment-model generators require Python 3 and NumPy.
No plotting or table-processing dependency is needed because the public package contains generation code only.

The OpenFOAM dictionaries identify OpenFOAM 12 and require an initialized OpenFOAM environment that provides `blockMesh`, `setFields`, the configured multiphase solver, and the standard run functions.
Exact compiler, MPI, and platform choices remain environment-specific.
