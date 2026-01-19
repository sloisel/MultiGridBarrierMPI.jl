```@meta
CurrentModule = MultiGridBarrierMPI
```

```@eval
using Markdown
using Pkg
using MultiGridBarrierMPI
v = string(pkgversion(MultiGridBarrierMPI))
md"# MultiGridBarrierMPI.jl $v"
```

**A Julia package that bridges MultiGridBarrier.jl and HPCLinearAlgebra.jl for distributed multigrid barrier computations.**

## Overview

MultiGridBarrierMPI.jl extends the MultiGridBarrier.jl package to work with HPCLinearAlgebra.jl's distributed matrix and vector types. This enables efficient parallel computation of multigrid barrier methods across multiple MPI ranks using pure Julia distributed types (no PETSc required).

## Key Features

- **1D, 2D, and 3D Support**: Full support for 1D elements, 2D triangular, and 3D hexahedral finite elements
- **Seamless Integration**: Drop-in replacement for MultiGridBarrier's native types
- **Pure Julia MPI**: Uses HPCLinearAlgebra.jl for distributed linear algebra
- **Type Conversion**: Easy conversion between native Julia arrays and MPI distributed types
- **MPI-Aware**: All operations correctly handle MPI collective requirements
- **MUMPS Solver**: Uses MUMPS direct solver for accurate Newton iterations

## Quick Example

Solve a 2D p-Laplace problem with distributed MPI types. Save this code to `example.jl`:

```julia
using MPI
MPI.Init()

using MultiGridBarrierMPI
using HPCLinearAlgebra
using MultiGridBarrier

# Solve with MPI distributed types (L=3 refinement levels)
sol_mpi = fem2d_mpi_solve(Float64; L=3, p=1.0, verbose=false)

# Convert to native types for visualization
sol_native = mpi_to_native(sol_mpi)

# Only rank 0 creates the plot
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    using PyPlot
    plot(sol_native)
    savefig("solution.png")
    println("Plot saved to solution.png")
end
```

**Run with MPI:**

```bash
mpiexec -n 4 julia --project example.jl
```

## Documentation Contents

```@contents
Pages = ["installation.md", "guide.md", "api.md"]
Depth = 2
```

## Package Ecosystem

This package is part of a larger ecosystem:

- **[MultiGridBarrier.jl](https://github.com/sloisel/MultiGridBarrier.jl)**: Core multigrid barrier method implementation (1D, 2D, and 3D)
- **[HPCLinearAlgebra.jl](https://github.com/sloisel/HPCLinearAlgebra.jl)**: Pure Julia distributed linear algebra with MPI
- **MPI.jl**: Julia MPI bindings for distributed computing

## Requirements

- Julia 1.10 or later (LTS version)
- MPI installation (OpenMPI, MPICH, or Intel MPI)
- MUMPS for sparse direct solves
- At least 2 MPI ranks recommended for testing

## Citation

If you use this package in your research, please cite:

```bibtex
@software{multigridbarriermpi,
  author = {Loisel, Sebastien},
  title = {MultiGridBarrierMPI.jl: Distributed Multigrid Barrier Methods with MPI},
  year = {2024},
  url = {https://github.com/sloisel/MultiGridBarrierMPI.jl}
}
```

## License

This package is licensed under the MIT License.
