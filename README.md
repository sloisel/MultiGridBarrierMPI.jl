# MultiGridBarrierMPI.jl

A Julia package that bridges MultiGridBarrier.jl and LinearAlgebraMPI.jl for distributed multigrid barrier computations using native MPI types.

## Overview

MultiGridBarrierMPI.jl extends the MultiGridBarrier.jl package to work with LinearAlgebraMPI.jl's distributed matrix and vector types. This enables efficient parallel computation of multigrid barrier methods across multiple MPI ranks without requiring PETSc.

## Key Features

- **1D, 2D, and 3D Support**: Full support for 1D, 2D triangular, and 3D hexahedral finite elements
- **Seamless Integration**: Drop-in replacement for MultiGridBarrier's native types
- **Pure Julia MPI**: Uses LinearAlgebraMPI.jl for distributed linear algebra (no external libraries required)
- **Type Conversion**: Easy conversion between native Julia arrays and MPI distributed types
- **MPI-Aware**: All operations correctly handle MPI collective requirements
- **MUMPS Solver**: Uses MUMPS direct solver for accurate Newton iterations

## Quick Example

Solve a 2D p-Laplace problem with distributed MPI types:

```julia
using MPI
MPI.Init()

using MultiGridBarrierMPI
using LinearAlgebraMPI

# Solve with MPI distributed types (L=3 refinement levels)
sol_mpi = fem2d_mpi_solve(Float64; L=3, p=1.0, verbose=false)

# Convert to native types for visualization
sol_native = mpi_to_native(sol_mpi)

# Only rank 0 creates the plot
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    using MultiGridBarrier
    using PyPlot
    plot(sol_native)
    savefig("solution.png")
    println("Plot saved to solution.png")
end
```

Run with MPI:

```bash
mpiexec -n 4 julia --project example.jl
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/sloisel/MultiGridBarrierMPI.jl")
```

Or for development:

```bash
git clone https://github.com/sloisel/MultiGridBarrierMPI.jl
cd MultiGridBarrierMPI.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Running Tests

```bash
cd MultiGridBarrierMPI.jl
mpiexec -n 2 julia --project test/runtests.jl
```

## Documentation

See the `docs/` folder for full documentation, or build locally:

```bash
cd docs
julia --project make.jl
```

## Package Ecosystem

This package is part of a larger ecosystem:

- **[MultiGridBarrier.jl](https://github.com/sloisel/MultiGridBarrier.jl)**: Core multigrid barrier method implementation
- **[LinearAlgebraMPI.jl](https://github.com/sloisel/LinearAlgebraMPI.jl)**: Pure Julia distributed linear algebra with MPI
- **MPI.jl**: Julia MPI bindings for distributed computing

## Requirements

- Julia 1.10 or later
- MPI installation (OpenMPI, MPICH, or Intel MPI)
- MUMPS for sparse direct solves

## License

MIT License
