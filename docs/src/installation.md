# Installation

## Prerequisites

### MPI

MultiGridBarrierMPI.jl requires an MPI implementation. When you install the package, Julia automatically provides `MPI.jl` with `MPI_jll` (bundled MPI implementation).

For HPC environments, you may want to configure MPI.jl to use your system's MPI installation. See the [MPI.jl documentation](https://juliaparallel.org/MPI.jl/stable/configuration/) for details.

### MUMPS

The package uses MUMPS for sparse direct solves through HPCSparseArrays.jl. MUMPS is typically available through your system's package manager or HPC module system.

## Package Installation

### Basic Installation

```julia
using Pkg
Pkg.add(url="https://github.com/sloisel/MultiGridBarrierMPI.jl")
```

### Development Installation

To install the development version:

```bash
git clone https://github.com/sloisel/MultiGridBarrierMPI.jl
cd MultiGridBarrierMPI.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Verification

Test your installation with MPI:

```bash
cd MultiGridBarrierMPI.jl
mpiexec -n 2 julia --project test/runtests.jl
```

All tests should pass. Expected output:
```
Test Summary:          | Pass  Total
MultiGridBarrierMPI.jl |    2      2
```

## Initialization Pattern

!!! tip "Initialization Pattern"
    Initialize MPI first, then load the package:

```julia
# CORRECT
using MPI
MPI.Init()

using MultiGridBarrierMPI
# Now you can use the package

# WRONG - MPI must be initialized before using MPI types
using MultiGridBarrierMPI
# Missing MPI.Init() - will fail when calling functions
```

## Running MPI Programs

### Multi-Rank Execution

For distributed execution, create a script file (e.g., `my_program.jl`):

```julia
using MPI
MPI.Init()

using MultiGridBarrierMPI
using HPCSparseArrays

# Your parallel code here
sol = fem2d_mpi_solve(Float64; L=3, p=1.0)
println(io0(), "Solution computed!")
```

Run with MPI:

```bash
mpiexec -n 4 julia --project my_program.jl
```

!!! tip "Output from Rank 0 Only"
    Use `io0()` from HPCSparseArrays for output to avoid duplicate messages:
    ```julia
    println(io0(), "This prints once from rank 0")
    ```

## Troubleshooting

### MPI Issues

If you see MPI-related errors, try rebuilding MPI.jl:

```julia
using Pkg; Pkg.build("MPI")
```

### MUMPS Issues

If MUMPS fails to load, ensure it's properly installed on your system and that HPCSparseArrays.jl can find it.

### Test Failures

If tests fail:

1. Ensure you're using at least Julia 1.10 (LTS version)
2. Check all dependencies are installed: `Pkg.status()`
3. Run with verbose output to see detailed errors

## Next Steps

Once installed, proceed to the [User Guide](@ref) to learn how to use the package.
