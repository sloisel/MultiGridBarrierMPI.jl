# API Reference

This page provides detailed documentation for all exported functions in HPCMultiGridBarrier.jl.

!!! note "All Functions Are Collective"
    All functions documented here are **MPI collective operations**. Every MPI rank must call these functions together with the same parameters. Failure to do so will result in deadlock.

## High-Level API

These functions provide the simplest interface for solving problems with MPI types.

### 1D Problems

```@docs
fem1d_hpc
fem1d_hpc_solve
```

### 2D Problems

```@docs
fem2d_hpc
fem2d_hpc_solve
```

### 3D Problems

```@docs
fem3d_hpc
fem3d_hpc_solve
```

## Type Conversion API

These functions convert between native Julia types and MPI distributed types.
The `hpc_to_native` function dispatches on type, handling `Geometry`, `AMGBSOL`, and `ParabolicSOL` objects.

```@docs
native_to_hpc
hpc_to_native
```

## Type Mappings Reference

### Native to MPI Conversions

When converting from native Julia types to MPI distributed types:

| Native Type | MPI Type | Usage |
|-------------|----------|-------|
| `Matrix{T}` | `HPCMatrix{T}` | Geometry coordinates, dense data |
| `Vector{T}` | `HPCVector{T}` | Weights, dense vectors |
| `SparseMatrixCSC{T,Int}` | `HPCSparseMatrix{T}` | Sparse operators, subspace matrices |

### MPI to Native Conversions

When converting from MPI distributed types back to native Julia types:

| MPI Type | Native Type |
|----------|-------------|
| `HPCMatrix{T}` | `Matrix{T}` |
| `HPCVector{T}` | `Vector{T}` |
| `HPCSparseMatrix{T}` | `SparseMatrixCSC{T,Int}` |

## Geometry Structure

The `Geometry` type from MultiGridBarrier is parameterized by its storage types:

**Native Geometry:**
```julia
Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}
```

**MPI Geometry:**
```julia
Geometry{T, HPCMatrix{T}, HPCVector{T}, HPCSparseMatrix{T}, Discretization}
```

### Fields

- **`discretization`**: Discretization information (domain, mesh, etc.)
- **`x`**: Geometry coordinates (Matrix or HPCMatrix)
- **`w`**: Quadrature weights (Vector or HPCVector)
- **`operators`**: Dictionary of operators (id, dx, dy, etc.)
- **`subspaces`**: Dictionary of subspace projection matrices
- **`refine`**: Vector of refinement matrices (coarse -> fine)
- **`coarsen`**: Vector of coarsening matrices (fine -> coarse)

## Solution Structure

The `AMGBSOL` type from MultiGridBarrier contains the complete solution:

### Fields

- **`z`**: Solution matrix/vector
- **`SOL_feasibility`**: NamedTuple with feasibility phase information
- **`SOL_main`**: NamedTuple with main solve information
  - `t_elapsed`: Elapsed solve time in seconds
  - `ts`: Barrier parameter values
  - `its`: Iterations per level
  - `c_dot_Dz`: Convergence measure values
- **`log`**: Vector of iteration logs
- **`geometry`**: The geometry used for solving

## MPI and IO Utilities

### HPCSparseArrays.io0()

Returns an IO stream that only writes on rank 0:

```julia
using HPCSparseArrays

println(io0(), "This prints once from rank 0")
```

### MPI Rank Information

```julia
using MPI

rank = MPI.Comm_rank(MPI.COMM_WORLD)  # Current rank (0 to nranks-1)
nranks = MPI.Comm_size(MPI.COMM_WORLD)  # Total number of ranks
```

## Examples

### Type Conversion Round-Trip

```julia
using MPI
MPI.Init()

using HPCMultiGridBarrier
using HPCSparseArrays
using MultiGridBarrier
using LinearAlgebra

# Create native geometry
g_native = fem2d(; L=2)

# Convert to MPI
g_hpc = native_to_hpc(g_native)

# Solve with MPI types
sol_hpc = amgb(g_hpc; p=2.0)

# Convert back to native
sol_native = hpc_to_native(sol_hpc)
g_back = hpc_to_native(g_hpc)

# Verify round-trip accuracy
@assert norm(g_native.x - g_back.x) < 1e-10
@assert norm(g_native.w - g_back.w) < 1e-10
```

### Accessing Operator Matrices

```julia
# Native geometry
g_native = fem2d(; L=2)
id_native = g_native.operators[:id]  # SparseMatrixCSC

# MPI geometry
g_hpc = native_to_hpc(g_native)
id_mpi = g_hpc.operators[:id]  # HPCSparseMatrix

# Convert back if needed
id_back = SparseMatrixCSC(id_mpi)  # SparseMatrixCSC
```

## Integration with MultiGridBarrier

All MultiGridBarrier functions work seamlessly with MPI types:

```julia
using MultiGridBarrier: amgb

# Create MPI geometry
g = fem2d_hpc(Float64; L=3)

# Use MultiGridBarrier functions directly
sol = amgb(g; p=1.0, verbose=true)
```

The package extends MultiGridBarrier's internal API (`amgb_zeros`, `amgb_diag`, `amgb_blockdiag`, `map_rows`, etc.) to work with MPI types automatically.
