# User Guide

This guide covers the essential workflows for using HPCMultiGridBarrier.jl.

## Initialization

Every program using HPCMultiGridBarrier.jl must initialize MPI before using the package:

```julia
using MPI
MPI.Init()

using HPCMultiGridBarrier
```

## Basic Workflow

The typical workflow consists of three steps:

1. **Solve with MPI types** (distributed computation)
2. **Convert to native types** (for analysis/plotting)
3. **Visualize or analyze** (using MultiGridBarrier's tools)

### Complete Example with Visualization

Here's a complete example that solves a 2D FEM problem, converts the solution, and plots it:

```julia
using MPI
MPI.Init()

using HPCMultiGridBarrier
using HPCSparseArrays
using MultiGridBarrier

# Step 1: Solve with MPI distributed types
sol_hpc = fem2d_hpc_solve(Float64; L=3, p=1.0, verbose=false)

# Step 2: Convert solution to native Julia types
sol_native = hpc_to_native(sol_hpc)

# Step 3: Plot the solution using MultiGridBarrier's plot function
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    using PyPlot
    figure(figsize=(10, 8))
    plot(sol_native)
    title("Multigrid Barrier Solution (L=3)")
    tight_layout()
    savefig("solution_plot.png")
end
println(io0(), "Solution plotted!")
```

!!! tip "Running This Example"
    Save this code to a file (e.g., `visualize.jl`) and run with:
    ```bash
    mpiexec -n 4 julia --project visualize.jl
    ```

## Understanding MPI Collective Operations

!!! warning "All Functions Are Collective"
    All exported functions in HPCMultiGridBarrier.jl are **MPI collective operations**. This means:
    - All MPI ranks must call the function
    - All ranks must call it with the same parameters
    - Deadlock will occur if only some ranks call a collective function

**Correct usage:**
```julia
# All ranks execute this together
sol = fem2d_hpc_solve(Float64; L=2, p=1.0)
```

**Incorrect usage (causes deadlock):**
```julia
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    sol = fem2d_hpc_solve(Float64; L=2, p=1.0)  # Only rank 0 calls - DEADLOCK!
end
```

## Type Conversions

### Native to MPI

Convert native Julia arrays to MPI distributed types:

```julia
using MultiGridBarrier

# Create native geometry
g_native = fem2d(; L=2)

# Convert to MPI types for distributed computation
g_hpc = native_to_hpc(g_native)
```

**Type mappings:**

| Native Type | MPI Type | Description |
|-------------|----------|-------------|
| `Matrix{T}` | `HPCMatrix{T}` | Dense distributed matrix |
| `Vector{T}` | `HPCVector{T}` | Dense distributed vector |
| `SparseMatrixCSC{T,Int}` | `HPCSparseMatrix{T}` | Sparse distributed matrix |

### MPI to Native

Convert MPI types back to native Julia arrays:

```julia
# Create and solve with MPI types
g_hpc = fem2d_hpc(Float64; L=2)
sol_hpc = amgb(g_hpc; p=2.0)

# Convert back for analysis
g_native = hpc_to_native(g_hpc)
sol_native = hpc_to_native(sol_hpc)

# Now you can use native Julia operations
using LinearAlgebra
z_matrix = sol_native.z
solution_norm = norm(z_matrix)
println(io0(), "Solution norm: ", solution_norm)
```

## Advanced Usage

### Custom Geometry Workflow

For more control, construct geometries manually:

```julia
using MPI
MPI.Init()

using HPCMultiGridBarrier
using HPCSparseArrays
using MultiGridBarrier

# 1. Create native geometry with specific parameters
g_native = fem2d(; L=2)

# 2. Convert to MPI for distributed solving
g_hpc = native_to_hpc(g_native)

# 3. Solve with custom barrier parameters
sol_hpc = amgb(g_hpc;
    p=1.5,           # Barrier power parameter
    verbose=true,    # Print convergence info
    maxit=100,       # Maximum iterations
    tol=1e-8)        # Convergence tolerance

# 4. Convert solution back
sol_native = hpc_to_native(sol_hpc)

# 5. Access solution components
println(io0(), "Newton steps: ", sum(sol_native.SOL_main.its))
println(io0(), "Elapsed time: ", sol_native.SOL_main.t_elapsed, " seconds")
```

### Comparing MPI vs Native Solutions

Verify that MPI and native implementations give the same results:

```julia
using MPI
MPI.Init()

using HPCMultiGridBarrier
using HPCSparseArrays
using MultiGridBarrier
using LinearAlgebra

# Solve with MPI (distributed)
sol_hpc_dist = fem2d_hpc_solve(Float64; L=2, p=1.0, verbose=false)
z_hpc = hpc_to_native(sol_hpc_dist).z

# Solve with native (sequential, on rank 0)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    sol_native = MultiGridBarrier.fem2d_solve(Float64; L=2, p=1.0, verbose=false)
    z_native = sol_native.z

    # Compare solutions
    diff = norm(z_hpc - z_native) / norm(z_native)
    println("Relative difference: ", diff)
    @assert diff < 1e-10 "Solutions should match!"
end
```

## IO and Output

### Printing from One Rank

Use `io0()` from HPCSparseArrays to print from rank 0 only:

```julia
using HPCSparseArrays

# This prints once (from rank 0)
println(io0(), "Hello from rank 0!")

# Without io0(), this prints from ALL ranks
println("Hello from rank ", MPI.Comm_rank(MPI.COMM_WORLD))
```

### MPI Rank Information

```julia
using MPI

rank = MPI.Comm_rank(MPI.COMM_WORLD)  # Current rank (0 to nranks-1)
nranks = MPI.Comm_size(MPI.COMM_WORLD)  # Total number of ranks
```

## Performance Considerations

### Threading

For optimal performance, use BLAS threading with a single Julia thread:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=10  # or your number of CPU cores
mpiexec -n 1 julia --project my_program.jl
```

- **BLAS threads** (`OPENBLAS_NUM_THREADS`) - Controls threading for dense matrix operations in both Julia and MUMPS
- **OpenMP threads** (`OMP_NUM_THREADS`) - Should be set to 1; MUMPS uses BLAS threading internally

You can also set these in Julia's startup.jl:

```julia
# In ~/.julia/config/startup.jl
ENV["OMP_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = string(Sys.CPU_THREADS)
```

### Performance Comparison (Single-Rank)

The following table compares HPCMultiGridBarrier (using MUMPS) against MultiGridBarrier.jl's native solver on a 2D p-Laplace problem. Both use `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=10`. Benchmarks were run on a 2025 M4 MacBook Pro with 10 CPU cores:

| L | n (grid points) | Native (s) | MPI (s) | Ratio | Diff |
|---|-----------------|------------|---------|-------|------|
| 1 | 14 | 0.018 | 0.029 | 1.60x | 0.00e+00 |
| 2 | 56 | 0.028 | 0.039 | 1.41x | 8.88e-16 |
| 3 | 224 | 0.074 | 0.078 | 1.06x | 4.13e-11 |
| 4 | 896 | 0.480 | 0.410 | 0.85x | 1.04e-14 |
| 5 | 3,584 | 2.508 | 1.771 | 0.71x | 3.55e-12 |
| 6 | 14,336 | 26.384 | 68.846 | 2.61x | 1.08e-13 |
| 7 | 57,344 | 96.694 | 118.070 | 1.22x | 3.31e-13 |
| 8 | 229,376 | 659.276 | 504.672 | 0.77x | 1.31e-11 |

*Ratio = MPI time / Native time (lower is better, <1.0 means MPI is faster)*
*Diff = sup-norm difference between native and MPI solutions*

For medium-sized problems (L=4-5) and large problems (L=8), the MPI version is faster than the native solver.

## 1D Problems

HPCMultiGridBarrier supports 1D finite element problems.

### Basic 1D Example

```julia
using MPI
MPI.Init()

using HPCMultiGridBarrier
using HPCSparseArrays

# Solve a 1D problem with 4 multigrid levels (2^4 = 16 elements)
sol = fem1d_hpc_solve(Float64; L=4, p=1.0, verbose=true)

# Convert solution to native types for analysis
sol_native = hpc_to_native(sol)

println(io0(), "Solution computed successfully!")
println(io0(), "Newton steps: ", sum(sol_native.SOL_main.its))
```

### 1D Parameters

The `fem1d_hpc` and `fem1d_hpc_solve` functions accept:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `L` | Number of multigrid levels (creates 2^L elements) | 4 |

## 2D Problems

### Basic 2D Example

```julia
using MPI
MPI.Init()

using HPCMultiGridBarrier
using HPCSparseArrays

# Solve a 2D problem
sol = fem2d_hpc_solve(Float64; L=2, p=1.0, verbose=true)

# Convert solution to native types for analysis
sol_native = hpc_to_native(sol)

println(io0(), "Solution computed successfully!")
println(io0(), "Newton steps: ", sum(sol_native.SOL_main.its))
```

### 2D Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `L` | Number of multigrid levels | 2 |
| `K` | Coarse mesh vertices (3n×2 matrix) | Unit square triangulation |

## 3D Problems

HPCMultiGridBarrier supports 3D hexahedral finite elements.

### Basic 3D Example

```julia
using MPI
MPI.Init()

using HPCMultiGridBarrier
using HPCSparseArrays

# Solve a 3D problem with Q3 elements and 2 multigrid levels
sol = fem3d_hpc_solve(Float64; L=2, k=3, p=1.0, verbose=true)

# Convert solution to native types for analysis
sol_native = hpc_to_native(sol)

println(io0(), "Solution computed successfully!")
println(io0(), "Newton steps: ", sum(sol_native.SOL_main.its))
```

### 3D Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `L` | Number of multigrid levels | 2 |
| `k` | Polynomial order of elements (Q_k) | 3 |

## Time-Dependent (Parabolic) Problems

HPCMultiGridBarrier supports time-dependent parabolic PDEs through MultiGridBarrier.jl's `parabolic_solve` function.

### Basic Parabolic Example

```julia
using MPI
MPI.Init()

using HPCMultiGridBarrier
using HPCSparseArrays
using MultiGridBarrier

# Create MPI geometry
g = fem2d_hpc(Float64; L=2)

# Solve time-dependent problem from t=0 to t=1 with timestep h=0.2
sol = parabolic_solve(g; h=0.2, p=1.0, verbose=true)

println(io0(), "Parabolic solve completed!")
println(io0(), "Number of timesteps: ", length(sol.ts))
```

### Converting Parabolic Solutions to Native Types

```julia
g = fem2d_hpc(Float64; L=2)
sol_hpc = parabolic_solve(g; h=0.25, p=1.0, verbose=false)

# Convert to native types
sol_native = hpc_to_native(sol_hpc)

# Now sol_native.u contains Vector{Matrix{Float64}}
println(io0(), "Native u type: ", typeof(sol_native.u))
println(io0(), "Snapshot size: ", size(sol_native.u[1]))
```

## Common Patterns

### Solve and Extract Specific Values

```julia
using MPI
MPI.Init()

using HPCMultiGridBarrier
using HPCSparseArrays

sol = fem2d_hpc_solve(Float64; L=3, p=1.0)
sol_native = hpc_to_native(sol)

# Access solution data
z = sol_native.z  # Solution matrix
iters = sum(sol_native.SOL_main.its)  # Total Newton steps
elapsed = sol_native.SOL_main.t_elapsed  # Elapsed time in seconds

println(io0(), "Converged in $iters iterations")
println(io0(), "Elapsed time: $elapsed seconds")
```

## Next Steps

- See the [API Reference](@ref) for detailed function documentation
- Check the `examples/` directory for complete runnable examples
- Consult MultiGridBarrier.jl documentation for barrier method theory
