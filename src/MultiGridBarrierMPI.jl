"""
    MultiGridBarrierMPI

A module that provides a convenient interface for using MultiGridBarrier with MPI
distributed types through LinearAlgebraMPI.

# Exports
- `Init`: Initialize MultiGridBarrierMPI with MPI
- `fem1d_mpi`: Creates an MPI-based Geometry from fem1d parameters
- `fem1d_mpi_solve`: Solves a fem1d problem using amgb with MPI types
- `fem2d_mpi`: Creates an MPI-based Geometry from fem2d parameters
- `fem2d_mpi_solve`: Solves a fem2d problem using amgb with MPI types
- `fem3d_mpi`: Creates an MPI-based Geometry from fem3d parameters
- `fem3d_mpi_solve`: Solves a fem3d problem using amgb with MPI types
- `native_to_mpi`: Converts native Geometry to MPI distributed types
- `mpi_to_native`: Converts MPI Geometry, AMGBSOL, or ParabolicSOL back to native Julia types

# Usage
```julia
using MPI
MPI.Init()
using MultiGridBarrierMPI
MultiGridBarrierMPI.Init()
using MultiGridBarrier: parabolic_solve

# 1D: Create MPI geometry and solve
g1d = fem1d_mpi(Float64; L=4)
sol1d = fem1d_mpi_solve(Float64; L=4, p=1.0, verbose=true)

# 2D: Create MPI geometry and solve
g = fem2d_mpi(Float64; maxh=0.1)
sol = fem2d_mpi_solve(Float64; maxh=0.1, p=2.0, verbose=true)

# 3D: Create MPI geometry and solve
g3d = fem3d_mpi(Float64; L=2, k=3)
sol3d = fem3d_mpi_solve(Float64; L=2, k=3, p=1.0, verbose=true)

# Time-dependent (parabolic) solve - use parabolic_solve directly with MPI geometry
g = fem1d_mpi(Float64; L=3)
sol_para = parabolic_solve(g; h=0.1, p=2.0, verbose=true)

# Convert solution back to native types for analysis
sol_native = mpi_to_native(sol)
```
"""
module MultiGridBarrierMPI

using MPI
using LinearAlgebraMPI
using LinearAlgebraMPI: VectorMPI, MatrixMPI, SparseMatrixMPI, io0
using LinearAlgebraMPI: VectorMPI_local, MatrixMPI_local, SparseMatrixMPI_local
using LinearAlgebra
using SparseArrays
using MultiGridBarrier
using MultiGridBarrier: Geometry, AMGBSOL, ParabolicSOL, fem1d, FEM1D, fem3d, FEM3D, parabolic_solve
using PrecompileTools

# ============================================================================
# MultiGridBarrier API Implementation for LinearAlgebraMPI Types
# ============================================================================

# Import the functions we need to extend
import MultiGridBarrier: amgb_zeros, amgb_all_isfinite, amgb_diag, amgb_blockdiag, map_rows

# amgb_zeros: Create zero matrices with appropriate type
MultiGridBarrier.amgb_zeros(::SparseMatrixMPI{T}, m, n) where {T} =
    SparseMatrixMPI{T}(spzeros(T, m, n))
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:SparseMatrixMPI{T}}, m, n) where {T} =
    SparseMatrixMPI{T}(spzeros(T, m, n))
MultiGridBarrier.amgb_zeros(::MatrixMPI{T}, m, n) where {T} =
    MatrixMPI(zeros(T, m, n))
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:MatrixMPI{T}}, m, n) where {T} =
    MatrixMPI(zeros(T, m, n))

# amgb_zeros for vectors (used in multigrid coarsening)
MultiGridBarrier.amgb_zeros(::Type{VectorMPI{T}}, m) where {T} = VectorMPI(zeros(T, m))

# amgb_all_isfinite: Check if all elements are finite
MultiGridBarrier.amgb_all_isfinite(z::VectorMPI{T}) where {T} = all(isfinite.(Vector(z)))

# amgb_diag: Create diagonal matrix from vector
MultiGridBarrier.amgb_diag(::SparseMatrixMPI{T}, z::VectorMPI{T}, m=length(z), n=length(z)) where {T} =
    spdiagm(m, n, 0 => z)
MultiGridBarrier.amgb_diag(::SparseMatrixMPI{T}, z::Vector{T}, m=length(z), n=length(z)) where {T} =
    SparseMatrixMPI{T}(spdiagm(m, n, 0 => z))
MultiGridBarrier.amgb_diag(::MatrixMPI{T}, z::VectorMPI{T}, m=length(z), n=length(z)) where {T} =
    spdiagm(m, n, 0 => z)
MultiGridBarrier.amgb_diag(::MatrixMPI{T}, z::Vector{T}, m=length(z), n=length(z)) where {T} =
    SparseMatrixMPI{T}(spdiagm(m, n, 0 => z))

# amgb_blockdiag: Block diagonal concatenation
MultiGridBarrier.amgb_blockdiag(args::SparseMatrixMPI{T}...) where {T} = blockdiag(args...)

# ============================================================================
# map_rows Implementation
# ============================================================================

"""
    MultiGridBarrier.map_rows(f, A::Union{VectorMPI{T}, MatrixMPI{T}}...) where {T}

**MPI Collective**

Apply a function `f` to corresponding rows across distributed MPI vectors and matrices.

This is a thin wrapper around `LinearAlgebraMPI.map_rows`. See that function for
full documentation.

# Examples
```julia
# Example 1: Sum rows of a matrix
B = MatrixMPI(randn(5, 3))
sums = map_rows(sum, B)  # Returns VectorMPI with 5 elements

# Example 2: Compute [sum, product] for each row (returns matrix)
stats = map_rows(x -> [sum(x), prod(x)]', B)  # Returns 5×2 MatrixMPI

# Example 3: Combine matrix and vector row-wise
C = VectorMPI(randn(5))
combined = map_rows((x, y) -> [sum(x), prod(x), y[1]]', B, C)  # Returns 5×3 MatrixMPI
```
"""
function MultiGridBarrier.map_rows(f, A::Union{VectorMPI{T}, MatrixMPI{T}}...) where {T}
    LinearAlgebraMPI.map_rows(f, A...)
end

# ============================================================================
# Type Conversion
# ============================================================================

"""
    native_to_mpi(g_native::Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}) where {T, Discretization}

**Collective**

Convert a native Geometry object (with Julia arrays) to use MPI distributed types.

This is a collective operation. Each rank calls fem2d() to get the same native
geometry, then this function converts:
- x::Matrix{T} -> x::MatrixMPI{T}
- w::Vector{T} -> w::VectorMPI{T}
- operators[key]::SparseMatrixCSC{T,Int} -> operators[key]::SparseMatrixMPI{T}
- subspaces[key][i]::SparseMatrixCSC{T,Int} -> subspaces[key][i]::SparseMatrixMPI{T}
"""
function native_to_mpi(g_native::Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}) where {T, Discretization}
    # Convert x (geometry coordinates) to MatrixMPI (dense)
    x_mpi = MatrixMPI(g_native.x)

    # Convert w (weights) to VectorMPI
    w_mpi = VectorMPI(g_native.w)

    # Convert all operators to SparseMatrixMPI
    # Sort keys to ensure deterministic order across all ranks
    operators_mpi = Dict{Symbol, Any}()
    for key in sort(collect(keys(g_native.operators)))
        op = g_native.operators[key]
        operators_mpi[key] = SparseMatrixMPI{T}(op)
    end

    # Convert all subspace matrices to SparseMatrixMPI
    # Sort keys and use explicit loops to ensure all ranks iterate in sync
    subspaces_mpi = Dict{Symbol, Vector{Any}}()
    for key in sort(collect(keys(g_native.subspaces)))
        subspace_vec = g_native.subspaces[key]
        mpi_vec = Vector{Any}(undef, length(subspace_vec))
        for i in 1:length(subspace_vec)
            mpi_vec[i] = SparseMatrixMPI{T}(subspace_vec[i])
        end
        subspaces_mpi[key] = mpi_vec
    end

    # Convert refine and coarsen vectors to SparseMatrixMPI
    refine_mpi = Vector{Any}(undef, length(g_native.refine))
    for i in 1:length(g_native.refine)
        refine_mpi[i] = SparseMatrixMPI{T}(g_native.refine[i])
    end

    coarsen_mpi = Vector{Any}(undef, length(g_native.coarsen))
    for i in 1:length(g_native.coarsen)
        coarsen_mpi[i] = SparseMatrixMPI{T}(g_native.coarsen[i])
    end

    # Determine MPI types for Geometry type parameters
    XType = typeof(x_mpi)
    WType = typeof(w_mpi)
    MType = typeof(operators_mpi[:id])  # Use id operator as representative
    DType = typeof(g_native.discretization)

    # Create typed dicts and vectors for Geometry constructor
    operators_typed = Dict{Symbol, MType}()
    for key in keys(operators_mpi)
        operators_typed[key] = operators_mpi[key]
    end

    subspaces_typed = Dict{Symbol, Vector{MType}}()
    for key in keys(subspaces_mpi)
        subspaces_typed[key] = convert(Vector{MType}, subspaces_mpi[key])
    end

    refine_typed = convert(Vector{MType}, refine_mpi)
    coarsen_typed = convert(Vector{MType}, coarsen_mpi)

    # Create new Geometry with MPI types using explicit type parameters
    return Geometry{T, XType, WType, MType, DType}(
        g_native.discretization,
        x_mpi,
        w_mpi,
        subspaces_typed,
        operators_typed,
        refine_typed,
        coarsen_typed
    )
end

"""
    mpi_to_native(g_mpi::Geometry{T, MatrixMPI{T}, VectorMPI{T}, SparseMatrixMPI{T}, Discretization}) where {T, Discretization}

**Collective**

Convert an MPI Geometry object (with distributed MPI types) back to native Julia arrays.

This is a collective operation. This function converts:
- x::MatrixMPI{T} -> x::Matrix{T}
- w::VectorMPI{T} -> w::Vector{T}
- operators[key]::SparseMatrixMPI{T} -> operators[key]::SparseMatrixCSC{T,Int}
- subspaces[key][i]::SparseMatrixMPI{T} -> subspaces[key][i]::SparseMatrixCSC{T,Int}
"""
function mpi_to_native(g_mpi::Geometry{T, MatrixMPI{T}, VectorMPI{T}, SparseMatrixMPI{T}, Discretization}) where {T, Discretization}
    # Convert x (geometry coordinates) from MatrixMPI to Matrix
    x_native = Matrix(g_mpi.x)

    # Convert w (weights) from VectorMPI to Vector
    w_native = Vector(g_mpi.w)

    # Convert all operators from SparseMatrixMPI to SparseMatrixCSC
    # Sort keys to ensure deterministic order across all ranks
    operators_native = Dict{Symbol, SparseMatrixCSC{T,Int}}()
    for key in sort(collect(keys(g_mpi.operators)))
        op = g_mpi.operators[key]
        operators_native[key] = SparseMatrixCSC(op)
    end

    # Convert all subspace matrices from SparseMatrixMPI to SparseMatrixCSC
    # Sort keys and use explicit loops to ensure all ranks iterate in sync
    subspaces_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Int}}}()
    for key in sort(collect(keys(g_mpi.subspaces)))
        subspace_vec = g_mpi.subspaces[key]
        native_vec = Vector{SparseMatrixCSC{T,Int}}(undef, length(subspace_vec))
        for i in 1:length(subspace_vec)
            native_vec[i] = SparseMatrixCSC(subspace_vec[i])
        end
        subspaces_native[key] = native_vec
    end

    # Convert refine and coarsen vectors from SparseMatrixMPI to SparseMatrixCSC
    refine_native = Vector{SparseMatrixCSC{T,Int}}(undef, length(g_mpi.refine))
    for i in 1:length(g_mpi.refine)
        refine_native[i] = SparseMatrixCSC(g_mpi.refine[i])
    end

    coarsen_native = Vector{SparseMatrixCSC{T,Int}}(undef, length(g_mpi.coarsen))
    for i in 1:length(g_mpi.coarsen)
        coarsen_native[i] = SparseMatrixCSC(g_mpi.coarsen[i])
    end

    # Create new Geometry with native Julia types using explicit type parameters
    return Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}(
        g_mpi.discretization,
        x_native,
        w_native,
        subspaces_native,
        operators_native,
        refine_native,
        coarsen_native
    )
end

"""
    mpi_to_native(sol_mpi::AMGBSOL{T, XType, WType, MType, Discretization}) where {T, XType, WType, MType, Discretization}

**Collective**

Convert an AMGBSOL solution object from MPI types back to native Julia types.

This is a collective operation that performs a deep conversion of the solution structure:
- z: MatrixMPI{T} -> Matrix{T} or VectorMPI{T} -> Vector{T}
- SOL_feasibility: NamedTuple with MPI types -> NamedTuple with native types
- SOL_main: NamedTuple with MPI types -> NamedTuple with native types
- geometry: Geometry with MPI types -> Geometry with native types
"""
function mpi_to_native(sol_mpi::AMGBSOL{T, XType, WType, MType, Discretization}) where {T, XType, WType, MType, Discretization}
    # Convert z - handles both MatrixMPI and VectorMPI types
    z_native = _convert_to_native(sol_mpi.z)

    # Helper function to recursively convert NamedTuples with MPI types
    function convert_namedtuple(nt)
        if nt === nothing
            return nothing
        end
        # Convert each field in the NamedTuple
        converted_fields = []
        for (name, value) in pairs(nt)
            converted_value = convert_value(value)
            push!(converted_fields, name => converted_value)
        end
        return NamedTuple(converted_fields)
    end

    # Helper function to convert individual values
    function convert_value(value)
        if isa(value, MatrixMPI) || isa(value, VectorMPI) || isa(value, SparseMatrixMPI)
            return _convert_to_native(value)
        elseif isa(value, Array)
            # Recursively convert arrays
            return map(convert_value, value)
        else
            # For non-MPI types (numbers, strings, etc.), return as-is
            return value
        end
    end

    # Convert SOL_feasibility and SOL_main NamedTuples
    SOL_feasibility_native = convert_namedtuple(sol_mpi.SOL_feasibility)
    SOL_main_native = convert_namedtuple(sol_mpi.SOL_main)

    # Convert the geometry
    geometry_native = mpi_to_native(sol_mpi.geometry)

    # Determine native types
    ZType = typeof(z_native)

    # Create and return the native AMGBSOL
    return AMGBSOL{T, ZType, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}(
        z_native,
        SOL_feasibility_native,
        SOL_main_native,
        sol_mpi.log,
        geometry_native
    )
end

"""
    mpi_to_native(sol_mpi::ParabolicSOL{T, XType, WType, MType, Discretization}) where {T, XType, WType, MType, Discretization}

**Collective**

Convert a ParabolicSOL solution object from MPI types back to native Julia types.

This is a collective operation that performs a deep conversion of the parabolic solution:
- geometry: Geometry with MPI types -> Geometry with native types
- ts: Vector{T} (unchanged, already native)
- u: Vector{MatrixMPI{T}} -> Vector{Matrix{T}} (each time snapshot converted)

# Example
```julia
g = fem2d_mpi(Float64; L=2)
sol_mpi = parabolic_solve(g; h=0.5, p=1.0)
sol_native = mpi_to_native(sol_mpi)
```
"""
function mpi_to_native(sol_mpi::ParabolicSOL{T, XType, WType, MType, Discretization}) where {T, XType, WType, MType, Discretization}
    # Convert the geometry
    geometry_native = mpi_to_native(sol_mpi.geometry)

    # ts is already Vector{T}, no conversion needed
    ts_native = sol_mpi.ts

    # Convert each time snapshot in u
    u_native = [_convert_to_native(u_k) for u_k in sol_mpi.u]

    # Determine native X type from converted u
    XTypeNative = typeof(u_native[1])

    # Create and return the native ParabolicSOL
    return ParabolicSOL{T, XTypeNative, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}(
        geometry_native,
        ts_native,
        u_native
    )
end

"""
    _convert_to_native(x)

Convert an MPI distributed type to its native Julia equivalent.
"""
_convert_to_native(x::MatrixMPI{T}) where {T} = Matrix(x)
_convert_to_native(x::VectorMPI{T}) where {T} = Vector(x)
_convert_to_native(x::SparseMatrixMPI{T}) where {T} = SparseMatrixCSC(x)
_convert_to_native(x) = x  # Fallback for non-MPI types

# ============================================================================
# Public API
# ============================================================================

# Module-level flag to track whether MultiGridBarrierMPI has been initialized
const MGBMPI_INITIALIZED = Ref(false)

"""
    Init()

**Collective**

Initialize MultiGridBarrierMPI by ensuring MPI is initialized.

This function should be called once before using any MultiGridBarrierMPI functionality.
It will initialize MPI if not already initialized.

# Example
```julia
using MPI
MPI.Init()
using MultiGridBarrierMPI
MultiGridBarrierMPI.Init()
```
"""
function Init()
    # Only initialize once
    if MGBMPI_INITIALIZED[]
        return
    end

    # Initialize MPI if not already initialized
    if !MPI.Initialized()
        MPI.Init()
    end

    # Get rank for output
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    if rank == 0
        println("MultiGridBarrierMPI initialized")
    end

    MGBMPI_INITIALIZED[] = true
end

"""
    fem1d_mpi(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Create an MPI-based Geometry from fem1d parameters.

This function calls `fem1d(kwargs...)` to create a native 1D geometry, then converts
it to use MPI distributed types for distributed computing.

Note: Call `MultiGridBarrierMPI.Init()` before using this function.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Additional keyword arguments passed to `fem1d()`:
  - `L::Int`: Number of multigrid levels (default: 4), creating 2^L elements

# Returns
A Geometry object with MPI distributed types.

# Example
```julia
MultiGridBarrierMPI.Init()
g = fem1d_mpi(Float64; L=4)
```
"""
function fem1d_mpi(::Type{T}=Float64; kwargs...) where {T}
    # Create native 1D geometry
    g_native = fem1d(T; kwargs...)

    # Convert to MPI types
    return native_to_mpi(g_native)
end

"""
    fem1d_mpi_solve(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Solve a fem1d problem using amgb with MPI distributed types.

This is a convenience function that combines `fem1d_mpi` and `amgb` into a
single call. It creates an MPI-based 1D geometry and solves the barrier problem.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Keyword arguments passed to both `fem1d_mpi` and `amgb`
  - `L::Int`: Number of multigrid levels (passed to fem1d)
  - `p`: Power parameter for the barrier (passed to amgb)
  - `verbose`: Verbosity flag (passed to amgb)
  - Other arguments specific to fem1d or amgb

# Returns
The solution object from `amgb`.

# Example
```julia
sol = fem1d_mpi_solve(Float64; L=4, p=1.0, verbose=true)
println("Solution norm: ", norm(sol.z))
```
"""
function fem1d_mpi_solve(::Type{T}=Float64; kwargs...) where {T}
    # Create MPI 1D geometry
    g = fem1d_mpi(T; kwargs...)

    # Solve using amgb (amgb auto-detects 1D from geometry.discretization)
    return amgb(g; kwargs...)
end

"""
    fem2d_mpi(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Create an MPI-based Geometry from fem2d parameters.

This function calls `fem2d(kwargs...)` to create a native geometry, then converts
it to use MPI distributed types for distributed computing.

Note: Call `MultiGridBarrierMPI.Init()` before using this function.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Additional keyword arguments passed to `fem2d()`

# Returns
A Geometry object with MPI distributed types.

# Example
```julia
MultiGridBarrierMPI.Init()
g = fem2d_mpi(Float64; maxh=0.1)
```
"""
function fem2d_mpi(::Type{T}=Float64; kwargs...) where {T}
    # Create native geometry
    g_native = fem2d(; kwargs...)

    # Convert to MPI types
    return native_to_mpi(g_native)
end

"""
    fem2d_mpi_solve(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Solve a fem2d problem using amgb with MPI distributed types.

This is a convenience function that combines `fem2d_mpi` and `amgb` into a
single call. It creates an MPI-based geometry and solves the barrier problem.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Keyword arguments passed to both `fem2d_mpi` and `amgb`
  - `maxh`: Maximum mesh size (passed to fem2d)
  - `p`: Power parameter for the barrier (passed to amgb)
  - `verbose`: Verbosity flag (passed to amgb)
  - Other arguments specific to fem2d or amgb

# Returns
The solution object from `amgb`.

# Example
```julia
sol = fem2d_mpi_solve(Float64; maxh=0.1, p=2.0, verbose=true)
println("Solution norm: ", norm(sol.z))
```
"""
function fem2d_mpi_solve(::Type{T}=Float64; kwargs...) where {T}
    # Create MPI geometry
    g = fem2d_mpi(T; kwargs...)

    # Solve using amgb
    return amgb(g; kwargs...)
end

"""
    fem3d_mpi(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Create an MPI-based Geometry from fem3d parameters.

This function calls `fem3d(kwargs...)` to create a native 3D geometry, then converts
it to use MPI distributed types for distributed computing.

Note: Call `MultiGridBarrierMPI.Init()` before using this function.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Additional keyword arguments passed to `fem3d()`:
  - `L::Int`: Number of multigrid levels (default: 2)
  - `k::Int`: Polynomial order of elements (default: 3)
  - `K`: Coarse Q1 mesh as an N×3 matrix (optional, defaults to unit cube)

# Returns
A Geometry object with MPI distributed types.

# Example
```julia
MultiGridBarrierMPI.Init()
g = fem3d_mpi(Float64; L=2, k=3)
```
"""
function fem3d_mpi(::Type{T}=Float64; kwargs...) where {T}
    # Create native 3D geometry
    g_native = fem3d(T; kwargs...)

    # Convert to MPI types
    return native_to_mpi(g_native)
end

"""
    fem3d_mpi_solve(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Solve a fem3d problem using amgb with MPI distributed types.

This is a convenience function that combines `fem3d_mpi` and `amgb` into a
single call. It creates an MPI-based 3D geometry and solves the barrier problem.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Keyword arguments passed to both `fem3d_mpi` and `amgb`
  - `L::Int`: Number of multigrid levels (passed to fem3d)
  - `k::Int`: Polynomial order of elements (passed to fem3d)
  - `p`: Power parameter for the barrier (passed to amgb)
  - `verbose`: Verbosity flag (passed to amgb)
  - `D`: Operator structure matrix (passed to amgb, defaults to 3D operators)
  - `f`: Source term function (passed to amgb, defaults to 3D source)
  - `g`: Boundary condition function (passed to amgb, defaults to 3D BCs)
  - Other arguments specific to fem3d or amgb

# Returns
The solution object from `amgb`.

# Example
```julia
sol = fem3d_mpi_solve(Float64; L=2, k=3, p=1.0, verbose=true)
println("Solution norm: ", norm(sol.z))
```
"""
function fem3d_mpi_solve(::Type{T}=Float64;
    D = [:u :id; :u :dx; :u :dy; :u :dz; :s :id],
    f = (x) -> T[0.5, 0.0, 0.0, 0.0, 1.0],
    g = (x) -> T[x[1]^2 + x[2]^2 + x[3]^2, 100.0],
    kwargs...) where {T}
    # Create MPI 3D geometry
    geom = fem3d_mpi(T; kwargs...)

    # Solve using amgb with 3D-specific defaults
    return amgb(geom; D=D, f=f, g=g, kwargs...)
end

# Export the public API
export Init
export fem1d_mpi, fem1d_mpi_solve
export fem2d_mpi, fem2d_mpi_solve
export fem3d_mpi, fem3d_mpi_solve
export native_to_mpi, mpi_to_native

# ============================================================================
# Precompilation Workload
# ============================================================================

@compile_workload begin
    # === MPI Jail Escape ===
    # When precompiling under mpiexec, the subprocess inherits MPI environment
    # variables but isn't part of the MPI job. Clean them to allow MPI.Init()
    # to succeed as a fresh single-rank process.
    for k in collect(keys(ENV))
        if startswith(k, "PMI") || startswith(k, "PMIX") || startswith(k, "OMPI_") || startswith(k, "MPI_")
            delete!(ENV, k)
        end
    end

    MPI.Init()
    Init()

    # Precompile 1D, 2D, and 3D solvers with minimal problem sizes
    fem1d_mpi_solve(; L=1, tol=0.1, verbose=false)
    fem2d_mpi_solve(; L=1, tol=0.1, verbose=false)
    fem3d_mpi_solve(; L=1, tol=0.1, verbose=false)
end

end # module MultiGridBarrierMPI
