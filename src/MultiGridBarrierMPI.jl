"""
    MultiGridBarrierMPI

A module that provides a convenient interface for using MultiGridBarrier with MPI
distributed types through LinearAlgebraMPI.

# Exports
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
using MultiGridBarrier: parabolic_solve

# 1D: Create MPI geometry and solve
g1d = fem1d_mpi(Float64; L=4)
sol1d = fem1d_mpi_solve(Float64; L=4, p=1.0, verbose=true)

# 2D: Create MPI geometry and solve
g = fem2d_mpi(Float64; L=3)
sol = fem2d_mpi_solve(Float64; L=3, p=2.0, verbose=true)

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
using MultiGridBarrier: Geometry, AMGBSOL, ParabolicSOL, fem1d, FEM1D, fem3d, FEM3D, parabolic_solve, amgb
using PrecompileTools

# ============================================================================
# MultiGridBarrier API Implementation for LinearAlgebraMPI Types
# ============================================================================

# Import the functions we need to extend
import MultiGridBarrier: amgb_zeros, amgb_all_isfinite, amgb_assert_uniform, amgb_diag, amgb_blockdiag, map_rows, map_rows_gpu, vertex_indices, _raw_array, _to_cpu_array, _rows_to_svectors

# amgb_zeros: Create distributed zero matrices/vectors using Base.zeros from LinearAlgebraMPI
MultiGridBarrier.amgb_zeros(::SparseMatrixMPI{T,Ti,AV}, m, n) where {T,Ti,AV} =
    zeros(SparseMatrixMPI{T,Ti,AV}, m, n)
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:SparseMatrixMPI{T,Ti,AV}}, m, n) where {T,Ti,AV} =
    zeros(SparseMatrixMPI{T,Ti,AV}, m, n)

MultiGridBarrier.amgb_zeros(::MatrixMPI{T,AM}, m, n) where {T,AM} =
    zeros(MatrixMPI{T,AM}, m, n)
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:MatrixMPI{T,AM}}, m, n) where {T,AM} =
    zeros(MatrixMPI{T,AM}, m, n)

# amgb_zeros for vectors (used in multigrid coarsening)
# Note: Only define the fully-parameterized version to ensure GPU types are preserved.
# The <:VectorMPI{T} fallback was causing dispatch issues where Julia selected the
# fallback even for fully-specified GPU types like VectorMPI{T,MtlVector{T}}.
MultiGridBarrier.amgb_zeros(::Type{VectorMPI{T,AV}}, m) where {T,AV} =
    zeros(VectorMPI{T,AV}, m)

# amgb_all_isfinite: Check if all elements are finite
# Uses GPU-friendly broadcasting and MPI reduction to avoid GPU->CPU transfer of full vector
function MultiGridBarrier.amgb_all_isfinite(z::VectorMPI{T,AV}) where {T,AV}
    # Check local elements using broadcasting (GPU-friendly)
    local_all_finite = all(isfinite.(z.v))
    # MPI reduce to get global result
    MPI.Allreduce(local_all_finite, &, MPI.COMM_WORLD)
end

# amgb_assert_uniform: Assert that a scalar value is identical on all MPI ranks
# Gathers all values to rank 0, checks uniformity, and aborts if not uniform
function MultiGridBarrier.amgb_assert_uniform(x::T, msg::String="") where T<:Number
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Gather all values to rank 0
    all_values = MPI.Gather(x, 0, comm)

    # Check uniformity on rank 0
    is_uniform = true
    if rank == 0
        ref_val = all_values[1]
        for i in 2:nranks
            # Use isequal for exact equality (handles NaN correctly: isequal(NaN,NaN)=true)
            if !isequal(all_values[i], ref_val)
                is_uniform = false
                break
            end
        end
    end

    # Broadcast uniformity result to all ranks
    is_uniform = MPI.Bcast(is_uniform, 0, comm)

    if !is_uniform
        # Print error info on rank 0 only (use stdout for visibility)
        if rank == 0
            println("\n" * "="^60)
            println("MPI UNIFORMITY ASSERTION FAILED: $msg")
            println("="^60)
            println("Values across ranks:")
            for i in 1:nranks
                println("  Rank $(i-1): $(all_values[i])")
            end
            println("\nStack trace:")
            for frame in stacktrace()
                println("  ", frame)
            end
            println("="^60)
            flush(stdout)
        end

        # Small delay to ensure output is flushed before abort
        sleep(0.1)

        # Abort all ranks
        MPI.Abort(comm, 1)
    end

    return nothing
end

# Also handle boolean specifically for converged flags
function MultiGridBarrier.amgb_assert_uniform(x::Bool, msg::String="")
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Convert to Int for MPI (some MPI implementations don't handle Bool well)
    x_int = Int32(x)
    all_values = MPI.Gather(x_int, 0, comm)

    # Check uniformity on rank 0
    is_uniform = true
    if rank == 0
        ref_val = all_values[1]
        for i in 2:nranks
            if all_values[i] != ref_val
                is_uniform = false
                break
            end
        end
    end

    # Broadcast uniformity result to all ranks
    is_uniform = MPI.Bcast(is_uniform, 0, comm)

    if !is_uniform
        # Print error info on rank 0 only (use stdout for visibility)
        if rank == 0
            println("\n" * "="^60)
            println("MPI UNIFORMITY ASSERTION FAILED: $msg")
            println("="^60)
            println("Boolean values across ranks:")
            for i in 1:nranks
                println("  Rank $(i-1): $(Bool(all_values[i]))")
            end
            println("\nStack trace:")
            for frame in stacktrace()
                println("  ", frame)
            end
            println("="^60)
            flush(stdout)
        end

        # Small delay to ensure output is flushed before abort
        sleep(0.1)

        MPI.Abort(comm, 1)
    end

    return nothing
end

# amgb_diag: Create diagonal matrix from vector
# SparseMatrixMPI with VectorMPI - preserves vector's array type in nzval
MultiGridBarrier.amgb_diag(::SparseMatrixMPI{T,Ti,AV}, z::VectorMPI{T,AV2}, m=length(z), n=length(z)) where {T,Ti,AV,AV2} =
    spdiagm(m, n, 0 => z)
# SparseMatrixMPI with plain Vector - creates CPU-based sparse
MultiGridBarrier.amgb_diag(::SparseMatrixMPI{T,Ti,AV}, z::Vector{T}, m=length(z), n=length(z)) where {T,Ti,AV} =
    SparseMatrixMPI{T}(spdiagm(m, n, 0 => z))
# MatrixMPI with VectorMPI - returns sparse (diagonal matrices are always sparse)
MultiGridBarrier.amgb_diag(::MatrixMPI{T,AM}, z::VectorMPI{T,AV}, m=length(z), n=length(z)) where {T,AM,AV} =
    spdiagm(m, n, 0 => z)
# MatrixMPI with plain Vector - creates CPU-based sparse
MultiGridBarrier.amgb_diag(::MatrixMPI{T,AM}, z::Vector{T}, m=length(z), n=length(z)) where {T,AM} =
    SparseMatrixMPI{T}(spdiagm(m, n, 0 => z))

# amgb_blockdiag: Block diagonal concatenation
MultiGridBarrier.amgb_blockdiag(args::SparseMatrixMPI{T,Ti,AV}...) where {T,Ti,AV} = blockdiag(args...)

# map_rows and map_rows_gpu: Delegate to LinearAlgebraMPI implementations
# Use AbstractVectorMPI/AbstractMatrixMPI union types for clean dispatch

const AbstractVectorMPI = VectorMPI
const AbstractMatrixMPI = Union{MatrixMPI, SparseMatrixMPI}
const AnyMPI = Union{VectorMPI, MatrixMPI, SparseMatrixMPI}

# map_rows: single method that handles all MPI combinations
# The key insight: if ANY argument is an MPI type, delegate to LinearAlgebraMPI
function MultiGridBarrier.map_rows(f, A::AnyMPI, args...)
    LinearAlgebraMPI.map_rows(f, A, args...)
end

# map_rows_gpu: True GPU execution via LinearAlgebraMPI.map_rows_gpu
# Barrier functions now receive row data via broadcasting (no scalar indexing)
# thanks to Q.args being splatted here. This enables true GPU kernel execution.
function MultiGridBarrier.map_rows_gpu(f, A::AnyMPI, args...)
    LinearAlgebraMPI.map_rows_gpu(f, A, args...)  # True GPU path
end

# _raw_array: Extract raw array from MPI wrappers
# VectorMPI.v is the local vector (e.g., MtlVector on GPU)
# MatrixMPI.A is the local matrix (e.g., MtlMatrix on GPU)
MultiGridBarrier._raw_array(x::VectorMPI) = x.v
MultiGridBarrier._raw_array(x::MatrixMPI) = x.A

# _rows_to_svectors: Extract raw array from MPI wrappers before processing
# This ensures reinterpret views have GPU-compatible isbits parents
MultiGridBarrier._rows_to_svectors(M::MatrixMPI) = MultiGridBarrier._rows_to_svectors(M.A)
MultiGridBarrier._rows_to_svectors(v::VectorMPI) = v.v  # Vectors pass through as raw array

# _to_cpu_array: Convert GPU arrays to CPU for barrier scalar indexing
# For CPU arrays, this is a no-op
MultiGridBarrier._to_cpu_array(x::Array) = x  # Already on CPU, no-op
# For MPI wrappers, extract the underlying array and convert to CPU
MultiGridBarrier._to_cpu_array(x::MatrixMPI) = Array(x.A)
MultiGridBarrier._to_cpu_array(x::VectorMPI) = Array(x.v)

# vertex_indices for MPI types
MultiGridBarrier.vertex_indices(A::VectorMPI) = LinearAlgebraMPI.vertex_indices(A)
MultiGridBarrier.vertex_indices(A::MatrixMPI) = LinearAlgebraMPI.vertex_indices(A)

# ============================================================================
# Type Conversion
# ============================================================================

"""
    native_to_mpi(g_native::Geometry; backend=identity)

**Collective**

Convert a native Geometry object (with Julia arrays) to use MPI distributed types.

# Arguments
- `g_native`: Native Geometry with Julia arrays
- `backend`: Backend conversion function (default: `identity` for CPU).
  Use `LinearAlgebraMPI.mtl` for Metal GPU, `LinearAlgebraMPI.cuda` for CUDA (future).

This is a collective operation. Each rank calls fem2d() to get the same native
geometry, then this function converts:
- x::Matrix{T} -> x::MatrixMPI{T}
- w::Vector{T} -> w::VectorMPI{T}
- operators[key]::SparseMatrixCSC{T,Int} -> operators[key]::SparseMatrixMPI{T}
- subspaces[key][i]::SparseMatrixCSC{T,Int} -> subspaces[key][i]::SparseMatrixMPI{T}

# Example
```julia
# CPU (default)
g_mpi = native_to_mpi(g_native)

# Metal GPU
using Metal
g_mpi = native_to_mpi(g_native; backend=LinearAlgebraMPI.mtl)
```
"""
function native_to_mpi(g_native::Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization};
                       backend=identity) where {T, Discretization}
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

    # Apply backend conversion (e.g., LinearAlgebraMPI.mtl for Metal GPU)
    x_mpi = backend(x_mpi)
    w_mpi = backend(w_mpi)
    for key in keys(operators_mpi)
        operators_mpi[key] = backend(operators_mpi[key])
    end
    for key in keys(subspaces_mpi)
        subspaces_mpi[key] = [backend(m) for m in subspaces_mpi[key]]
    end
    refine_mpi = [backend(r) for r in refine_mpi]
    coarsen_mpi = [backend(c) for c in coarsen_mpi]

    # Determine MPI types for Geometry type parameters
    XType = typeof(x_mpi)
    WType = typeof(w_mpi)
    # Use SparseMatrixMPI{T} as MType - this is a UnionAll type that accepts
    # any Ti and any AV (CPU Vector or GPU MtlVector), allowing mixed backends
    # when GPU_MIN_SIZE threshold causes some operators to stay on CPU
    MType = SparseMatrixMPI{T}
    DType = typeof(g_native.discretization)

    # Create typed dicts and vectors for Geometry constructor
    # Using SparseMatrixMPI{T} allows heterogeneous backends in the same collection
    operators_typed = Dict{Symbol, MType}()
    for key in keys(operators_mpi)
        operators_typed[key] = operators_mpi[key]
    end

    subspaces_typed = Dict{Symbol, Vector{MType}}()
    for key in keys(subspaces_mpi)
        subspaces_typed[key] = Vector{MType}(subspaces_mpi[key])
    end

    refine_typed = Vector{MType}(refine_mpi)
    coarsen_typed = Vector{MType}(coarsen_mpi)

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
    mpi_to_native(g_mpi::Geometry{T, MatrixMPI{T}, VectorMPI{T}, <:SparseMatrixMPI{T}, Discretization}) where {T, Discretization}

**Collective**

Convert an MPI Geometry object (with distributed MPI types) back to native Julia arrays.

This is a collective operation. This function converts:
- x::MatrixMPI{T} -> x::Matrix{T}
- w::VectorMPI{T} -> w::Vector{T}
- operators[key]::SparseMatrixMPI{T} -> operators[key]::SparseMatrixCSC{T,Int}
- subspaces[key][i]::SparseMatrixMPI{T} -> subspaces[key][i]::SparseMatrixCSC{T,Int}
"""
function mpi_to_native(g_mpi::Geometry{T, <:MatrixMPI{T}, <:VectorMPI{T}, <:SparseMatrixMPI{T}, Discretization}) where {T, Discretization}
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

Convert an MPI distributed type to its native Julia (CPU) equivalent.
This gathers distributed data to rank 0 and converts to standard Julia arrays.
"""
_convert_to_native(x::MatrixMPI{T,AM}) where {T,AM} = Matrix(x)
_convert_to_native(x::VectorMPI{T,AV}) where {T,AV} = Vector(x)
_convert_to_native(x::SparseMatrixMPI{T,Ti,AV}) where {T,Ti,AV} = SparseMatrixCSC(x)
_convert_to_native(x) = x  # Fallback for non-MPI types

# ============================================================================
# Public API
# ============================================================================

"""
    fem1d_mpi(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Create an MPI-based Geometry from fem1d parameters.

This function calls `fem1d(kwargs...)` to create a native 1D geometry, then converts
it to use MPI distributed types for distributed computing.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Additional keyword arguments passed to `fem1d()`:
  - `L::Int`: Number of multigrid levels (default: 4), creating 2^L elements

# Returns
A Geometry object with MPI distributed types.

# Example
```julia
using MPI; MPI.Init()
using MultiGridBarrierMPI
g = fem1d_mpi(Float64; L=4)
```
"""
function fem1d_mpi(::Type{T}=Float64; backend=identity, kwargs...) where {T}
    # Create native 1D geometry
    g_native = fem1d(T; kwargs...)

    # Convert to MPI types (with optional backend conversion)
    return native_to_mpi(g_native; backend=backend)
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

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Additional keyword arguments passed to `fem2d()`

# Returns
A Geometry object with MPI distributed types.

# Example
```julia
using MPI; MPI.Init()
using MultiGridBarrierMPI
g = fem2d_mpi(Float64; L=3)
```
"""
function fem2d_mpi(::Type{T}=Float64; backend=identity, kwargs...) where {T}
    # Create native geometry with the specified element type
    g_native = fem2d(T; kwargs...)

    # Convert to MPI types (with optional backend conversion)
    return native_to_mpi(g_native; backend=backend)
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
  - `L`: Number of multigrid levels (passed to fem2d)
  - `p`: Power parameter for the barrier (passed to amgb)
  - `verbose`: Verbosity flag (passed to amgb)
  - Other arguments specific to fem2d or amgb

# Returns
The solution object from `amgb`.

# Example
```julia
sol = fem2d_mpi_solve(Float64; L=3, p=2.0, verbose=true)
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
using MPI; MPI.Init()
using MultiGridBarrierMPI
g = fem3d_mpi(Float64; L=2, k=3)
```
"""
function fem3d_mpi(::Type{T}=Float64; backend=identity, kwargs...) where {T}
    # Create native 3D geometry
    g_native = fem3d(T; kwargs...)

    # Convert to MPI types (with optional backend conversion)
    return native_to_mpi(g_native; backend=backend)
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
export fem1d_mpi, fem1d_mpi_solve
export fem2d_mpi, fem2d_mpi_solve
export fem3d_mpi, fem3d_mpi_solve
export native_to_mpi, mpi_to_native
export amgb

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

    # Precompile 1D and 2D solvers with minimal problem sizes
    # (3D is slower and shares most code paths with 2D)
    fem1d_mpi_solve(; L=1, tol=0.1, verbose=false)
    fem2d_mpi_solve(; L=1, tol=0.1, verbose=false)
end

end # module MultiGridBarrierMPI
