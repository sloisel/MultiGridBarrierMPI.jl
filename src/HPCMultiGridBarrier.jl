"""
    HPCMultiGridBarrier

A module that provides a convenient interface for using MultiGridBarrier with HPC
distributed types through HPCSparseArrays.

# Exports
- `fem1d_hpc`: Creates an HPC-based Geometry from fem1d parameters
- `fem1d_hpc_solve`: Solves a fem1d problem using amgb with HPC types
- `fem2d_hpc`: Creates an HPC-based Geometry from fem2d parameters
- `fem2d_hpc_solve`: Solves a fem2d problem using amgb with HPC types
- `fem3d_hpc`: Creates an HPC-based Geometry from fem3d parameters
- `fem3d_hpc_solve`: Solves a fem3d problem using amgb with HPC types
- `native_to_hpc`: Converts native Geometry to HPC distributed types
- `hpc_to_native`: Converts HPC Geometry, AMGBSOL, or ParabolicSOL back to native Julia types

# Usage
```julia
using MPI
MPI.Init()
using HPCMultiGridBarrier
using MultiGridBarrier: parabolic_solve

# 1D: Create HPC geometry and solve
g1d = fem1d_hpc(Float64; L=4)
sol1d = fem1d_hpc_solve(Float64; L=4, p=1.0, verbose=true)

# 2D: Create HPC geometry and solve
g = fem2d_hpc(Float64; L=3)
sol = fem2d_hpc_solve(Float64; L=3, p=2.0, verbose=true)

# 3D: Create HPC geometry and solve
g3d = fem3d_hpc(Float64; L=2, k=3)
sol3d = fem3d_hpc_solve(Float64; L=2, k=3, p=1.0, verbose=true)

# Time-dependent (parabolic) solve - use parabolic_solve directly with HPC geometry
g = fem1d_hpc(Float64; L=3)
sol_para = parabolic_solve(g; h=0.1, p=2.0, verbose=true)

# Convert solution back to native types for analysis
sol_native = hpc_to_native(sol)
```
"""
module HPCMultiGridBarrier

using MPI
using HPCSparseArrays
using HPCSparseArrays: HPCVector, HPCMatrix, HPCSparseMatrix, io0
using HPCSparseArrays: HPCVector_local, HPCMatrix_local, HPCSparseMatrix_local
using HPCSparseArrays: HPCBackend, backend_cpu_mpi, eltype_backend, indextype_backend, comm_allreduce
using LinearAlgebra
using SparseArrays
using MultiGridBarrier
using MultiGridBarrier: Geometry, AMGBSOL, ParabolicSOL, fem1d, FEM1D, fem3d, FEM3D, parabolic_solve, amgb
using PrecompileTools

# ============================================================================
# MultiGridBarrier API Implementation for HPCSparseArrays Types
# ============================================================================

# Import the functions we need to extend
import MultiGridBarrier: amgb_zeros, amgb_all_isfinite, amgb_diag, amgb_blockdiag, map_rows, map_rows_gpu, vertex_indices, _raw_array, _to_cpu_array, _rows_to_svectors

# amgb_zeros: Create distributed zero matrices/vectors using Base.zeros from HPCSparseArrays
# New API: zeros(T, Ti, HPCSparseMatrix, backend, m, n) - extract backend from existing matrix
MultiGridBarrier.amgb_zeros(A::HPCSparseMatrix{T,Ti,B}, m, n) where {T,Ti,B} =
    zeros(T, Ti, HPCSparseMatrix, A.backend, m, n)
MultiGridBarrier.amgb_zeros(A::LinearAlgebra.Adjoint{T, <:HPCSparseMatrix{T,Ti,B}}, m, n) where {T,Ti,B} =
    zeros(T, Ti, HPCSparseMatrix, parent(A).backend, m, n)

# New API: zeros(T, HPCMatrix, backend, m, n) - extract backend from existing matrix
MultiGridBarrier.amgb_zeros(A::HPCMatrix{T,B}, m, n) where {T,B} =
    zeros(T, HPCMatrix, A.backend, m, n)
MultiGridBarrier.amgb_zeros(A::LinearAlgebra.Adjoint{T, <:HPCMatrix{T,B}}, m, n) where {T,B} =
    zeros(T, HPCMatrix, parent(A).backend, m, n)

# amgb_zeros for vectors (used in multigrid coarsening)
# New API: zeros(T, HPCVector, backend, m) - need to map backend type to instance
# This is a bit hacky but works for the known backend types
using HPCSparseArrays: backend_cpu_serial

# Cache for GPU backend instances (created lazily when needed)
# Key: (T, Ti) -> backend instance
const _GPU_BACKEND_CACHE = Dict{Tuple{DataType,DataType,DataType},Any}()

function _backend_instance_from_type(::Type{B}) where B
    # HPCBackend is now parameterized as HPCBackend{T, Ti, Device, Comm, Solver}
    T = B.parameters[1]
    Ti = B.parameters[2]
    device_type = B.parameters[3]

    if device_type === HPCSparseArrays.DeviceCPU
        comm_type = B.parameters[4]
        if comm_type === HPCSparseArrays.CommSerial
            return backend_cpu_serial(T, Ti)
        else
            return backend_cpu_mpi(T, Ti)
        end
    elseif device_type === HPCSparseArrays.DeviceCUDA
        cache_key = (T, Ti, device_type)
        if !haskey(_GPU_BACKEND_CACHE, cache_key)
            _GPU_BACKEND_CACHE[cache_key] = HPCSparseArrays.backend_cuda_mpi(T, Ti)
        end
        return _GPU_BACKEND_CACHE[cache_key]
    elseif device_type === HPCSparseArrays.DeviceMetal
        cache_key = (T, Ti, device_type)
        if !haskey(_GPU_BACKEND_CACHE, cache_key)
            _GPU_BACKEND_CACHE[cache_key] = HPCSparseArrays.backend_metal_mpi(T, Ti)
        end
        return _GPU_BACKEND_CACHE[cache_key]
    else
        error("Unknown backend device type: $device_type")
    end
end

MultiGridBarrier.amgb_zeros(::Type{HPCVector{T,B}}, m) where {T,B} =
    zeros(T, HPCVector, _backend_instance_from_type(B), m)

# amgb_all_isfinite: Check if all elements are finite
# Uses GPU-friendly broadcasting and reduction to avoid GPU->CPU transfer of full vector
# Uses comm_allreduce which handles both MPI and serial modes
function MultiGridBarrier.amgb_all_isfinite(z::HPCVector{T,B}) where {T,B}
    # Check local elements using broadcasting (GPU-friendly)
    local_all_finite = all(isfinite.(z.v))
    # Reduce across ranks (no-op for CommSerial, MPI.Allreduce for CommMPI)
    comm_allreduce(z.backend.comm, local_all_finite, &)
end

function MultiGridBarrier.amgb_all_isfinite(z::HPCMatrix{T,B}) where {T,B}
    # Check local elements using broadcasting (GPU-friendly)
    local_all_finite = all(isfinite.(z.A))
    # Reduce across ranks (no-op for CommSerial, MPI.Allreduce for CommMPI)
    comm_allreduce(z.backend.comm, local_all_finite, &)
end

# amgb_diag: Create diagonal matrix from vector
# HPCSparseMatrix with HPCVector - preserves vector's backend and Ti
MultiGridBarrier.amgb_diag(::HPCSparseMatrix{T,Ti,B}, z::HPCVector{T,B2}, m=length(z), n=length(z)) where {T,Ti,B,B2} =
    _convert_hpc_sparse_indices(spdiagm(m, n, 0 => z), Ti)
# HPCSparseMatrix with plain Vector - extract backend from sparse matrix, preserve Ti
MultiGridBarrier.amgb_diag(A::HPCSparseMatrix{T,Ti,B}, z::Vector{T}, m=length(z), n=length(z)) where {T,Ti,B} =
    HPCSparseMatrix(_convert_sparse_indices(spdiagm(m, n, 0 => z), Ti), A.backend)
# HPCMatrix with HPCVector - returns sparse (diagonal matrices are always sparse), use Int32
MultiGridBarrier.amgb_diag(::HPCMatrix{T,B}, z::HPCVector{T,B2}, m=length(z), n=length(z)) where {T,B,B2} =
    _convert_hpc_sparse_indices(spdiagm(m, n, 0 => z), Int32)
# HPCMatrix with plain Vector - extract backend from dense matrix, use Int32 for memory efficiency
MultiGridBarrier.amgb_diag(A::HPCMatrix{T,B}, z::Vector{T}, m=length(z), n=length(z)) where {T,B} =
    HPCSparseMatrix(_convert_sparse_indices(spdiagm(m, n, 0 => z), Int32), A.backend)

# amgb_blockdiag: Block diagonal concatenation
MultiGridBarrier.amgb_blockdiag(args::HPCSparseMatrix{T,Ti,AV}...) where {T,Ti,AV} = blockdiag(args...)

# map_rows and map_rows_gpu: Delegate to HPCSparseArrays implementations
# Use AbstractHPCVector/AbstractHPCMatrix union types for clean dispatch

const AbstractHPCVector = HPCVector
const AbstractHPCMatrix = Union{HPCMatrix, HPCSparseMatrix}
const AnyHPC = Union{HPCVector, HPCMatrix, HPCSparseMatrix}

# map_rows: single method that handles all MPI combinations
# The key insight: if ANY argument is an MPI type, delegate to HPCSparseArrays
function MultiGridBarrier.map_rows(f, A::AnyHPC, args...)
    HPCSparseArrays.map_rows(f, A, args...)
end

# map_rows_gpu: True GPU execution via HPCSparseArrays.map_rows_gpu
# Barrier functions now receive row data via broadcasting (no scalar indexing)
# thanks to Q.args being splatted here. This enables true GPU kernel execution.
function MultiGridBarrier.map_rows_gpu(f, A::AnyHPC, args...)
    HPCSparseArrays.map_rows_gpu(f, A, args...)  # True GPU path
end

# _raw_array: Extract raw array from MPI wrappers
# HPCVector.v is the local vector (e.g., MtlVector on GPU)
# HPCMatrix.A is the local matrix (e.g., MtlMatrix on GPU)
MultiGridBarrier._raw_array(x::HPCVector) = x.v
MultiGridBarrier._raw_array(x::HPCMatrix) = x.A

# _rows_to_svectors: Extract raw array from MPI wrappers before processing
# This ensures reinterpret views have GPU-compatible isbits parents
MultiGridBarrier._rows_to_svectors(M::HPCMatrix) = MultiGridBarrier._rows_to_svectors(M.A)
MultiGridBarrier._rows_to_svectors(v::HPCVector) = v.v  # Vectors pass through as raw array

# _to_cpu_array: Convert GPU arrays to CPU for barrier scalar indexing
# For CPU arrays, this is a no-op
MultiGridBarrier._to_cpu_array(x::Array) = x  # Already on CPU, no-op
# For MPI wrappers, extract the underlying array and convert to CPU
MultiGridBarrier._to_cpu_array(x::HPCMatrix) = Array(x.A)
MultiGridBarrier._to_cpu_array(x::HPCVector) = Array(x.v)

# vertex_indices for MPI types
MultiGridBarrier.vertex_indices(A::HPCVector) = HPCSparseArrays.vertex_indices(A)
MultiGridBarrier.vertex_indices(A::HPCMatrix) = HPCSparseArrays.vertex_indices(A)

# ============================================================================
# Type Conversion
# ============================================================================

"""
    _convert_sparse_indices(A::SparseMatrixCSC{T}, ::Type{Ti}) -> SparseMatrixCSC{T,Ti}

Convert a SparseMatrixCSC to use a different index type Ti.
Used to convert Int64 indices to Int32 for memory efficiency in MPI operations.
"""
function _convert_sparse_indices(A::SparseMatrixCSC{T}, ::Type{Ti}) where {T, Ti<:Integer}
    SparseMatrixCSC{T,Ti}(A.m, A.n, Ti.(A.colptr), Ti.(A.rowval), A.nzval)
end

"""
    _convert_hpc_sparse_indices(A::HPCSparseMatrix, ::Type{Ti}) -> HPCSparseMatrix

Convert an HPCSparseMatrix to use a different index type Ti.
Used to convert Int64 indices to Int32 for memory efficiency.
"""
function _convert_hpc_sparse_indices(A::HPCSparseMatrix{T,Ti_old,B}, ::Type{Ti}) where {T, Ti_old, Ti<:Integer, B}
    Ti_old == Ti && return A  # Already the right type
    HPCSparseMatrix{T,Ti,B}(
        A.structural_hash, A.row_partition, A.col_partition,
        Ti.(A.col_indices), Ti.(A.colptr), Ti.(A.rowval), A.nzval,
        A.nrows_local, A.ncols_compressed, nothing, A.has_sorted_rows,
        Ti.(A.cached_rowptr_base), Ti.(A.cached_colval_base), A.backend
    )
end

"""
    native_to_hpc(g_native::Geometry; Ti=Int32, backend=nothing)

**Collective**

Convert a native Geometry object (with Julia arrays) to use MPI distributed types.

# Arguments
- `g_native`: Native Geometry with Julia arrays
- `Ti`: Index type for sparse matrices (default: `Int32` for memory efficiency).
  Use `Int` or `Int64` for problems with >2 billion non-zeros.
- `backend`: HPCBackend to use. If `nothing` (default), creates `backend_cpu_mpi(T, Ti)`
  where T is inferred from the geometry. Pass a backend explicitly for GPU backends:
  `backend_metal_mpi(T, Ti)` for Metal GPU, `backend_cuda_mpi(T, Ti)` for CUDA.

This is a collective operation. Each rank calls fem2d() to get the same native
geometry, then this function converts:
- x::Matrix{T} -> x::HPCMatrix{T}
- w::Vector{T} -> w::HPCVector{T}
- operators[key]::SparseMatrixCSC{T,Int} -> operators[key]::HPCSparseMatrix{T,Ti}
- subspaces[key][i]::SparseMatrixCSC{T,Int} -> subspaces[key][i]::HPCSparseMatrix{T,Ti}

# Example
```julia
# CPU with MPI (default, uses Int32 indices for memory efficiency)
g_hpc = native_to_hpc(g_native)

# Use Int64 indices for very large problems
g_hpc = native_to_hpc(g_native; Ti=Int64)

# Metal GPU with MPI
using Metal
g_hpc = native_to_hpc(g_native; backend=backend_metal_mpi(Float64, Int32))
```
"""
function native_to_hpc(g_native::Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization};
                       Ti::Type{<:Integer}=Int32,
                       backend::Union{Nothing,HPCBackend}=nothing) where {T, Discretization}
    # Create default backend if not provided (with matching T and Ti)
    actual_backend = backend === nothing ? backend_cpu_mpi(T, Ti) : backend
    # Convert x (geometry coordinates) to HPCMatrix (dense)
    x_hpc = HPCMatrix(g_native.x, actual_backend)

    # Convert w (weights) to HPCVector
    w_hpc = HPCVector(g_native.w, actual_backend)

    # Helper to convert and wrap sparse matrices with the specified index type Ti
    convert_sparse = op -> HPCSparseMatrix(_convert_sparse_indices(op, Ti), actual_backend)

    # Convert all operators to HPCSparseMatrix
    # Sort keys to ensure deterministic order across all ranks
    operators_hpc = Dict{Symbol, Any}()
    for key in sort(collect(keys(g_native.operators)))
        op = g_native.operators[key]
        operators_hpc[key] = convert_sparse(op)
    end

    # Convert all subspace matrices to HPCSparseMatrix
    # Sort keys and use explicit loops to ensure all ranks iterate in sync
    subspaces_hpc = Dict{Symbol, Vector{Any}}()
    for key in sort(collect(keys(g_native.subspaces)))
        subspace_vec = g_native.subspaces[key]
        hpc_vec = Vector{Any}(undef, length(subspace_vec))
        for i in 1:length(subspace_vec)
            hpc_vec[i] = convert_sparse(subspace_vec[i])
        end
        subspaces_hpc[key] = hpc_vec
    end

    # Convert refine and coarsen vectors to HPCSparseMatrix
    refine_hpc = Vector{Any}(undef, length(g_native.refine))
    for i in 1:length(g_native.refine)
        refine_hpc[i] = convert_sparse(g_native.refine[i])
    end

    coarsen_hpc = Vector{Any}(undef, length(g_native.coarsen))
    for i in 1:length(g_native.coarsen)
        coarsen_hpc[i] = convert_sparse(g_native.coarsen[i])
    end

    # Determine MPI types for Geometry type parameters
    XType = typeof(x_hpc)
    WType = typeof(w_hpc)
    # Use HPCSparseMatrix{T} as MType - this is a UnionAll type that accepts
    # any Ti and any backend, allowing mixed backends when GPU_MIN_SIZE
    # threshold causes some operators to stay on CPU
    MType = HPCSparseMatrix{T}
    DType = typeof(g_native.discretization)

    # Create typed dicts and vectors for Geometry constructor
    # Using HPCSparseMatrix{T} allows heterogeneous backends in the same collection
    operators_typed = Dict{Symbol, MType}()
    for key in keys(operators_hpc)
        operators_typed[key] = operators_hpc[key]
    end

    subspaces_typed = Dict{Symbol, Vector{MType}}()
    for key in keys(subspaces_hpc)
        subspaces_typed[key] = Vector{MType}(subspaces_hpc[key])
    end

    refine_typed = Vector{MType}(refine_hpc)
    coarsen_typed = Vector{MType}(coarsen_hpc)

    # Create new Geometry with MPI types using explicit type parameters
    return Geometry{T, XType, WType, MType, DType}(
        g_native.discretization,
        x_hpc,
        w_hpc,
        subspaces_typed,
        operators_typed,
        refine_typed,
        coarsen_typed
    )
end

"""
    hpc_to_native(g_hpc::Geometry{T, <:HPCMatrix{T}, <:HPCVector{T}, <:HPCSparseMatrix{T}, Discretization}) where {T, Discretization}

**Collective**

Convert an MPI Geometry object (with distributed MPI types) back to native Julia arrays.

This is a collective operation. This function converts:
- x::HPCMatrix{T} -> x::Matrix{T}
- w::HPCVector{T} -> w::Vector{T}
- operators[key]::HPCSparseMatrix{T,Ti} -> operators[key]::SparseMatrixCSC{T,Ti}
- subspaces[key][i]::HPCSparseMatrix{T,Ti} -> subspaces[key][i]::SparseMatrixCSC{T,Ti}

The index type Ti is preserved from the input HPCSparseMatrix.
"""
function hpc_to_native(g_hpc::Geometry{T, <:HPCMatrix{T}, <:HPCVector{T}, <:HPCSparseMatrix{T}, Discretization}) where {T, Discretization}
    # Convert x (geometry coordinates) from HPCMatrix to Matrix
    x_native = Matrix(g_hpc.x)

    # Convert w (weights) from HPCVector to Vector
    w_native = Vector(g_hpc.w)

    # Extract Ti from the first operator matrix (all matrices should have the same Ti)
    first_op = first(values(g_hpc.operators))
    Ti = eltype(first_op.rowptr)

    # Convert all operators from HPCSparseMatrix to SparseMatrixCSC
    # Sort keys to ensure deterministic order across all ranks
    operators_native = Dict{Symbol, SparseMatrixCSC{T,Ti}}()
    for key in sort(collect(keys(g_hpc.operators)))
        op = g_hpc.operators[key]
        operators_native[key] = SparseMatrixCSC(op)
    end

    # Convert all subspace matrices from HPCSparseMatrix to SparseMatrixCSC
    # Sort keys and use explicit loops to ensure all ranks iterate in sync
    subspaces_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Ti}}}()
    for key in sort(collect(keys(g_hpc.subspaces)))
        subspace_vec = g_hpc.subspaces[key]
        native_vec = Vector{SparseMatrixCSC{T,Ti}}(undef, length(subspace_vec))
        for i in 1:length(subspace_vec)
            native_vec[i] = SparseMatrixCSC(subspace_vec[i])
        end
        subspaces_native[key] = native_vec
    end

    # Convert refine and coarsen vectors from HPCSparseMatrix to SparseMatrixCSC
    refine_native = Vector{SparseMatrixCSC{T,Ti}}(undef, length(g_hpc.refine))
    for i in 1:length(g_hpc.refine)
        refine_native[i] = SparseMatrixCSC(g_hpc.refine[i])
    end

    coarsen_native = Vector{SparseMatrixCSC{T,Ti}}(undef, length(g_hpc.coarsen))
    for i in 1:length(g_hpc.coarsen)
        coarsen_native[i] = SparseMatrixCSC(g_hpc.coarsen[i])
    end

    # Create new Geometry with native Julia types using explicit type parameters
    return Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Ti}, Discretization}(
        g_hpc.discretization,
        x_native,
        w_native,
        subspaces_native,
        operators_native,
        refine_native,
        coarsen_native
    )
end

"""
    hpc_to_native(sol_hpc::AMGBSOL{T, XType, WType, MType, Discretization}) where {T, XType, WType, MType, Discretization}

**Collective**

Convert an AMGBSOL solution object from MPI types back to native Julia types.

This is a collective operation that performs a deep conversion of the solution structure:
- z: HPCMatrix{T} -> Matrix{T} or HPCVector{T} -> Vector{T}
- SOL_feasibility: NamedTuple with MPI types -> NamedTuple with native types
- SOL_main: NamedTuple with MPI types -> NamedTuple with native types
- geometry: Geometry with MPI types -> Geometry with native types
"""
function hpc_to_native(sol_hpc::AMGBSOL{T, XType, WType, MType, Discretization}) where {T, XType, WType, MType, Discretization}
    # Convert z - handles both HPCMatrix and HPCVector types
    z_native = _convert_to_native(sol_hpc.z)

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
        if isa(value, HPCMatrix) || isa(value, HPCVector) || isa(value, HPCSparseMatrix)
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
    SOL_feasibility_native = convert_namedtuple(sol_hpc.SOL_feasibility)
    SOL_main_native = convert_namedtuple(sol_hpc.SOL_main)

    # Convert the geometry
    geometry_native = hpc_to_native(sol_hpc.geometry)

    # Determine native types
    ZType = typeof(z_native)

    # Extract the matrix type from the geometry
    MTypeNative = typeof(geometry_native).parameters[4]

    # Create and return the native AMGBSOL
    return AMGBSOL{T, ZType, Vector{T}, MTypeNative, Discretization}(
        z_native,
        SOL_feasibility_native,
        SOL_main_native,
        sol_hpc.log,
        geometry_native
    )
end

"""
    hpc_to_native(sol_hpc::ParabolicSOL{T, XType, WType, MType, Discretization}) where {T, XType, WType, MType, Discretization}

**Collective**

Convert a ParabolicSOL solution object from MPI types back to native Julia types.

This is a collective operation that performs a deep conversion of the parabolic solution:
- geometry: Geometry with MPI types -> Geometry with native types
- ts: Vector{T} (unchanged, already native)
- u: Vector{HPCMatrix{T}} -> Vector{Matrix{T}} (each time snapshot converted)

# Example
```julia
g = fem2d_hpc(Float64; L=2)
sol_hpc = parabolic_solve(g; h=0.5, p=1.0)
sol_native = hpc_to_native(sol_hpc)
```
"""
function hpc_to_native(sol_hpc::ParabolicSOL{T, XType, WType, MType, Discretization}) where {T, XType, WType, MType, Discretization}
    # Convert the geometry
    geometry_native = hpc_to_native(sol_hpc.geometry)

    # ts is already Vector{T}, no conversion needed
    ts_native = sol_hpc.ts

    # Convert each time snapshot in u
    u_native = [_convert_to_native(u_k) for u_k in sol_hpc.u]

    # Determine native X type from converted u
    XTypeNative = typeof(u_native[1])

    # Extract Ti from the geometry's matrix type
    MTypeNative = typeof(geometry_native).parameters[4]

    # Create and return the native ParabolicSOL
    return ParabolicSOL{T, XTypeNative, Vector{T}, MTypeNative, Discretization}(
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
_convert_to_native(x::HPCMatrix{T,AM}) where {T,AM} = Matrix(x)
_convert_to_native(x::HPCVector{T,AV}) where {T,AV} = Vector(x)
_convert_to_native(x::HPCSparseMatrix{T,Ti,AV}) where {T,Ti,AV} = SparseMatrixCSC(x)
_convert_to_native(x) = x  # Fallback for non-MPI types

# ============================================================================
# Public API
# ============================================================================

"""
    fem1d_hpc(::Type{T}=Float64; kwargs...) where {T}

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
using HPCMultiGridBarrier
g = fem1d_hpc(Float64; L=4)
```
"""
function fem1d_hpc(::Type{T}=Float64; Ti::Type{<:Integer}=Int32, backend::Union{Nothing,HPCBackend}=nothing, kwargs...) where {T}
    # Create native 1D geometry
    g_native = fem1d(T; kwargs...)

    # Convert to HPC types with specified backend (native_to_hpc handles nothing → backend_cpu_mpi(T,Ti))
    return native_to_hpc(g_native; Ti=Ti, backend=backend)
end

"""
    fem1d_hpc_solve(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Solve a fem1d problem using amgb with MPI distributed types.

This is a convenience function that combines `fem1d_hpc` and `amgb` into a
single call. It creates an MPI-based 1D geometry and solves the barrier problem.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Keyword arguments passed to both `fem1d_hpc` and `amgb`
  - `L::Int`: Number of multigrid levels (passed to fem1d)
  - `p`: Power parameter for the barrier (passed to amgb)
  - `verbose`: Verbosity flag (passed to amgb)
  - Other arguments specific to fem1d or amgb

# Returns
The solution object from `amgb`.

# Example
```julia
sol = fem1d_hpc_solve(Float64; L=4, p=1.0, verbose=true)
println("Solution norm: ", norm(sol.z))
```
"""
function fem1d_hpc_solve(::Type{T}=Float64; kwargs...) where {T}
    # Create MPI 1D geometry
    g = fem1d_hpc(T; kwargs...)

    # Solve using amgb (amgb auto-detects 1D from geometry.discretization)
    return amgb(g; kwargs...)
end

"""
    fem2d_hpc(::Type{T}=Float64; kwargs...) where {T}

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
using HPCMultiGridBarrier
g = fem2d_hpc(Float64; L=3)
```
"""
function fem2d_hpc(::Type{T}=Float64; Ti::Type{<:Integer}=Int32, backend::Union{Nothing,HPCBackend}=nothing, kwargs...) where {T}
    # Create native geometry with the specified element type
    g_native = fem2d(T; kwargs...)

    # Convert to HPC types with specified backend (native_to_hpc handles nothing → backend_cpu_mpi(T,Ti))
    return native_to_hpc(g_native; Ti=Ti, backend=backend)
end

"""
    fem2d_hpc_solve(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Solve a fem2d problem using amgb with MPI distributed types.

This is a convenience function that combines `fem2d_hpc` and `amgb` into a
single call. It creates an MPI-based geometry and solves the barrier problem.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Keyword arguments passed to both `fem2d_hpc` and `amgb`
  - `L`: Number of multigrid levels (passed to fem2d)
  - `p`: Power parameter for the barrier (passed to amgb)
  - `verbose`: Verbosity flag (passed to amgb)
  - Other arguments specific to fem2d or amgb

# Returns
The solution object from `amgb`.

# Example
```julia
sol = fem2d_hpc_solve(Float64; L=3, p=2.0, verbose=true)
println("Solution norm: ", norm(sol.z))
```
"""
function fem2d_hpc_solve(::Type{T}=Float64; kwargs...) where {T}
    # Create MPI geometry
    g = fem2d_hpc(T; kwargs...)

    # Solve using amgb
    return amgb(g; kwargs...)
end

"""
    fem3d_hpc(::Type{T}=Float64; kwargs...) where {T}

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
using HPCMultiGridBarrier
g = fem3d_hpc(Float64; L=2, k=3)
```
"""
function fem3d_hpc(::Type{T}=Float64; Ti::Type{<:Integer}=Int32, backend::Union{Nothing,HPCBackend}=nothing, kwargs...) where {T}
    # Create native 3D geometry
    g_native = fem3d(T; kwargs...)

    # Convert to HPC types with specified backend (native_to_hpc handles nothing → backend_cpu_mpi(T,Ti))
    return native_to_hpc(g_native; Ti=Ti, backend=backend)
end

"""
    fem3d_hpc_solve(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Solve a fem3d problem using amgb with MPI distributed types.

This is a convenience function that combines `fem3d_hpc` and `amgb` into a
single call. It creates an MPI-based 3D geometry and solves the barrier problem.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Keyword arguments passed to both `fem3d_hpc` and `amgb`
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
sol = fem3d_hpc_solve(Float64; L=2, k=3, p=1.0, verbose=true)
println("Solution norm: ", norm(sol.z))
```
"""
function fem3d_hpc_solve(::Type{T}=Float64;
    D = [:u :id; :u :dx; :u :dy; :u :dz; :s :id],
    f = (x) -> T[0.5, 0.0, 0.0, 0.0, 1.0],
    g = (x) -> T[x[1]^2 + x[2]^2 + x[3]^2, 100.0],
    kwargs...) where {T}
    # Create MPI 3D geometry
    geom = fem3d_hpc(T; kwargs...)

    # Solve using amgb with 3D-specific defaults
    return amgb(geom; D=D, f=f, g=g, kwargs...)
end

# Export the public API
export fem1d_hpc, fem1d_hpc_solve
export fem2d_hpc, fem2d_hpc_solve
export fem3d_hpc, fem3d_hpc_solve
export native_to_hpc, hpc_to_native
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
    fem1d_hpc_solve(; L=1, tol=0.1, verbose=false)
    fem2d_hpc_solve(; L=1, tol=0.1, verbose=false)
end

end # module HPCMultiGridBarrier
