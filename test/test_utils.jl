# Shared test utilities for parameterized testing
# This module provides test configurations for CPU and GPU backends

module TestUtils

using SparseArrays
using MPI

# Detect Metal availability BEFORE loading HPCLinearAlgebra
# (Metal must be loaded first for GPU detection to work)
const METAL_AVAILABLE = try
    using Metal
    Metal.functional()
catch e
    false
end

if METAL_AVAILABLE
    @info "Metal is available for GPU tests"
end

# Detect CUDA availability BEFORE loading HPCLinearAlgebra
# (CUDA must be loaded first for GPU detection to work)
const CUDA_AVAILABLE = try
    using CUDA, NCCL, CUDSS_jll
    CUDA.functional()
catch e
    false
end

if CUDA_AVAILABLE
    @info "CUDA is available for GPU tests"
end

# Import HPCLinearAlgebra after GPU checks - this triggers extension loading
using HPCLinearAlgebra
using HPCLinearAlgebra: HPCVector, HPCMatrix, HPCSparseMatrix
using HPCLinearAlgebra: BACKEND_CPU_MPI, HPCBackend, to_backend

# Backend configurations: (ScalarType, backend_instance, backend_name)
# Tests should use: HPCSparseMatrix(sparse_matrix, backend) etc.
# NOTE: GPU backends require MPI to be initialized, so we create them lazily

const CPU_CONFIGS = [
    (Float64, BACKEND_CPU_MPI, "CPU")
]

# Cached GPU backend instances (created lazily after MPI.Init)
const _METAL_BACKEND = Ref{Union{Nothing, HPCBackend}}(nothing)
const _CUDA_BACKEND = Ref{Union{Nothing, HPCBackend}}(nothing)

function _get_metal_backend()
    if _METAL_BACKEND[] === nothing && METAL_AVAILABLE
        _METAL_BACKEND[] = HPCLinearAlgebra.backend_metal_mpi(MPI.COMM_WORLD)
    end
    return _METAL_BACKEND[]
end

function _get_cuda_backend()
    if _CUDA_BACKEND[] === nothing && CUDA_AVAILABLE
        _CUDA_BACKEND[] = HPCLinearAlgebra.backend_cuda_mpi(MPI.COMM_WORLD)
    end
    return _CUDA_BACKEND[]
end

# Metal configs (for 1D and 2D) - lazy creation
function get_metal_configs()
    if METAL_AVAILABLE
        backend = _get_metal_backend()
        return [(Float32, backend, "Metal")]
    else
        return Tuple{Type, HPCBackend, String}[]
    end
end

# CUDA configs (for 2D ONLY - cuDSS has a bug with tridiagonal matrices in 1D)
# NOTE: CUDA32 temporarily disabled - hangs during solve (cuDSS Float32 issue?)
function get_cuda_configs()
    if CUDA_AVAILABLE
        backend = _get_cuda_backend()
        return [
            # (Float32, backend, "CUDA32"),  # DISABLED: hangs during solve
            (Float64, backend, "CUDA64")
        ]
    else
        return Tuple{Type, HPCBackend, String}[]
    end
end

# Lazy-evaluated configs that include GPU backends
# These call the get_* functions which require MPI to be initialized
const METAL_CONFIGS = Tuple{Type, HPCBackend, String}[]  # Placeholder for backward compat
const CUDA_CONFIGS = Tuple{Type, HPCBackend, String}[]   # Placeholder for backward compat

# 1D configs: CPU + Metal only (no CUDA due to cuDSS tridiagonal bug)
function get_all_configs_1d()
    return [CPU_CONFIGS; get_metal_configs()]
end

# 2D configs: CPU + Metal + CUDA (2D matrices are not tridiagonal)
# DEBUG: Only CUDA64 to debug hang
function get_all_configs_2d()
    return get_cuda_configs()  # [CPU_CONFIGS; get_metal_configs(); get_cuda_configs()]
end

# Eager constants for backward compatibility (CPU only)
const ALL_CONFIGS_1D = CPU_CONFIGS
const ALL_CONFIGS_2D = Tuple{Type, HPCBackend, String}[]

# Legacy alias (for backward compatibility - uses CPU configs only)
const ALL_CONFIGS = CPU_CONFIGS

"""
    tolerance(T)

Return appropriate tolerance for type T.
"""
tolerance(::Type{Float64}) = 1e-10
tolerance(::Type{Float32}) = 1e-4

"""
    to_cpu(x)

Convert to CPU backend if on GPU, otherwise return as-is.
Works for both CPU and GPU backends.
"""
to_cpu(x) = x

# For HPC types: check if already CPU backend, otherwise convert
function to_cpu(x::HPCVector{T,B}) where {T, B<:HPCBackend}
    if B.parameters[1] === HPCLinearAlgebra.DeviceCPU
        return x
    else
        return to_backend(x, BACKEND_CPU_MPI)
    end
end

function to_cpu(x::HPCMatrix{T,B}) where {T, B<:HPCBackend}
    if B.parameters[1] === HPCLinearAlgebra.DeviceCPU
        return x
    else
        return to_backend(x, BACKEND_CPU_MPI)
    end
end

function to_cpu(x::HPCSparseMatrix{T,Ti,B}) where {T, Ti, B<:HPCBackend}
    if B.parameters[1] === HPCLinearAlgebra.DeviceCPU
        return x
    else
        return to_backend(x, BACKEND_CPU_MPI)
    end
end

export METAL_AVAILABLE, CUDA_AVAILABLE
export CPU_CONFIGS, METAL_CONFIGS, CUDA_CONFIGS
export ALL_CONFIGS, ALL_CONFIGS_1D, ALL_CONFIGS_2D
export get_metal_configs, get_cuda_configs, get_all_configs_1d, get_all_configs_2d
export tolerance, to_cpu

end # module
