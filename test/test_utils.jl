# Shared test utilities for parameterized testing
# This module provides test configurations for CPU and GPU backends

module TestUtils

using SparseArrays

# Detect Metal availability BEFORE loading LinearAlgebraMPI
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

# Import LinearAlgebraMPI after Metal check
using LinearAlgebraMPI
using LinearAlgebraMPI: VectorMPI, MatrixMPI, SparseMatrixMPI

# Backend configurations: (ScalarType, backend_fn, backend_name)
const CPU_CONFIGS = [
    (Float64, identity, "CPU")
]

const GPU_CONFIGS = if METAL_AVAILABLE
    [
        (Float32, LinearAlgebraMPI.mtl, "Metal")
    ]
else
    Tuple{Type, Function, String}[]
end

const ALL_CONFIGS = [CPU_CONFIGS; GPU_CONFIGS]

"""
    tolerance(T)

Return appropriate tolerance for type T.
"""
tolerance(::Type{Float64}) = 1e-10
tolerance(::Type{Float32}) = 1e-4

"""
    to_cpu(x)

Convert to CPU if on GPU, otherwise return as-is.
Works for both CPU and GPU arrays.
"""
to_cpu(x) = x

# For VectorMPI: return as-is if already CPU
to_cpu(x::VectorMPI{T, Vector{T}}) where T = x
to_cpu(x::SparseMatrixMPI{T, Ti, Vector{T}}) where {T, Ti} = x
to_cpu(x::MatrixMPI{T, Matrix{T}}) where T = x

# GPU versions (only available when Metal is loaded)
if METAL_AVAILABLE
    to_cpu(x::VectorMPI{T, <:Metal.MtlVector}) where T = LinearAlgebraMPI.cpu(x)
    to_cpu(x::SparseMatrixMPI{T, Ti, <:Metal.MtlVector}) where {T, Ti} = LinearAlgebraMPI.cpu(x)
    to_cpu(x::MatrixMPI{T, <:Metal.MtlMatrix}) where T = LinearAlgebraMPI.cpu(x)
end

export METAL_AVAILABLE, CPU_CONFIGS, GPU_CONFIGS, ALL_CONFIGS
export tolerance, to_cpu

end # module
