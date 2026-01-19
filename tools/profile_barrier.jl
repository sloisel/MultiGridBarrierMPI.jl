#!/usr/bin/env julia
# Profile barrier function evaluations (F0, F1, F2)
using MPI
MPI.Init()

using MultiGridBarrier
using MultiGridBarrierMPI
using HPCSparseArrays
using LinearAlgebra
using SparseArrays
import Statistics: mean, median

MultiGridBarrierMPI.Init()

println("="^70)
println("Barrier function profiling")
println("="^70)

const N_ITER = 20

for L in [5, 6]
    println("\n" * "="^70)
    println("L = $L")
    println("="^70)

    # Create geometry
    g_mpi = fem2d_mpi(Float64; L=L)
    g_native = fem2d(Float64; L=L)

    n = size(g_native.x, 1)
    println("Grid points: $n")

    # Create initial solution (like in amgb)
    dim = 2
    z_native = hcat(g_native.x, ones(n, dim+2))  # n x (2+dim+2) = n x 6
    z_mpi = HPCSparseArrays.HPCMatrix(z_native; row_partition=g_mpi.x.row_partition)

    # Get the operators we need
    x_native = g_native.x
    w_native = g_native.w
    x_mpi = g_mpi.x
    w_mpi = g_mpi.w

    # 1. Profile map_rows with scalar function
    f0_like = (row_z, w) -> w * (row_z[3]^2 + row_z[4]^2)

    println("\n1. f0-like (scalar map_rows):")
    native_f0 = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = sum(MultiGridBarrier.map_rows(f0_like, z_native, g_native.w))
        t = time_ns() - t
        push!(native_f0, t)
    end

    mpi_f0 = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = sum(HPCSparseArrays.map_rows(f0_like, z_mpi, w_mpi))
        t = time_ns() - t
        push!(mpi_f0, t)
    end

    println("  Native: $(round(median(native_f0)/1000, digits=1)) μs")
    println("  MPI:    $(round(median(mpi_f0)/1000, digits=1)) μs")
    println("  Ratio:  $(round(median(mpi_f0) / median(native_f0), digits=2))x")

    # 2. f1-like: returns row vector
    f1_like = (row_z, w) -> begin
        grad = [2*w*row_z[3], 2*w*row_z[4], 0.0, 0.0, 0.0, 0.0]
        reshape(grad, 1, 6)
    end

    println("\n2. f1-like (row-vector map_rows):")
    native_f1 = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = MultiGridBarrier.map_rows(f1_like, z_native, g_native.w)
        t = time_ns() - t
        push!(native_f1, t)
    end

    mpi_f1 = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = HPCSparseArrays.map_rows(f1_like, z_mpi, w_mpi)
        t = time_ns() - t
        push!(mpi_f1, t)
    end

    println("  Native: $(round(median(native_f1)/1000, digits=1)) μs")
    println("  MPI:    $(round(median(mpi_f1)/1000, digits=1)) μs")
    println("  Ratio:  $(round(median(mpi_f1) / median(native_f1), digits=2))x")

    # 3. sum of HPCVector
    println("\n3. sum of HPCVector:")
    v_native = w_native
    v_mpi = w_mpi

    native_sum = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = sum(v_native)
        t = time_ns() - t
        push!(native_sum, t)
    end

    mpi_sum = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = sum(v_mpi)
        t = time_ns() - t
        push!(mpi_sum, t)
    end

    println("  Native: $(round(median(native_sum)/1000, digits=1)) μs")
    println("  MPI:    $(round(median(mpi_sum)/1000, digits=1)) μs")
    println("  Ratio:  $(round(median(mpi_sum) / median(native_sum), digits=2))x")

    # 4. Multiple map_rows in sequence
    println("\n4. Sequence of 5 map_rows calls:")
    native_seq = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        r1 = sum(MultiGridBarrier.map_rows(f0_like, z_native, g_native.w))
        r2 = MultiGridBarrier.map_rows(f1_like, z_native, g_native.w)
        r3 = sum(MultiGridBarrier.map_rows(f0_like, z_native, g_native.w))
        r4 = MultiGridBarrier.map_rows(f1_like, z_native, g_native.w)
        r5 = sum(MultiGridBarrier.map_rows(f0_like, z_native, g_native.w))
        t = time_ns() - t
        push!(native_seq, t)
    end

    mpi_seq = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        r1 = sum(HPCSparseArrays.map_rows(f0_like, z_mpi, w_mpi))
        r2 = HPCSparseArrays.map_rows(f1_like, z_mpi, w_mpi)
        r3 = sum(HPCSparseArrays.map_rows(f0_like, z_mpi, w_mpi))
        r4 = HPCSparseArrays.map_rows(f1_like, z_mpi, w_mpi)
        r5 = sum(HPCSparseArrays.map_rows(f0_like, z_mpi, w_mpi))
        t = time_ns() - t
        push!(mpi_seq, t)
    end

    println("  Native: $(round(median(native_seq)/1000, digits=1)) μs")
    println("  MPI:    $(round(median(mpi_seq)/1000, digits=1)) μs")
    println("  Ratio:  $(round(median(mpi_seq) / median(native_seq), digits=2))x")
    expected_native = 3*median(native_f0) + 2*median(native_f1)
    expected_mpi = 3*median(mpi_f0) + 2*median(mpi_f1)
    println("  Expected from individual: ~$(round(expected_native/1000, digits=1)) μs native, ~$(round(expected_mpi/1000, digits=1)) μs MPI")
end

println("\n" * "="^70)
println("Done.")
