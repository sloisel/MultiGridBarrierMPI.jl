#!/usr/bin/env julia
# Profile operations at different scales to find O(n^2) behavior
using MPI
MPI.Init()

using MultiGridBarrier
using MultiGridBarrierMPI
using HPCLinearAlgebra
using LinearAlgebra
import Statistics: mean, median

MultiGridBarrierMPI.Init()

println("="^70)
println("Scaling analysis: L=5 vs L=6")
println("="^70)

const N_ITER = 20

for L in [5, 6]
    println("\n" * "="^70)
    println("L = $L")
    println("="^70)

    g_mpi = fem2d_mpi(Float64; L=L)
    x_mpi = g_mpi.x
    w_mpi = g_mpi.w
    x_native = x_mpi.A
    w_native = w_mpi.v

    n = size(x_native, 1)
    println("Grid points: $n")

    # Create test vectors
    u_native = ones(n)
    u_mpi = HPCLinearAlgebra.HPCVector(u_native; partition=w_mpi.partition)

    # 1. map_rows with scalar function
    println("\n1. map_rows (scalar):")
    f_scalar = (row_x, w) -> w * sum(row_x)

    native_times = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = MultiGridBarrier.map_rows(f_scalar, x_native, w_native)
        t = time_ns() - t
        push!(native_times, t)
    end

    mpi_times = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = HPCLinearAlgebra.map_rows(f_scalar, x_mpi, w_mpi)
        t = time_ns() - t
        push!(mpi_times, t)
    end

    println("  Native: $(round(median(native_times)/1000, digits=1)) μs")
    println("  MPI:    $(round(median(mpi_times)/1000, digits=1)) μs")
    println("  Ratio:  $(round(median(mpi_times) / median(native_times), digits=2))x")

    # 2. Sparse matrix-vector product
    println("\n2. Sparse matvec (A*u):")
    # Get the sparse matrix from operators
    A_mpi = g_mpi.operators[:id]
    g_native = fem2d(Float64; L=L)
    A_native = g_native.operators[:id]

    native_mv = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = A_native * u_native
        t = time_ns() - t
        push!(native_mv, t)
    end

    mpi_mv = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = A_mpi * u_mpi
        t = time_ns() - t
        push!(mpi_mv, t)
    end

    println("  Native: $(round(median(native_mv)/1000, digits=1)) μs")
    println("  MPI:    $(round(median(mpi_mv)/1000, digits=1)) μs")
    println("  Ratio:  $(round(median(mpi_mv) / median(native_mv), digits=2))x")

    # 3. Dot product
    println("\n3. Dot product:")
    native_dot = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = dot(u_native, u_native)
        t = time_ns() - t
        push!(native_dot, t)
    end

    mpi_dot = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = dot(u_mpi, u_mpi)
        t = time_ns() - t
        push!(mpi_dot, t)
    end

    println("  Native: $(round(median(native_dot)/1000, digits=1)) μs")
    println("  MPI:    $(round(median(mpi_dot)/1000, digits=1)) μs")
    println("  Ratio:  $(round(median(mpi_dot) / median(native_dot), digits=2))x")

    # 4. Vector operations
    println("\n4. Vector add (u + v):")
    v_native = 2 .* u_native
    v_mpi = HPCLinearAlgebra.HPCVector(v_native; partition=w_mpi.partition)

    native_add = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = u_native + v_native
        t = time_ns() - t
        push!(native_add, t)
    end

    mpi_add = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = u_mpi + v_mpi
        t = time_ns() - t
        push!(mpi_add, t)
    end

    println("  Native: $(round(median(native_add)/1000, digits=1)) μs")
    println("  MPI:    $(round(median(mpi_add)/1000, digits=1)) μs")
    println("  Ratio:  $(round(median(mpi_add) / median(native_add), digits=2))x")

    # 5. map_rows with row-vector function (like barrier f1)
    println("\n5. map_rows (row-vector, f1-like):")
    f_rowvec = (row_x, w) -> begin
        val = w * sum(row_x)
        [val/row_x[1]  val/row_x[2]]  # 1x2 matrix
    end

    native_rv = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = MultiGridBarrier.map_rows(f_rowvec, x_native, w_native)
        t = time_ns() - t
        push!(native_rv, t)
    end

    mpi_rv = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        result = HPCLinearAlgebra.map_rows(f_rowvec, x_mpi, w_mpi)
        t = time_ns() - t
        push!(mpi_rv, t)
    end

    println("  Native: $(round(median(native_rv)/1000, digits=1)) μs")
    println("  MPI:    $(round(median(mpi_rv)/1000, digits=1)) μs")
    println("  Ratio:  $(round(median(mpi_rv) / median(native_rv), digits=2))x")
end

println("\n" * "="^70)
println("Done.")
