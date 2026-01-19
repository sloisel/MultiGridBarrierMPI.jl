#!/usr/bin/env julia
# Profile linear solves (MUMPS) at different scales
using MPI
MPI.Init()

using MultiGridBarrier
using HPCMultiGridBarrier
using HPCSparseArrays
using LinearAlgebra
using SparseArrays
import Statistics: mean, median

HPCMultiGridBarrier.Init()

println("="^70)
println("Linear solver scaling: L=5 vs L=6")
println("="^70)

const N_ITER = 5

for L in [5, 6]
    println("\n" * "="^70)
    println("L = $L")
    println("="^70)

    g_hpc = fem2d_hpc(Float64; L=L)
    x_hpc = g_hpc.x
    w_hpc = g_hpc.w
    x_native = x_hpc.A
    w_native = w_hpc.v

    n = size(x_native, 1)
    println("Grid points: $n")

    # Create test system: A*x = b (use Laplacian = dx'*dx + dy'*dy)
    g_native = fem2d(Float64; L=L)
    Dx_native = g_native.operators[:dx]
    Dy_native = g_native.operators[:dy]
    A_native = Dx_native' * Dx_native + Dy_native' * Dy_native + sparse(I, n, n)  # Add I for stability

    Dx_hpc = g_hpc.operators[:dx]
    Dy_mpi = g_hpc.operators[:dy]
    I_mpi = g_hpc.operators[:id]
    A_mpi = Dx_hpc' * Dx_hpc + Dy_mpi' * Dy_mpi + I_mpi

    b_native = ones(n)
    b_mpi = HPCSparseArrays.HPCVector(b_native; partition=w_hpc.partition)

    # Native solve
    println("\nNative linear solve (sparse \\):")
    native_times = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        x = A_native \ b_native
        t = time_ns() - t
        push!(native_times, t)
    end
    println("  Median: $(round(median(native_times)/1e6, digits=1)) ms")

    # MPI solve (MUMPS)
    println("\nMPI linear solve (MUMPS):")
    mpi_times = Float64[]
    for _ in 1:N_ITER
        t = time_ns()
        x = A_mpi \ b_mpi
        t = time_ns() - t
        push!(mpi_times, t)
    end
    println("  Median: $(round(median(mpi_times)/1e6, digits=1)) ms")

    println("\nRatio: $(round(median(mpi_times) / median(native_times), digits=2))x")
end

println("\n" * "="^70)
println("Done.")
