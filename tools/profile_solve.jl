#!/usr/bin/env julia
# Profile the solve phase to find the bottleneck
using MPI
MPI.Init()

using MultiGridBarrier
using MultiGridBarrierMPI
using HPCLinearAlgebra
using LinearAlgebra
import Statistics: mean, median

MultiGridBarrierMPI.Init()

const L = 5  # Use L=5 for faster testing

println("="^70)
println("Profile solve phase at L=$L")
println("="^70)

# Create geometry
g_mpi = fem2d_mpi(Float64; L=L)
g_native = fem2d(Float64; L=L)

println("Grid points: ", sum(g_mpi.x.row_partition) - 2)

# Time the barrier solve
println("\n--- Timing barrier solve ---")

# Native
t_native = time_ns()
sol_native = MultiGridBarrier.amgb(g_native; verbose=false, tol=0.1)
t_native = (time_ns() - t_native) / 1e9
println("Native solve: $(round(t_native, digits=3))s")

# MPI
t_mpi = time_ns()
sol_mpi = MultiGridBarrier.amgb(g_mpi; verbose=false, tol=0.1)
t_mpi = (time_ns() - t_mpi) / 1e9
println("MPI solve:    $(round(t_mpi, digits=3))s")

println("Ratio: $(round(t_mpi / t_native, digits=2))x")

# Now let's profile individual operations within the solve
println("\n--- Profile individual operations ---")

# Get the basic data structures
x_mpi = g_mpi.x
w_mpi = g_mpi.w
x_native = x_mpi.A
w_native = w_mpi.v

# Create a test vector like u
u_test_native = ones(length(w_native))
u_test_mpi = HPCLinearAlgebra.HPCVector(u_test_native; partition=w_mpi.partition)

N_ITER = 10

# Profile f1 function (used in barrier)
f1 = (row_x, row_w, row_u) -> begin
    n = length(row_x)
    result = zeros(eltype(row_u), 1, n)
    for i in 1:n
        result[1, i] = row_w * row_u^2 / row_x[i]^2
    end
    return result
end

f1_simple = (row_x, w, u) -> (w * u^2 / row_x[1]^2, w * u^2 / row_x[2]^2)

# Time map_rows with 3 arguments (like f1)
println("\nmap_rows with scalar function and 3 args:")
f_scalar3 = (row_x, w, u) -> w * u * sum(row_x)

native_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = MultiGridBarrier.map_rows(f_scalar3, x_native, w_native, u_test_native)
    t = time_ns() - t
    push!(native_times, t)
end

mpi_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = HPCLinearAlgebra.map_rows(f_scalar3, x_mpi, w_mpi, u_test_mpi)
    t = time_ns() - t
    push!(mpi_times, t)
end

println("  Native: $(round(median(native_times)/1000, digits=1)) μs")
println("  MPI:    $(round(median(mpi_times)/1000, digits=1)) μs")
println("  Ratio:  $(round(median(mpi_times) / median(native_times), digits=2))x")

# Time map_rows returning row vector (like f1)
println("\nmap_rows with row-vector function (f1-like):")
f_rowvec = (row_x, w, u) -> begin
    val = w * u^2
    [val / row_x[1]^2  val / row_x[2]^2]  # 1x2 matrix
end

native_rv_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = MultiGridBarrier.map_rows(f_rowvec, x_native, w_native, u_test_native)
    t = time_ns() - t
    push!(native_rv_times, t)
end

mpi_rv_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = HPCLinearAlgebra.map_rows(f_rowvec, x_mpi, w_mpi, u_test_mpi)
    t = time_ns() - t
    push!(mpi_rv_times, t)
end

println("  Native: $(round(median(native_rv_times)/1000, digits=1)) μs")
println("  MPI:    $(round(median(mpi_rv_times)/1000, digits=1)) μs")
println("  Ratio:  $(round(median(mpi_rv_times) / median(native_rv_times), digits=2))x")

# Time matrix-vector product (Hessian application)
println("\nMatrix-vector product (matvec):")
# Create a sparse matrix like the Hessian
A_native = g_native.A
A_mpi = g_mpi.A

native_mv_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = A_native * u_test_native
    t = time_ns() - t
    push!(native_mv_times, t)
end

mpi_mv_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = A_mpi * u_test_mpi
    t = time_ns() - t
    push!(mpi_mv_times, t)
end

println("  Native: $(round(median(native_mv_times)/1000, digits=1)) μs")
println("  MPI:    $(round(median(mpi_mv_times)/1000, digits=1)) μs")
println("  Ratio:  $(round(median(mpi_mv_times) / median(native_mv_times), digits=2))x")

# Time dot product
println("\nDot product:")
native_dot_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = dot(u_test_native, u_test_native)
    t = time_ns() - t
    push!(native_dot_times, t)
end

mpi_dot_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = dot(u_test_mpi, u_test_mpi)
    t = time_ns() - t
    push!(mpi_dot_times, t)
end

println("  Native: $(round(median(native_dot_times)/1000, digits=1)) μs")
println("  MPI:    $(round(median(mpi_dot_times)/1000, digits=1)) μs")
println("  Ratio:  $(round(median(mpi_dot_times) / median(native_dot_times), digits=2))x")

println("\nDone.")
