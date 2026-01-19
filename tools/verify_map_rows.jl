#!/usr/bin/env julia
# Verify that map_rows uses the optimized path
using MPI
MPI.Init()

using MultiGridBarrier
using MultiGridBarrierMPI
using HPCLinearAlgebra
using HPCLinearAlgebra: HPCVector, HPCMatrix
import Statistics: mean, median

MultiGridBarrierMPI.Init()

const L = 6
const N_ITER = 100

println("="^70)
println("Verify map_rows optimization at L=$L")
println("="^70)

g_mpi = fem2d_mpi(Float64; L=L)
x_mpi = g_mpi.x
w_mpi = g_mpi.w

x_native = x_mpi.A
w_native = w_mpi.v

println("Local rows: ", size(x_native, 1))

f = (row_x, w) -> w * sum(row_x)

# Time native map_rows
native_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = MultiGridBarrier.map_rows(f, x_native, w_native)
    t = time_ns() - t
    push!(native_times, t)
end

# Time MPI map_rows (should use the optimized code)
mpi_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = HPCLinearAlgebra.map_rows(f, x_mpi, w_mpi)
    t = time_ns() - t
    push!(mpi_times, t)
end

println("\nNative map_rows: $(round(median(native_times)/1000, digits=1)) μs")
println("MPI map_rows:    $(round(median(mpi_times)/1000, digits=1)) μs")
println("Ratio:           $(round(median(mpi_times) / median(native_times), digits=2))x")

# Also test with row-vector returning function
f_rowvec = (row_x, w) -> (w * row_x)'

native_rv_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = MultiGridBarrier.map_rows(f_rowvec, x_native, w_native)
    t = time_ns() - t
    push!(native_rv_times, t)
end

mpi_rv_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = HPCLinearAlgebra.map_rows(f_rowvec, x_mpi, w_mpi)
    t = time_ns() - t
    push!(mpi_rv_times, t)
end

println("\nWith row-vector function:")
println("Native: $(round(median(native_rv_times)/1000, digits=1)) μs")
println("MPI:    $(round(median(mpi_rv_times)/1000, digits=1)) μs")
println("Ratio:  $(round(median(mpi_rv_times) / median(native_rv_times), digits=2))x")

println("\nDone.")
