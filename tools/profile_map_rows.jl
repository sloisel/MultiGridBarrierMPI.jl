#!/usr/bin/env julia
#
# Profile map_rows overhead
#
# Run with: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=10 mpiexec -n 1 julia --project=. tools/profile_map_rows.jl
#

using MPI
MPI.Init()

using BenchmarkTools
using MultiGridBarrier
using MultiGridBarrierMPI
using HPCLinearAlgebra
using HPCLinearAlgebra: HPCVector, HPCMatrix
using LinearAlgebra

MultiGridBarrierMPI.Init()

const L = 6
println("="^70)
println("Profile map_rows: Native vs MPI at L=$L")
println("="^70)

# Create both geometries
println("\nCreating geometries...")
g_native = fem2d(Float64; L=L)
g_mpi = fem2d_mpi(Float64; L=L)

n = size(g_native.x, 1)
println("Grid size: n = $n")

# Test data
x_native = g_native.x  # n x 2 matrix
w_native = g_native.w  # n vector
x_mpi = g_mpi.x
w_mpi = g_mpi.w

# Typical barrier function operations involve map_rows
# Let's test different patterns

println("\n" * "-"^70)
println("1. Simple map_rows: sum of row")
println("-"^70)

f_sum = row -> sum(row)

# Native
t_native_sum = @benchmark MultiGridBarrier.map_rows($f_sum, $x_native) samples=100 evals=5
# MPI
t_mpi_sum = @benchmark MultiGridBarrier.map_rows($f_sum, $x_mpi) samples=100 evals=5

println("   Native: $(round(median(t_native_sum).time/1e3, digits=3)) μs")
println("   MPI:    $(round(median(t_mpi_sum).time/1e3, digits=3)) μs")
println("   Ratio:  $(round(median(t_mpi_sum).time / median(t_native_sum).time, digits=2))x")

println("\n" * "-"^70)
println("2. map_rows with log: log.(row)")
println("-"^70)

f_log = row -> log.(abs.(row) .+ 1e-10)

t_native_log = @benchmark MultiGridBarrier.map_rows($f_log, $x_native) samples=100 evals=5
t_mpi_log = @benchmark MultiGridBarrier.map_rows($f_log, $x_mpi) samples=100 evals=5

println("   Native: $(round(median(t_native_log).time/1e3, digits=3)) μs")
println("   MPI:    $(round(median(t_mpi_log).time/1e3, digits=3)) μs")
println("   Ratio:  $(round(median(t_mpi_log).time / median(t_native_log).time, digits=2))x")

println("\n" * "-"^70)
println("3. map_rows with 2 args: weighted sum")
println("-"^70)

f_weighted = (row, w) -> w * sum(row)

t_native_weighted = @benchmark MultiGridBarrier.map_rows($f_weighted, $x_native, $w_native) samples=100 evals=5
t_mpi_weighted = @benchmark MultiGridBarrier.map_rows($f_weighted, $x_mpi, $w_mpi) samples=100 evals=5

println("   Native: $(round(median(t_native_weighted).time/1e3, digits=3)) μs")
println("   MPI:    $(round(median(t_mpi_weighted).time/1e3, digits=3)) μs")
println("   Ratio:  $(round(median(t_mpi_weighted).time / median(t_native_weighted).time, digits=2))x")

println("\n" * "-"^70)
println("4. Direct broadcast (no map_rows): x .* w")
println("-"^70)

t_native_broadcast = @benchmark $x_native .* $w_native samples=100 evals=5
t_mpi_broadcast = @benchmark $x_mpi .* $w_mpi samples=100 evals=5

println("   Native: $(round(median(t_native_broadcast).time/1e3, digits=3)) μs")
println("   MPI:    $(round(median(t_mpi_broadcast).time/1e3, digits=3)) μs")
println("   Ratio:  $(round(median(t_mpi_broadcast).time / median(t_native_broadcast).time, digits=2))x")

println("\n" * "-"^70)
println("5. Reduction with map_rows vs direct: sum(w .* x[:,1])")
println("-"^70)

# Native direct
t_native_direct = @benchmark sum($w_native .* $x_native[:,1]) samples=100 evals=5
# MPI direct
t_mpi_direct = @benchmark sum($w_mpi .* $x_mpi[:,1]) samples=100 evals=5

println("   Native (direct): $(round(median(t_native_direct).time/1e3, digits=3)) μs")
println("   MPI (direct):    $(round(median(t_mpi_direct).time/1e3, digits=3)) μs")
println("   Ratio:  $(round(median(t_mpi_direct).time / median(t_native_direct).time, digits=2))x")

# With map_rows
f_reduce = (x, w) -> w * x[1]
t_native_reduce_mr = @benchmark sum(MultiGridBarrier.map_rows($f_reduce, $x_native, $w_native)) samples=100 evals=5
t_mpi_reduce_mr = @benchmark sum(MultiGridBarrier.map_rows($f_reduce, $x_mpi, $w_mpi)) samples=100 evals=5

println("   Native (map_rows): $(round(median(t_native_reduce_mr).time/1e3, digits=3)) μs")
println("   MPI (map_rows):    $(round(median(t_mpi_reduce_mr).time/1e3, digits=3)) μs")
println("   Ratio:  $(round(median(t_mpi_reduce_mr).time / median(t_native_reduce_mr).time, digits=2))x")

println("\n" * "-"^70)
println("6. Complex barrier-like computation")
println("-"^70)

# Something similar to what barrier functions do
f_barrier_like = (x, w) -> begin
    s = x[1]^2 + x[2]^2
    return w * log(max(s, 1e-10))
end

t_native_barrier = @benchmark MultiGridBarrier.map_rows($f_barrier_like, $x_native, $w_native) samples=50 evals=3
t_mpi_barrier = @benchmark MultiGridBarrier.map_rows($f_barrier_like, $x_mpi, $w_mpi) samples=50 evals=3

println("   Native: $(round(median(t_native_barrier).time/1e3, digits=3)) μs")
println("   MPI:    $(round(median(t_mpi_barrier).time/1e3, digits=3)) μs")
println("   Ratio:  $(round(median(t_mpi_barrier).time / median(t_native_barrier).time, digits=2))x")

println("\n" * "="^70)
println("Summary")
println("="^70)

results = [
    ("Simple sum", median(t_mpi_sum).time / median(t_native_sum).time),
    ("Log transform", median(t_mpi_log).time / median(t_native_log).time),
    ("Weighted sum", median(t_mpi_weighted).time / median(t_native_weighted).time),
    ("Direct broadcast", median(t_mpi_broadcast).time / median(t_native_broadcast).time),
    ("Direct reduction", median(t_mpi_direct).time / median(t_native_direct).time),
    ("map_rows reduction", median(t_mpi_reduce_mr).time / median(t_native_reduce_mr).time),
    ("Barrier-like", median(t_mpi_barrier).time / median(t_native_barrier).time),
]

for (name, ratio) in results
    status = ratio > 1.5 ? "⚠️ " : ratio > 1.1 ? "  " : "✓ "
    println("$status $(rpad(name, 20)) $(round(ratio, digits=2))x")
end

println("\nDone.")
