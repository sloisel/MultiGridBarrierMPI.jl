#!/usr/bin/env julia
# Use Julia's built-in profiler to find bottlenecks
using MPI
MPI.Init()

using Profile
using MultiGridBarrier
using HPCMultiGridBarrier
using HPCSparseArrays

HPCMultiGridBarrier.Init()

const L = 6

println("="^70)
println("Julia profiling at L=$L")
println("="^70)

# Create geometry
g_hpc = fem2d_hpc(Float64; L=L)
println("Grid points: ", sum(g_hpc.x.row_partition) - 2)

# Warmup
println("Warmup...")
MultiGridBarrier.amgb(g_hpc; verbose=false, tol=0.1)

# Profile the solve
println("Profiling MPI solve...")
Profile.clear()
@profile MultiGridBarrier.amgb(g_hpc; verbose=false, tol=0.1)

# Print profile
println("\nTop 30 by flat count:")
Profile.print(format=:flat, sortedby=:count, mincount=100, maxdepth=30)

println("\n" * "="^70)
println("Done.")
