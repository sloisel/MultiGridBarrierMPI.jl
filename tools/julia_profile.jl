#!/usr/bin/env julia
# Use Julia's built-in profiler to find bottlenecks
using MPI
MPI.Init()

using Profile
using MultiGridBarrier
using MultiGridBarrierMPI
using HPCLinearAlgebra

MultiGridBarrierMPI.Init()

const L = 6

println("="^70)
println("Julia profiling at L=$L")
println("="^70)

# Create geometry
g_mpi = fem2d_mpi(Float64; L=L)
println("Grid points: ", sum(g_mpi.x.row_partition) - 2)

# Warmup
println("Warmup...")
MultiGridBarrier.amgb(g_mpi; verbose=false, tol=0.1)

# Profile the solve
println("Profiling MPI solve...")
Profile.clear()
@profile MultiGridBarrier.amgb(g_mpi; verbose=false, tol=0.1)

# Print profile
println("\nTop 30 by flat count:")
Profile.print(format=:flat, sortedby=:count, mincount=100, maxdepth=30)

println("\n" * "="^70)
println("Done.")
