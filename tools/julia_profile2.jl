#!/usr/bin/env julia
# Use Julia's built-in profiler with tree view
using MPI
MPI.Init()

using Profile
using MultiGridBarrier
using MultiGridBarrierMPI
using HPCSparseArrays

MultiGridBarrierMPI.Init()

const L = 6

println("="^70)
println("Julia profiling at L=$L (tree view)")
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

# Print tree view (collapsed)
println("\nTree view (mincount=200):")
Profile.print(format=:tree, mincount=200, maxdepth=25)

println("\n" * "="^70)
println("Done.")
