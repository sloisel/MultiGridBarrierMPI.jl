#!/usr/bin/env julia
#
# Profile GPU execution at L=6
#

using MPI
MPI.Init()

println("Loading packages...")
using Metal
using MultiGridBarrierMPI
using MultiGridBarrier
using HPCSparseArrays
using Profile

L = 6

println("\n" * "="^70)
println("Profiling fem2d_mpi_solve GPU at L=$L")
println("="^70)

# Warmup run
println("\nWarmup...")
fem2d_mpi_solve(Float32; L=L, backend=HPCSparseArrays.mtl, verbose=false)

# Profile
println("\nProfiling...")
Profile.clear()
@profile fem2d_mpi_solve(Float32; L=L, backend=HPCSparseArrays.mtl, verbose=false)

println("\n" * "="^70)
println("Profile results (flat, top 40)")
println("="^70)
Profile.print(format=:flat, maxdepth=100, sortedby=:count, mincount=10, noisefloor=2.0)

println("\n" * "="^70)
println("Profile results (tree)")
println("="^70)
Profile.print(maxdepth=20, mincount=50)
