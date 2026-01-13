#!/usr/bin/env julia
#
# Profile the actual solver to find where MPI is slower
#
# Run with: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=10 mpiexec -n 1 julia --project=. tools/profile_solver.jl
#

using MPI
MPI.Init()

using Profile
using MultiGridBarrier
using MultiGridBarrierMPI
using HPCLinearAlgebra

MultiGridBarrierMPI.Init()

const L = 5  # Use L=5 for faster profiling

println("="^70)
println("Profiling fem2d_solve vs fem2d_mpi_solve at L=$L")
println("="^70)

# Warmup
println("\nWarming up native...")
fem2d_solve(Float64; L=L, verbose=false)
println("Warming up MPI...")
fem2d_mpi_solve(Float64; L=L, verbose=false)

# Profile native
println("\n" * "-"^70)
println("Profiling NATIVE fem2d_solve...")
println("-"^70)
Profile.clear()
@profile fem2d_solve(Float64; L=L, verbose=false)
println("\nTop functions (native):")
Profile.print(maxdepth=30, mincount=50, sortedby=:count)

# Profile MPI
println("\n" * "-"^70)
println("Profiling MPI fem2d_mpi_solve...")
println("-"^70)
Profile.clear()
@profile fem2d_mpi_solve(Float64; L=L, verbose=false)
println("\nTop functions (MPI):")
Profile.print(maxdepth=30, mincount=50, sortedby=:count)

println("\n" * "-"^70)
println("Flat profile (MPI) - Functions by total time:")
println("-"^70)
Profile.print(format=:flat, mincount=50, sortedby=:count)

println("\nDone.")
