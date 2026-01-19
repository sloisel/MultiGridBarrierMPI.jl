#!/usr/bin/env julia
#
# Profile fem2d_hpc_solve at L=5 to identify overhead
#

using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

io0(args...) = rank == 0 && println(args...)

io0("Loading packages...")
using MultiGridBarrier
using HPCMultiGridBarrier
using LinearAlgebra
using Profile
using Printf

L = 5

io0("\n" * "="^70)
io0("Profiling fem2d_hpc_solve at L=$L")
io0("="^70)

# Warmup
io0("\nWarmup...")
fem2d_hpc_solve(Float64; L=L, verbose=false)

# Profile the MPI solve
io0("\nProfiling MPI solve...")
Profile.clear()
@profile fem2d_hpc_solve(Float64; L=L, verbose=false)

if rank == 0
    io0("\n--- Profile Results ---\n")
    Profile.print(mincount=50, noisefloor=2.0)
end
