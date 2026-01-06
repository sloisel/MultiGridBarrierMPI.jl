#!/usr/bin/env julia
#
# Profile hash computation overhead
#

using MPI
MPI.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
io0(args...) = rank == 0 && println(args...)

io0("Loading packages...")
using MultiGridBarrierMPI
using LinearAlgebraMPI
using LinearAlgebra
using SparseArrays

L = 5

io0("\n" * "="^70)
io0("Hash computation overhead at L=$L")
io0("="^70)

g = fem2d_mpi(Float64; L=L)
n = size(g.x, 1)
io0("  n = $n")

Dx = g.operators[:dx]
Dy = g.operators[:dy]

# First operation - includes hash computation
io0("\nFirst Dx' * Dy (includes hash computation):")
t1 = @elapsed R1 = Dx' * Dy
io0("  Time: $(round(t1*1000, digits=2)) ms")

# Second operation - hash should be cached
io0("Second Dx' * Dy (hash cached):")
t2 = @elapsed R2 = Dx' * Dy
io0("  Time: $(round(t2*1000, digits=2)) ms")

# Now create a NEW sparse matrix (simulates what happens in barrier iteration)
io0("\nCreate new diagonal matrix (like barrier does):")
d = VectorMPI(randn(n))
diag_mat = spdiagm(n, n, 0 => d)  # This creates a NEW SparseMatrixMPI
io0("  Type: $(typeof(diag_mat))")

io0("First Dx' * diag_mat (new matrix, needs hash):")
t3 = @elapsed R3 = Dx' * diag_mat
io0("  Time: $(round(t3*1000, digits=2)) ms")

io0("Second Dx' * diag_mat (hash should be cached now):")
t4 = @elapsed R4 = Dx' * diag_mat
io0("  Time: $(round(t4*1000, digits=2)) ms")

# Create another new diagonal and multiply
io0("\nCreate ANOTHER new diagonal matrix:")
d2 = VectorMPI(randn(n))
diag_mat2 = spdiagm(n, n, 0 => d2)
io0("First Dx' * diag_mat2 (new matrix):")
t5 = @elapsed R5 = Dx' * diag_mat2
io0("  Time: $(round(t5*1000, digits=2)) ms")

io0("\n--- Summary ---")
io0("Hash overhead per new matrix: ~$(round((t3-t4)*1000, digits=2)) ms")
io0("If barrier creates 10 new matrices per iteration with 20 iterations,")
io0("that's 200 * $(round((t3-t4)*1000, digits=2)) ms = $(round(200*(t3-t4)*1000, digits=0)) ms total hash overhead")

io0("\n" * "="^70)
