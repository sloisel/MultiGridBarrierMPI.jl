#!/usr/bin/env julia
#
# Instrument the actual multiplication to find the slowdown
#
# Run with: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=10 mpiexec -n 1 julia --project=. tools/profile_fem2d.jl
#

using MPI
MPI.Init()

println("Loading packages...")
using MultiGridBarrier
using MultiGridBarrierMPI
using LinearAlgebraMPI
using LinearAlgebraMPI: VectorMPI, MatrixMPI, SparseMatrixMPI, _ensure_hash
using LinearAlgebra
using SparseArrays

MultiGridBarrierMPI.Init()

println("\n" * "="^70)
println("Instrumenting *(::TransposedSparseMatrixMPI, ::SparseMatrixMPI)")
println("="^70)

g_mpi = fem2d_mpi(Float64; L=6)
A = g_mpi.operators[:dx]

# Pre-warm everything
At_cached = SparseMatrixMPI(transpose(A))
_ensure_hash(A)
_ensure_hash(At_cached)
_ = At_cached * A  # Warm up MatrixPlan cache
_ = At_cached * A  # Second call to ensure cached

println("\n1. Baseline: At_cached * A (everything warmed up)")
t = time(); r = At_cached * A; t = time() - t
println("   Time: $(round(t*1000, digits=2)) ms")

println("\n2. Now instrument the Transpose path manually:")

# Manually do what *(::TransposedSparseMatrixMPI, ::SparseMatrixMPI) does
At_wrapper = transpose(A)  # Creates lazy Transpose wrapper
println("   Step a: Create Transpose wrapper")
t_a = time()
At_wrapper2 = transpose(A)
t_a = time() - t_a
println("   Time: $(round(t_a*1000000, digits=2)) μs")

println("   Step b: Extract parent")
t_b = time()
A_parent = At_wrapper.parent
t_b = time() - t_b
println("   Time: $(round(t_b*1000000, digits=2)) μs")
println("   A_parent === A: ", A_parent === A)

println("   Step c: Get cached transpose via SparseMatrixMPI(transpose(A_parent))")
t_c = time()
A_transposed = SparseMatrixMPI(transpose(A_parent))
t_c = time() - t_c
println("   Time: $(round(t_c*1000000, digits=2)) μs")
println("   A_transposed === At_cached: ", A_transposed === At_cached)

println("   Step d: Multiply A_transposed * A")
t_d = time()
result = A_transposed * A
t_d = time() - t_d
println("   Time: $(round(t_d*1000, digits=2)) ms")

println("   Total manual steps: $(round((t_a+t_b+t_c+t_d)*1000, digits=2)) ms")

println("\n3. Compare: Call A' * A directly")
t3 = time()
r3 = A' * A
t3 = time() - t3
println("   Time: $(round(t3*1000, digits=2)) ms")

println("\n4. And again:")
t4 = time()
r4 = A' * A
t4 = time() - t4
println("   Time: $(round(t4*1000, digits=2)) ms")

println("\n5. Check if A' creates a new wrapper each time:")
w1 = A'
w2 = A'
println("   w1 === w2: ", w1 === w2)
println("   w1.parent === w2.parent === A: ", w1.parent === A && w2.parent === A)

println("\n6. Time just creating A':")
t6 = time()
for i in 1:1000
    _ = A'
end
t6 = (time() - t6) / 1000
println("   Time per A': $(round(t6*1000000, digits=2)) μs")

println("\nDone.")
