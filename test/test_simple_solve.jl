using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

using MultiGridBarrierMPI
MultiGridBarrierMPI.Init()

using LinearAlgebraMPI
using LinearAlgebraMPI: VectorMPI, MatrixMPI, SparseMatrixMPI, io0
using LinearAlgebra
using SparseArrays
using MultiGridBarrier

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println(io0(), "[DEBUG] Testing with smaller problem (L=2)")

# Try L=2 first (simpler problem)
println(io0(), "[DEBUG] Trying L=2...")

try
    sol2 = fem1d_mpi_solve(Float64; L=2, p=1.0, verbose=false)
    println(io0(), "[DEBUG] L=2 succeeded!")

    # Convert to native
    sol2_native = mpi_to_native(sol2)
    println(io0(), "[DEBUG] Solution z norm: $(norm(sol2_native.z))")

    # Compare with native solve
    sol2_ref = fem1d_solve(Float64; L=2, p=1.0, verbose=false)
    z_diff = norm(sol2_native.z - sol2_ref.z)
    println(io0(), "[DEBUG] Difference from native: $z_diff")

catch e
    println(io0(), "[DEBUG] L=2 FAILED: $e")
    if rank == 0
        for (ex, bt) in current_exceptions()
            showerror(stdout, ex, bt)
            println()
        end
    end
end

println(io0(), "[DEBUG] Test completed")
