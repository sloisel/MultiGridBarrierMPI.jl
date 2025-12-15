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

println(io0(), "[DEBUG] Testing Hessian assembly (nranks=$nranks)")

# Create geometry at L=2 (simplest case)
g = fem1d_mpi(Float64; L=2)

println(io0(), "[DEBUG] Geometry created")
println(io0(), "[DEBUG] x size: $(size(g.x))")
println(io0(), "[DEBUG] w length: $(length(g.w))")
println(io0(), "[DEBUG] Operators: $(keys(g.operators))")
println(io0(), "[DEBUG] Subspaces: $(keys(g.subspaces))")

# Get key matrices
D = g.operators[:dx]  # Derivative operator
w = g.w               # Quadrature weights

println(io0(), "[DEBUG] D size: $(size(D))")
println(io0(), "[DEBUG] D row_partition: $(D.row_partition)")
println(io0(), "[DEBUG] D col_partition: $(D.col_partition)")
println(io0(), "[DEBUG] w length: $(length(w))")
println(io0(), "[DEBUG] w partition: $(w.partition)")

# Build a simple Hessian-like matrix: D' * diag(w) * D
n = length(w)
W_diag = spdiagm(n, n, 0 => w)

println(io0(), "[DEBUG] W_diag size: $(size(W_diag))")
println(io0(), "[DEBUG] W_diag row_partition: $(W_diag.row_partition)")
println(io0(), "[DEBUG] W_diag col_partition: $(W_diag.col_partition)")

DtWD = D' * W_diag * D

println(io0(), "[DEBUG] DtWD size: $(size(DtWD))")
println(io0(), "[DEBUG] DtWD row_partition: $(DtWD.row_partition)")
println(io0(), "[DEBUG] DtWD col_partition: $(DtWD.col_partition)")

# Convert to native and examine
DtWD_native = SparseMatrixCSC(DtWD)

if rank == 0
    println("[DEBUG] DtWD nnz: $(nnz(DtWD_native))")
    println("[DEBUG] DtWD diagonal: $(diag(DtWD_native))")

    # Check eigenvalues (only rank 0 does this)
    DtWD_dense = Matrix(DtWD_native)
    eigs = eigvals(Symmetric(DtWD_dense))
    println("[DEBUG] DtWD eigenvalues: $eigs")
    println("[DEBUG] Min eigenvalue: $(minimum(eigs))")
    println("[DEBUG] Max eigenvalue: $(maximum(eigs))")
end

# Now try with the Dirichlet restriction
if haskey(g.subspaces, :dirichlet) && length(g.subspaces[:dirichlet]) > 0
    R = g.subspaces[:dirichlet][end]  # Finest level restriction

    println(io0(), "[DEBUG] R size: $(size(R))")
    println(io0(), "[DEBUG] R row_partition: $(R.row_partition)")
    println(io0(), "[DEBUG] R col_partition: $(R.col_partition)")

    # Build restricted system: R' * D' * diag(w) * D * R
    RtDtWDR = R' * DtWD * R

    println(io0(), "[DEBUG] RtDtWDR size: $(size(RtDtWDR))")
    println(io0(), "[DEBUG] RtDtWDR row_partition: $(RtDtWDR.row_partition)")
    println(io0(), "[DEBUG] RtDtWDR col_partition: $(RtDtWDR.col_partition)")

    # Convert to native and examine
    RtDtWDR_native = SparseMatrixCSC(RtDtWDR)

    if rank == 0
        println("[DEBUG] RtDtWDR nnz: $(nnz(RtDtWDR_native))")
        println("[DEBUG] RtDtWDR diagonal: $(diag(RtDtWDR_native))")

        RtDtWDR_dense = Matrix(RtDtWDR_native)
        eigs = eigvals(Symmetric(RtDtWDR_dense))
        println("[DEBUG] RtDtWDR eigenvalues: $eigs")
        println("[DEBUG] Min eigenvalue: $(minimum(eigs))")
    end

    # Try factorizing the restricted system
    println(io0(), "[DEBUG] Attempting to factorize RtDtWDR...")
    n_coarse = size(RtDtWDR, 1)
    b_test = VectorMPI(ones(n_coarse))

    try
        x_test = RtDtWDR \ b_test
        println(io0(), "[DEBUG] Factorization succeeded!")
        x_native = Vector(x_test)
        println(io0(), "[DEBUG] Solution norm: $(norm(x_native))")
    catch e
        println(io0(), "[DEBUG] Factorization FAILED: $e")
    end
end

# Also test the native version for comparison
println(io0(), "[DEBUG] \n--- Comparing with native Julia ---")
g_native = fem1d(Float64; L=2)
D_native = g_native.operators[:dx]
w_native = g_native.w
n_native = length(w_native)
W_diag_native = spdiagm(n_native, n_native, 0 => w_native)
DtWD_native2 = D_native' * W_diag_native * D_native

if rank == 0
    println("[DEBUG] Native DtWD nnz: $(nnz(DtWD_native2))")
    println("[DEBUG] Native DtWD diagonal: $(diag(DtWD_native2))")

    # Compare MPI vs native
    diff = norm(DtWD_native - DtWD_native2)
    println("[DEBUG] DtWD difference (MPI vs native): $diff")
end

if haskey(g_native.subspaces, :dirichlet) && length(g_native.subspaces[:dirichlet]) > 0
    R_native = g_native.subspaces[:dirichlet][end]
    RtDtWDR_native2 = R_native' * DtWD_native2 * R_native

    if rank == 0
        println("[DEBUG] Native RtDtWDR nnz: $(nnz(RtDtWDR_native2))")
        println("[DEBUG] Native RtDtWDR diagonal: $(diag(RtDtWDR_native2))")

        # Compare
        diff = norm(RtDtWDR_native - RtDtWDR_native2)
        println("[DEBUG] RtDtWDR difference (MPI vs native): $diff")
    end
end

println(io0(), "[DEBUG] Hessian debug test completed")
