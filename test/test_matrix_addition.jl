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

# Test the specific pattern that caused the bug: accumulating D[j]'*foo*D[k] terms
# where different additions with the same input structural hashes can produce
# results with different structures (due to numerical cancellation).

# Create geometry
g = fem1d_mpi(Float64; L=2)
D_op = [g.operators[:dx], g.operators[:id]]
w_mpi = g.w
R = g.subspaces[:dirichlet][end]

# Native version for comparison
g_native = fem1d(Float64; L=2)
D_native_op = [g_native.operators[:dx], g_native.operators[:id]]
w_native = g_native.w
R_native = g_native.subspaces[:dirichlet][end]

# Create test y values (simulating output from map_rows in f2)
n = length(w_mpi)
y_vals = zeros(n, 4)
y_vals[:, 1] = ones(n) .* 0.5  # Hessian entry (1,1)
y_vals[:, 2] = ones(n) .* 0.1  # Hessian entry (1,2)
y_vals[:, 3] = ones(n) .* 0.1  # Hessian entry (2,1)
y_vals[:, 4] = ones(n) .* 0.3  # Hessian entry (2,2)

y_mpi = MatrixMPI(y_vals)

# Build Hessian the same way f2 does (MPI version)
# j=1: dx' * diag(w .* y[:,1]) * dx
foo = MultiGridBarrier.amgb_diag(D_op[1], w_mpi .* y_mpi[:, 1])
ret_mpi = D_op[1]' * foo * D_op[1]

# j=2: id' * diag(w .* y[:,4]) * id
foo = MultiGridBarrier.amgb_diag(D_op[1], w_mpi .* y_mpi[:, 4])
bar = D_op[2]' * foo * D_op[2]
ret_mpi = ret_mpi + bar

# Cross terms: (id' * foo * dx) + (dx' * foo * id)
foo = MultiGridBarrier.amgb_diag(D_op[1], w_mpi .* y_mpi[:, 3])
term1 = D_op[2]' * foo * D_op[1]  # id' * foo * dx
term2 = D_op[1]' * foo * D_op[2]  # dx' * foo * id
cross_term = term1 + term2  # This was failing before the fix!
ret_mpi = ret_mpi + cross_term

# Final restriction
H_mpi = R' * ret_mpi * R
H_mpi_native = SparseMatrixCSC(H_mpi)

# Build native version for comparison
foo_native = spdiagm(n, n, 0 => w_native .* y_vals[:, 1])
ret_native = D_native_op[1]' * foo_native * D_native_op[1]

foo_native = spdiagm(n, n, 0 => w_native .* y_vals[:, 4])
bar_native = D_native_op[2]' * foo_native * D_native_op[2]
ret_native = ret_native + bar_native

foo_native = spdiagm(n, n, 0 => w_native .* y_vals[:, 3])
cross_term_native = D_native_op[2]' * foo_native * D_native_op[1] + D_native_op[1]' * foo_native * D_native_op[2]
ret_native = ret_native + cross_term_native

H_native = R_native' * ret_native * R_native

# Run tests
@testset "Matrix addition with numerical cancellation" begin
    # Check nnz matches
    @test nnz(H_mpi_native) == nnz(H_native)

    # Check matrix values match
    @test norm(H_mpi_native - H_native) ≈ 0.0 atol=1e-12

    # Check eigenvalues match
    eigs_mpi = eigvals(Symmetric(Matrix(H_mpi_native)))
    eigs_native = eigvals(Symmetric(Matrix(H_native)))
    @test eigs_mpi ≈ eigs_native atol=1e-12
end

println(io0(), "Test completed successfully")
