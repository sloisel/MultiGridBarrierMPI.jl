using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

using HPCMultiGridBarrier
HPCMultiGridBarrier.Init()

using HPCSparseArrays
using HPCSparseArrays: HPCVector, HPCMatrix, HPCSparseMatrix, io0
using LinearAlgebra
using SparseArrays
using MultiGridBarrier

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println(io0(), "[DEBUG] Testing non-square matrix operations")

# Create geometry
g = fem1d_hpc(Float64; L=3)

# Get restriction matrix (not square)
R = g.subspaces[:dirichlet][3]  # Finest level, 16x7

println(io0(), "[DEBUG] R size: $(size(R))")
println(io0(), "[DEBUG] R row_partition: $(R.row_partition)")
println(io0(), "[DEBUG] R col_partition: $(R.col_partition)")

# Create a vector of size 7 (ncols of R)
z_native = collect(Float64, 1:7)
z_hpc = HPCVector(z_native)

println(io0(), "[DEBUG] z size: $(length(z_hpc))")
println(io0(), "[DEBUG] z partition: $(z_hpc.partition)")

# Test R * z
println(io0(), "[DEBUG] Computing R * z...")
Rz_hpc = R * z_hpc
Rz_native = Vector(Rz_hpc)

R_native = SparseMatrixCSC(R)
Rz_expected = R_native * z_native

println(io0(), "[DEBUG] R*z size: $(length(Rz_hpc))")
println(io0(), "[DEBUG] R*z partition: $(Rz_hpc.partition)")
println(io0(), "[DEBUG] R*z expected: $Rz_expected")
println(io0(), "[DEBUG] R*z got: $Rz_native")
println(io0(), "[DEBUG] R*z match: $(Rz_native ≈ Rz_expected)")

@test Rz_native ≈ Rz_expected

# Test R' * v where v is size 16
v_native = ones(16)
v_mpi = HPCVector(v_native)

println(io0(), "[DEBUG] Computing R' * v...")
Rtv_mpi = R' * v_mpi
Rtv_native = Vector(Rtv_mpi)

Rtv_expected = R_native' * v_native

println(io0(), "[DEBUG] R'*v size: $(length(Rtv_mpi))")
println(io0(), "[DEBUG] R'*v expected: $Rtv_expected")
println(io0(), "[DEBUG] R'*v got: $Rtv_native")
println(io0(), "[DEBUG] R'*v match: $(Rtv_native ≈ Rtv_expected)")

@test Rtv_native ≈ Rtv_expected

# Test R' * A * R where A is 16x16
D = g.operators[:dx]
A = D' * D  # 16x16

println(io0(), "[DEBUG] A size: $(size(A))")
println(io0(), "[DEBUG] Computing R' * A * R...")
AR_mpi = A * R
println(io0(), "[DEBUG] A*R size: $(size(AR_mpi))")

RtAR_mpi = R' * AR_mpi
RtAR_native = SparseMatrixCSC(RtAR_mpi)

A_native = SparseMatrixCSC(A)
RtAR_expected = R_native' * A_native * R_native

println(io0(), "[DEBUG] R'*A*R size: $(size(RtAR_mpi))")
println(io0(), "[DEBUG] R'*A*R expected nnz: $(nnz(RtAR_expected))")
println(io0(), "[DEBUG] R'*A*R got nnz: $(nnz(RtAR_native))")
println(io0(), "[DEBUG] R'*A*R match: $(norm(RtAR_native - RtAR_expected) < 1e-10)")

@test norm(RtAR_native - RtAR_expected) < 1e-10

println(io0(), "[DEBUG] All non-square tests completed")
