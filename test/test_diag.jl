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

if rank == 0
    println("[DEBUG] Testing amgb_diag function")
    flush(stdout)
end

# Test 1: Create a simple diagonal matrix and verify it matches native
n = 10
d_native = collect(Float64, 1:n)
D_ref = spdiagm(0 => d_native)

d_mpi = HPCVector(d_native)
template = HPCSparseMatrix{Float64}(spzeros(Float64, n, n))
D_mpi = MultiGridBarrier.amgb_diag(template, d_mpi)

# Convert back to native
D_mpi_native = SparseMatrixCSC(D_mpi)

if rank == 0
    println("[DEBUG] Native diagonal: $(diag(D_ref))")
    println("[DEBUG] MPI diagonal: $(diag(D_mpi_native))")
    println("[DEBUG] Match: $(D_mpi_native == D_ref)")
    flush(stdout)
end

@test D_mpi_native == D_ref

# Test 2: Solve A x = b where A is diagonal
b_native = ones(n)
b_mpi = HPCVector(b_native)
x_hpc = D_mpi \ b_mpi
x_native = Vector(x_hpc)
x_ref = D_ref \ b_native

if rank == 0
    println("[DEBUG] Reference solution: $x_ref")
    println("[DEBUG] MPI solution: $x_native")
    diff = norm(x_native - x_ref)
    println("[DEBUG] Difference: $diff")
    flush(stdout)
end

@test norm(x_native - x_ref) < 1e-10

# Test 3: Check matrix-vector multiplication with diagonal
y_mpi = D_mpi * b_mpi
y_native = Vector(y_mpi)
y_ref = D_ref * b_native

if rank == 0
    println("[DEBUG] D * b reference: $y_ref")
    println("[DEBUG] D * b MPI: $y_native")
    println("[DEBUG] Difference: $(norm(y_native - y_ref))")
    flush(stdout)
end

@test norm(y_native - y_ref) < 1e-10

# Test 4: Check adjoint multiplication
Dt_mpi = D_mpi'
z_hpc = Dt_mpi * b_mpi
z_native = Vector(z_hpc)
z_ref = D_ref' * b_native

if rank == 0
    println("[DEBUG] D' * b reference: $z_ref")
    println("[DEBUG] D' * b MPI: $z_native")
    println("[DEBUG] Difference: $(norm(z_native - z_ref))")
    flush(stdout)
end

@test norm(z_native - z_ref) < 1e-10

# Test 5: Check D' * D operation (Hessian-like)
DtD_mpi = Dt_mpi * D_mpi
DtD_native = SparseMatrixCSC(DtD_mpi)
DtD_ref = D_ref' * D_ref

if rank == 0
    println("[DEBUG] D'D reference diag: $(diag(DtD_ref))")
    println("[DEBUG] D'D MPI diag: $(diag(DtD_native))")
    println("[DEBUG] Difference: $(norm(DtD_native - DtD_ref))")
    flush(stdout)
end

@test norm(DtD_native - DtD_ref) < 1e-10

if rank == 0
    println("[DEBUG] All amgb_diag tests completed successfully")
    flush(stdout)
end
