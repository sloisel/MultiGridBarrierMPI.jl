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

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Testing basic matrix operations")
    flush(stdout)
end

# Test: Two HPCSparseMatrix multiplication
A_native = sparse([1.0 0.0; 2.0 3.0; 0.0 4.0])  # 3x2
B_native = sparse([1.0 2.0 3.0; 4.0 5.0 6.0])   # 2x3
C_expected = A_native * B_native                  # 3x3

A_mpi = HPCSparseMatrix{Float64}(A_native)
B_mpi = HPCSparseMatrix{Float64}(B_native)

if rank == 0
    println("[DEBUG] A size: $(size(A_mpi)), B size: $(size(B_mpi))")
    flush(stdout)
end

C_mpi = A_mpi * B_mpi
C_native = SparseMatrixCSC(C_mpi)

if rank == 0
    println("[DEBUG] A*B expected:")
    println(Matrix(C_expected))
    println("[DEBUG] A*B got:")
    println(Matrix(C_native))
    println("[DEBUG] Match: $(C_native ≈ C_expected)")
    flush(stdout)
end

@test C_native ≈ C_expected

# Test: A' * A
AtA_expected = A_native' * A_native
AtA_mpi = A_mpi' * A_mpi
AtA_native = SparseMatrixCSC(AtA_mpi)

if rank == 0
    println("[DEBUG] A'A expected:")
    println(Matrix(AtA_expected))
    println("[DEBUG] A'A got:")
    println(Matrix(AtA_native))
    println("[DEBUG] Match: $(AtA_native ≈ AtA_expected)")
    flush(stdout)
end

@test AtA_native ≈ AtA_expected

# Test: Linear solve with A'A (which should be positive semi-definite)
# Add regularization to make it positive definite
n = size(AtA_expected, 1)
reg = 0.01 * sparse(I, n, n)
AtA_reg = AtA_expected + reg
AtA_reg_hpc = HPCSparseMatrix{Float64}(AtA_reg)

b = ones(n)
b_mpi = HPCVector(b)

if rank == 0
    println("[DEBUG] Attempting to solve (A'A + reg) \\ b...")
    flush(stdout)
end

try
    x_hpc = AtA_reg_hpc \ b_mpi
    x_native = Vector(x_hpc)
    x_expected = AtA_reg \ b

    if rank == 0
        println("[DEBUG] Solve succeeded!")
        println("[DEBUG] Expected: $x_expected")
        println("[DEBUG] Got: $x_native")
        println("[DEBUG] Difference: $(norm(x_native - x_expected))")
        flush(stdout)
    end

    @test norm(x_native - x_expected) < 1e-10
catch e
    if rank == 0
        println("[DEBUG] Solve failed: $e")
        println(stacktrace(catch_backtrace()))
        flush(stdout)
    end
    @test false
end

if rank == 0
    println("[DEBUG] All basic ops tests completed")
    flush(stdout)
end
