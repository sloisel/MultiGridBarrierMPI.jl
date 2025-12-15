using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

# Use MultiGridBarrierMPI initializer
using MultiGridBarrierMPI
MultiGridBarrierMPI.Init()

# Now load dependencies for tests
using LinearAlgebraMPI
using LinearAlgebraMPI: VectorMPI, MatrixMPI, SparseMatrixMPI, io0
using LinearAlgebra
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Helper functions test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "Helper functions tests" begin

# Test 1: amgb_zeros with SparseMatrixMPI
if rank == 0
    println("[DEBUG] Test 1: amgb_zeros with SparseMatrixMPI")
    flush(stdout)
end

A_proto = SparseMatrixMPI{Float64}(spzeros(10, 10))
Z = MultiGridBarrierMPI.amgb_zeros(A_proto, 5, 5)
@test Z isa SparseMatrixMPI
@test size(Z) == (5, 5)

# Test 2: amgb_zeros with MatrixMPI
if rank == 0
    println("[DEBUG] Test 2: amgb_zeros with MatrixMPI dense matrix")
    flush(stdout)
end

A_proto_dense = MatrixMPI(zeros(10, 10))
Z_dense = MultiGridBarrierMPI.amgb_zeros(A_proto_dense, 4, 6)
@test Z_dense isa MatrixMPI
@test size(Z_dense) == (4, 6)

# Test 3: amgb_all_isfinite with valid VectorMPI
if rank == 0
    println("[DEBUG] Test 3: amgb_all_isfinite with finite values")
    flush(stdout)
end

v_finite = VectorMPI([1.0, 2.0, 3.0])
@test MultiGridBarrierMPI.amgb_all_isfinite(v_finite) == true

# Test 4: amgb_all_isfinite with invalid VectorMPI
if rank == 0
    println("[DEBUG] Test 4: amgb_all_isfinite with infinite values")
    flush(stdout)
end

v_inf = VectorMPI([1.0, Inf, 3.0])
@test MultiGridBarrierMPI.amgb_all_isfinite(v_inf) == false

# Test 5: amgb_diag with VectorMPI
if rank == 0
    println("[DEBUG] Test 5: amgb_diag with VectorMPI")
    flush(stdout)
end

A_proto = SparseMatrixMPI{Float64}(spzeros(10, 10))
v = VectorMPI([1.0, 2.0, 3.0])
D = MultiGridBarrierMPI.amgb_diag(A_proto, v)
@test D isa SparseMatrixMPI
@test size(D) == (3, 3)

# Test 6: amgb_diag with Vector
if rank == 0
    println("[DEBUG] Test 6: amgb_diag with Vector")
    flush(stdout)
end

A_proto = SparseMatrixMPI{Float64}(spzeros(10, 10))
v_native = [1.0, 2.0, 3.0, 4.0]
D = MultiGridBarrierMPI.amgb_diag(A_proto, v_native)
@test D isa SparseMatrixMPI
@test size(D) == (4, 4)

# Test 7: amgb_blockdiag
if rank == 0
    println("[DEBUG] Test 7: amgb_blockdiag for block diagonal construction")
    flush(stdout)
end

A = SparseMatrixMPI{Float64}(sparse(1.0 * I(2)))
B = SparseMatrixMPI{Float64}(sparse(1.0 * I(3)))
C = MultiGridBarrierMPI.amgb_blockdiag(A, B)
@test C isa SparseMatrixMPI
@test size(C) == (5, 5)

# Test 8: map_rows with MatrixMPI (scalar output)
if rank == 0
    println("[DEBUG] Test 8: map_rows with MatrixMPI (scalar output)")
    flush(stdout)
end

x = MatrixMPI([1.0 2.0; 3.0 4.0; 5.0 6.0])
result = MultiGridBarrierMPI.map_rows(row -> sum(row), x)
@test result isa VectorMPI
@test length(result) == 3
result_native = Vector(result)
@test result_native[1] == 3.0  # 1+2
@test result_native[2] == 7.0  # 3+4
@test result_native[3] == 11.0  # 5+6

# Test 9: map_rows with MatrixMPI (adjoint output)
if rank == 0
    println("[DEBUG] Test 9: map_rows with MatrixMPI (adjoint output)")
    flush(stdout)
end

x = MatrixMPI([1.0 2.0; 3.0 4.0])
result = MultiGridBarrierMPI.map_rows(row -> [sum(row), prod(row)]', x)
@test result isa MatrixMPI
@test size(result) == (2, 2)
result_native = Matrix(result)
@test result_native[1, 1] == 3.0   # sum of row 1
@test result_native[1, 2] == 2.0   # prod of row 1
@test result_native[2, 1] == 7.0   # sum of row 2
@test result_native[2, 2] == 12.0  # prod of row 2

# Test 10: map_rows with multiple inputs
if rank == 0
    println("[DEBUG] Test 10: map_rows with multiple inputs")
    flush(stdout)
end

x = MatrixMPI([1.0 2.0; 3.0 4.0])
y = VectorMPI([10.0, 20.0])
result = MultiGridBarrierMPI.map_rows((row_x, row_y) -> sum(row_x) + row_y[1], x, y)
@test result isa VectorMPI
@test length(result) == 2
result_native = Vector(result)
@test result_native[1] == 13.0  # (1+2) + 10
@test result_native[2] == 27.0  # (3+4) + 20

if rank == 0
    println("[DEBUG] All helper tests completed")
    flush(stdout)
end

end  # End of QuietTestSet

# Aggregate per-rank counts and print a single summary on root
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
    get(ts.counts, :broken, 0),
    get(ts.counts, :skip, 0),
]

global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    println("Test Summary: Helper functions tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

if rank == 0
    println("[DEBUG] Helper functions test file completed successfully")
    flush(stdout)
end
