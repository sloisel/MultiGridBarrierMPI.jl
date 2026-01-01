using Test

# Load test utilities BEFORE MPI (for Metal detection)
include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

using MPI

# Initialize MPI
if !MPI.Initialized()
    MPI.Init()
end

# Use MultiGridBarrierMPI initializer
using MultiGridBarrierMPI

# Now load dependencies for tests
using LinearAlgebraMPI
using LinearAlgebraMPI: VectorMPI, MatrixMPI, SparseMatrixMPI, io0
using LinearAlgebra
using SparseArrays
using StaticArrays: SVector
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Helper functions test starting")
    println("[DEBUG] Configs to test: ", length(TestUtils.ALL_CONFIGS))
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "Helper functions tests" begin

for (T, backend, backend_name) in TestUtils.ALL_CONFIGS
    TOL = TestUtils.tolerance(T)

    if rank == 0
        println("[DEBUG] Testing with $T, $backend_name")
        flush(stdout)
    end

    # Test 1: amgb_zeros with SparseMatrixMPI
    if rank == 0
        println("[DEBUG] Test 1: amgb_zeros with SparseMatrixMPI ($backend_name)")
        flush(stdout)
    end

    A_proto = backend(SparseMatrixMPI{T}(spzeros(T, 10, 10)))
    Z = MultiGridBarrierMPI.amgb_zeros(A_proto, 5, 5)
    @test Z isa SparseMatrixMPI
    @test size(Z) == (5, 5)

    # Test 2: amgb_zeros with MatrixMPI
    if rank == 0
        println("[DEBUG] Test 2: amgb_zeros with MatrixMPI dense matrix ($backend_name)")
        flush(stdout)
    end

    A_proto_dense = backend(MatrixMPI(zeros(T, 10, 10)))
    Z_dense = MultiGridBarrierMPI.amgb_zeros(A_proto_dense, 4, 6)
    @test Z_dense isa MatrixMPI
    @test size(Z_dense) == (4, 6)

    # Test 3: amgb_all_isfinite with valid VectorMPI
    if rank == 0
        println("[DEBUG] Test 3: amgb_all_isfinite with finite values ($backend_name)")
        flush(stdout)
    end

    v_finite = backend(VectorMPI(T[1, 2, 3]))
    @test MultiGridBarrierMPI.amgb_all_isfinite(v_finite) == true

    # Test 4: amgb_all_isfinite with invalid VectorMPI
    if rank == 0
        println("[DEBUG] Test 4: amgb_all_isfinite with infinite values ($backend_name)")
        flush(stdout)
    end

    v_inf = backend(VectorMPI(T[1, Inf, 3]))
    @test MultiGridBarrierMPI.amgb_all_isfinite(v_inf) == false

    # Test 5: amgb_diag with VectorMPI
    if rank == 0
        println("[DEBUG] Test 5: amgb_diag with VectorMPI ($backend_name)")
        flush(stdout)
    end

    A_proto = backend(SparseMatrixMPI{T}(spzeros(T, 10, 10)))
    v = backend(VectorMPI(T[1, 2, 3]))
    D = MultiGridBarrierMPI.amgb_diag(A_proto, v)
    @test D isa SparseMatrixMPI
    @test size(D) == (3, 3)

    # Test 6: amgb_diag with Vector (native, not MPI)
    if rank == 0
        println("[DEBUG] Test 6: amgb_diag with Vector ($backend_name)")
        flush(stdout)
    end

    A_proto = backend(SparseMatrixMPI{T}(spzeros(T, 10, 10)))
    v_native = T[1, 2, 3, 4]
    D = MultiGridBarrierMPI.amgb_diag(A_proto, v_native)
    @test D isa SparseMatrixMPI
    @test size(D) == (4, 4)

    # Test 7: amgb_blockdiag
    if rank == 0
        println("[DEBUG] Test 7: amgb_blockdiag for block diagonal construction ($backend_name)")
        flush(stdout)
    end

    A = backend(SparseMatrixMPI{T}(sparse(T(1) * I(2))))
    B = backend(SparseMatrixMPI{T}(sparse(T(1) * I(3))))
    C = MultiGridBarrierMPI.amgb_blockdiag(A, B)
    @test C isa SparseMatrixMPI
    @test size(C) == (5, 5)

    # Test 8: map_rows with MatrixMPI (scalar output)
    if rank == 0
        println("[DEBUG] Test 8: map_rows with MatrixMPI (scalar output) ($backend_name)")
        flush(stdout)
    end

    x = backend(MatrixMPI(T[1 2; 3 4; 5 6]))
    result = MultiGridBarrierMPI.map_rows(row -> sum(row), x)
    @test result isa VectorMPI
    @test length(result) == 3
    result_native = Vector(TestUtils.to_cpu(result))
    @test result_native[1] == T(3)   # 1+2
    @test result_native[2] == T(7)   # 3+4
    @test result_native[3] == T(11)  # 5+6

    # Test 9: map_rows with MatrixMPI (SVector output)
    if rank == 0
        println("[DEBUG] Test 9: map_rows with MatrixMPI (SVector output) ($backend_name)")
        flush(stdout)
    end

    x = backend(MatrixMPI(T[1 2; 3 4]))
    result = MultiGridBarrierMPI.map_rows(row -> SVector(sum(row), prod(row)), x)
    @test result isa MatrixMPI
    @test size(result) == (2, 2)
    result_native = Matrix(TestUtils.to_cpu(result))
    @test result_native[1, 1] == T(3)   # sum of row 1
    @test result_native[1, 2] == T(2)   # prod of row 1
    @test result_native[2, 1] == T(7)   # sum of row 2
    @test result_native[2, 2] == T(12)  # prod of row 2

    # Test 10: map_rows with multiple inputs
    if rank == 0
        println("[DEBUG] Test 10: map_rows with multiple inputs ($backend_name)")
        flush(stdout)
    end

    x = backend(MatrixMPI(T[1 2; 3 4]))
    y = backend(VectorMPI(T[10, 20]))
    result = MultiGridBarrierMPI.map_rows((row_x, row_y) -> sum(row_x) + row_y[1], x, y)
    @test result isa VectorMPI
    @test length(result) == 2
    result_native = Vector(TestUtils.to_cpu(result))
    @test result_native[1] == T(13)  # (1+2) + 10
    @test result_native[2] == T(27)  # (3+4) + 20

    if rank == 0
        println("[DEBUG] Completed tests for $T, $backend_name")
        flush(stdout)
    end

end  # for loop over configs

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
