using Test

# Load test utilities BEFORE MPI (for Metal/CUDA detection)
include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils

using MPI

# Initialize MPI
if !MPI.Initialized()
    MPI.Init()
end

# Set up GPU device assignment for CUDA (must happen before NCCL initialization)
if TestUtils.CUDA_AVAILABLE
    using CUDA
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    num_gpus = length(CUDA.devices())
    gpu_id = mod(rank, num_gpus)
    CUDA.device!(gpu_id)
    if rank == 0
        println("[DEBUG] CUDA setup: $num_gpus GPUs available, rank=$rank using GPU $gpu_id")
        flush(stdout)
    end
end

# Use MultiGridBarrierMPI initializer
using MultiGridBarrierMPI

# Now load dependencies for tests
using HPCLinearAlgebra
using HPCLinearAlgebra: HPCVector, HPCMatrix, HPCSparseMatrix, io0
using LinearAlgebra
using SparseArrays
using MultiGridBarrier
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] 2D integration test starting")
    flush(stdout)
end

# Get 2D configs lazily (after MPI is initialized)
configs_2d = TestUtils.get_all_configs_2d()

if rank == 0
    println("[DEBUG] Configs to test: ", length(configs_2d))
    if TestUtils.CUDA_AVAILABLE
        println("[DEBUG] CUDA is available - testing CUDA64")
    end
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "2D integration tests" begin

# Use get_all_configs_2d() which includes CUDA (2D matrices are not tridiagonal)
for (T, backend, backend_name) in configs_2d
    TOL = TestUtils.tolerance(T)

    if rank == 0
        println("[DEBUG] Testing with $T, $backend_name")
        flush(stdout)
    end

    # Test 1: Create 2D geometry with MPI types
    if rank == 0
        println("[DEBUG] Test 1: fem2d_mpi geometry creation ($backend_name)")
        flush(stdout)
    end

    g = fem2d_mpi(T; L=2, backend=backend)
    @test g isa MultiGridBarrier.Geometry
    @test g.x isa HPCMatrix
    @test g.w isa HPCVector
    @test g.operators[:id] isa HPCSparseMatrix

    if rank == 0
        println("[DEBUG] Geometry created successfully")
        println("[DEBUG]   x size: $(size(g.x))")
        println("[DEBUG]   w length: $(length(g.w))")
        flush(stdout)
    end

    # Test 2: Convert back to native and verify
    if rank == 0
        println("[DEBUG] Test 2: mpi_to_native conversion ($backend_name)")
        flush(stdout)
    end

    g_native = mpi_to_native(g)
    @test g_native.x isa Matrix
    @test g_native.w isa Vector
    @test g_native.operators[:id] isa SparseMatrixCSC

    if rank == 0
        println("[DEBUG] Conversion to native successful")
        flush(stdout)
    end

    # Test 3: Verify native geometry matches original fem2d
    if rank == 0
        println("[DEBUG] Test 3: Verify conversion preserves data ($backend_name)")
        flush(stdout)
    end

    g_ref = fem2d(T; L=2)
    # Use approximate comparison for Float32
    @test isapprox(g_native.x, g_ref.x; atol=TOL)
    @test isapprox(g_native.w, g_ref.w; atol=TOL)

    if rank == 0
        println("[DEBUG] Data preservation verified")
        flush(stdout)
    end

    # Test 4: Solve a simple 2D problem
    if rank == 0
        println("[DEBUG] Test 4: fem2d_mpi geometry + amgb solve ($backend_name)")
        flush(stdout)
    end

    # Use small L to keep test fast, solve using the geometry we created
    sol = amgb(g; p=T(2), verbose=false, tol=TOL, logfile=io0())
    @test sol isa MultiGridBarrier.AMGBSOL

    if rank == 0
        println("[DEBUG] Solution computed successfully")
        flush(stdout)
    end

    # Test 5: Convert solution to native
    if rank == 0
        println("[DEBUG] Test 5: Convert solution to native ($backend_name)")
        flush(stdout)
    end

    sol_native = mpi_to_native(sol)
    @test sol_native.z isa Matrix
    @test sol_native.geometry.x isa Matrix

    if rank == 0
        println("[DEBUG] Solution conversion successful")
        flush(stdout)
    end

    # Test 6: Compare MPI solution with native solution
    if rank == 0
        println("[DEBUG] Test 6: Compare MPI vs native results ($backend_name)")
        flush(stdout)
    end

    sol_ref = fem2d_solve(T; L=2, p=T(2), verbose=false, tol=TOL)
    z_diff = norm(sol_native.z - sol_ref.z)
    # Use looser tolerance for comparison (solver convergence + Float32 precision)
    @test z_diff < TOL * 1000

    if rank == 0
        println("[DEBUG] MPI and native solutions match (diff = $z_diff)")
        flush(stdout)
    end

    if rank == 0
        println("[DEBUG] Completed tests for $T, $backend_name")
        flush(stdout)
    end

end  # for loop over configs

if rank == 0
    println("[DEBUG] All 2D integration tests completed")
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
    println("Test Summary: 2D integration tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

if rank == 0
    println("[DEBUG] 2D integration test file completed successfully")
    flush(stdout)
end
