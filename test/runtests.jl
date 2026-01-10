using Test
using MPI

println("=== runtests.jl starting ===")
flush(stdout)

# Force precompilation of test dependencies before MPI tests
# This ensures packages are compiled in the main process, not in each MPI subprocess
println("Loading packages for precompilation...")
flush(stdout)
try
    @eval using MultiGridBarrierMPI
    println("  - MultiGridBarrierMPI loaded")
    flush(stdout)
    @eval using HPCLinearAlgebra
    println("  - HPCLinearAlgebra loaded")
    flush(stdout)
    @eval using MultiGridBarrier
    println("  - MultiGridBarrier loaded")
    flush(stdout)
    @eval using SparseArrays
    @eval using LinearAlgebra
    println("Precompilation complete for test environment")
    flush(stdout)
catch err
    @warn "Precompile step hit an error; tests may still proceed" err
end

# Helper to run individual test files with MPI
function run_mpi_test(test_file::AbstractString; nprocs::Integer=4, nthreads::Integer=2, expect_success::Bool=true)
    test_path = joinpath(@__DIR__, test_file)
    # Allow overriding mpiexec via environment variable (useful for CI with system MPI)
    mpiexec_cmd = get(ENV, "MPIEXEC_PATH", nothing)
    if mpiexec_cmd === nothing
        mpiexec_cmd = MPI.mpiexec()
    else
        mpiexec_cmd = Cmd([mpiexec_cmd])
    end
    test_proj = Base.active_project()
    cmd = `$mpiexec_cmd -n $nprocs $(Base.julia_cmd()) --threads=$nthreads --project=$test_proj $test_path`
    println(">>> Running: $cmd")
    flush(stdout)
    proc = run(ignorestatus(cmd))
    ok = success(proc)
    println(">>> Finished: $test_file (success=$ok)")
    flush(stdout)
    if ok != expect_success
        @info "MPI test exit status mismatch" test_file=test_file ok=ok expect_success=expect_success exitcode=proc.exitcode cmd=cmd
    end
    @test ok == expect_success
end

println("=== Starting test suite ===")
flush(stdout)

@testset "MultiGridBarrierMPI.jl" begin
    println(">>> Test 1/4: Helper functions")
    flush(stdout)
    @testset "Helper functions" begin
        run_mpi_test("test_helpers.jl")
    end

    println(">>> Test 2/4: Quick integration test (1D)")
    flush(stdout)
    @testset "Quick integration test (1D)" begin
        run_mpi_test("test_quick.jl")
    end

    println(">>> Test 3/4: 2D integration test (with CUDA if available)")
    flush(stdout)
    @testset "2D integration test" begin
        # Use 2 ranks for CUDA tests to match typical 2-GPU setup
        # (NCCL fails when ranks exceed available GPUs)
        run_mpi_test("test_2d.jl"; nprocs=2)
    end

    println(">>> Test 4/4: Parabolic solver")
    flush(stdout)
    @testset "Parabolic solver" begin
        run_mpi_test("test_parabolic.jl")
    end
end

println("=== Test suite complete ===")
flush(stdout)
