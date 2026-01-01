using Test
using MPI

# Force precompilation of test dependencies before MPI tests
# This ensures packages are compiled in the main process, not in each MPI subprocess
try
    @eval using MultiGridBarrierMPI
    @eval using LinearAlgebraMPI
    @eval using MultiGridBarrier
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
    proc = run(ignorestatus(cmd))
    ok = success(proc)
    if ok != expect_success
        @info "MPI test exit status mismatch" test_file=test_file ok=ok expect_success=expect_success exitcode=proc.exitcode cmd=cmd
    end
    @test ok == expect_success
end

@testset "MultiGridBarrierMPI.jl" begin
    @testset "Helper functions" begin
        run_mpi_test("test_helpers.jl")
    end

    @testset "Quick integration test (1D)" begin
        run_mpi_test("test_quick.jl")
    end

    @testset "Parabolic solver" begin
        run_mpi_test("test_parabolic.jl")
    end
end
