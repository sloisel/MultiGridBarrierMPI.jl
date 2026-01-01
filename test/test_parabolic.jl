using Test
using MPI

# Initialize MPI first
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
using MultiGridBarrier
using MultiGridBarrier: parabolic_solve, ParabolicSOL
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Parabolic solver test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "Parabolic solver tests" begin

# Test 1: Create 1D MPI geometry and run parabolic_solve
if rank == 0
    println("[DEBUG] Test 1: parabolic_solve with 1D MPI geometry")
    flush(stdout)
end

if rank == 0; println("[DEBUG] Creating geometry..."); flush(stdout); end
g = fem1d_mpi(Float64; L=2)
if rank == 0; println("[DEBUG] Geometry created"); flush(stdout); end
@test g isa MultiGridBarrier.Geometry
@test g.x isa MatrixMPI

# Run parabolic solve with small time steps
if rank == 0; println("[DEBUG] Starting parabolic_solve..."); flush(stdout); end
sol = parabolic_solve(g; h=0.5, t1=1.0, p=2.0, verbose=false, logfile=io0())
if rank == 0; println("[DEBUG] parabolic_solve complete"); flush(stdout); end
@test sol isa ParabolicSOL
@test sol.geometry === g
@test length(sol.ts) >= 2  # At least initial and final time
@test length(sol.u) == length(sol.ts)

if rank == 0
    println("[DEBUG] Parabolic solve completed, $(length(sol.ts)) time steps")
    flush(stdout)
end

# Test 2: Verify solution snapshots are MPI types
if rank == 0
    println("[DEBUG] Test 2: Verify solution snapshots are MPI types")
    flush(stdout)
end

@test sol.u[1] isa MatrixMPI
@test sol.u[end] isa MatrixMPI

# Test 3: Convert ParabolicSOL to native
if rank == 0
    println("[DEBUG] Test 3: mpi_to_native(ParabolicSOL)")
    flush(stdout)
end

sol_native = mpi_to_native(sol)
@test sol_native isa ParabolicSOL
@test sol_native.geometry.x isa Matrix
@test sol_native.u[1] isa Matrix
@test sol_native.u[end] isa Matrix
@test sol_native.ts == sol.ts  # ts should be unchanged

if rank == 0
    println("[DEBUG] ParabolicSOL conversion successful")
    flush(stdout)
end

# Test 4: Compare MPI parabolic solution with native
if rank == 0
    println("[DEBUG] Test 4: Compare MPI vs native parabolic results")
    flush(stdout)
end

g_ref = MultiGridBarrier.fem1d(Float64; L=2)
sol_ref = parabolic_solve(g_ref; h=0.5, t1=1.0, p=2.0, verbose=false)

# Check that converted solution matches native solution
@test length(sol_native.ts) == length(sol_ref.ts)
@test sol_native.ts == sol_ref.ts

# Compare each time snapshot
for k in 1:length(sol_native.ts)
    u_diff = norm(sol_native.u[k] - sol_ref.u[k])
    @test u_diff < 1e-10
end

if rank == 0
    println("[DEBUG] MPI and native parabolic solutions match")
    flush(stdout)
end

if rank == 0
    println("[DEBUG] All parabolic tests completed")
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
    println("Test Summary: Parabolic solver tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

if rank == 0
    println("[DEBUG] Parabolic test file completed successfully")
    flush(stdout)
end
