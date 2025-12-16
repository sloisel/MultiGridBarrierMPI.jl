#!/usr/bin/env julia
#
# Benchmark: MultiGridBarrier.jl vs MultiGridBarrierMPI vs MultiGridBarrierPETSc
#
# Compares fem3d_solve performance for L=1,2,3
#
# Run with: mpiexec -n 1 julia --project=. tools/benchmark_fem3d.jl
#

using MPI
MPI.Init()

# Only rank 0 prints output
rank = MPI.Comm_rank(MPI.COMM_WORLD)
io0(args...) = rank == 0 && println(args...)

io0("Loading packages...")
using BenchmarkTools
using MultiGridBarrier
using MultiGridBarrierMPI
using MultiGridBarrierPETSc
using Statistics

# Initialize distributed packages
MultiGridBarrierMPI.Init()
MultiGridBarrierPETSc.Init()

# Benchmark configuration
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 300

io0("\n" * "="^80)
io0("Benchmark: fem3d_solve - Native vs MPI vs PETSc")
io0("="^80)

# Store results
results = Dict{Int, NamedTuple}()

for L in 1:3
    io0("\n--- L = $L ---")

    # Warmup run for Native (first call has compilation overhead)
    io0("  Warmup Native...")
    fem3d_solve(Float64; L=L, verbose=false)

    # Warmup run for MPI
    io0("  Warmup MPI...")
    fem3d_mpi_solve(Float64; L=L, verbose=false)

    # Warmup run for PETSc
    io0("  Warmup PETSc...")
    fem3d_petsc_solve(Float64; L=L, verbose=false)

    # Benchmark Native version
    io0("  Benchmarking Native...")
    native_times = Float64[]
    for i in 1:5
        t = @elapsed fem3d_solve(Float64; L=L, verbose=false)
        push!(native_times, t)
    end
    native_median = median(native_times)
    native_min = minimum(native_times)
    native_std = std(native_times)

    # Benchmark MPI version
    io0("  Benchmarking MPI...")
    mpi_times = Float64[]
    for i in 1:5
        t = @elapsed fem3d_mpi_solve(Float64; L=L, verbose=false)
        push!(mpi_times, t)
    end
    mpi_median = median(mpi_times)
    mpi_min = minimum(mpi_times)
    mpi_std = std(mpi_times)

    # Benchmark PETSc version
    io0("  Benchmarking PETSc...")
    petsc_times = Float64[]
    for i in 1:5
        t = @elapsed fem3d_petsc_solve(Float64; L=L, verbose=false)
        push!(petsc_times, t)
    end
    petsc_median = median(petsc_times)
    petsc_min = minimum(petsc_times)
    petsc_std = std(petsc_times)

    # Calculate speedups (relative to Native)
    mpi_vs_native = native_median / mpi_median
    petsc_vs_native = native_median / petsc_median
    mpi_vs_petsc = petsc_median / mpi_median

    results[L] = (
        native_median = native_median,
        native_min = native_min,
        native_std = native_std,
        mpi_median = mpi_median,
        mpi_min = mpi_min,
        mpi_std = mpi_std,
        petsc_median = petsc_median,
        petsc_min = petsc_min,
        petsc_std = petsc_std,
        mpi_vs_native = mpi_vs_native,
        petsc_vs_native = petsc_vs_native,
        mpi_vs_petsc = mpi_vs_petsc
    )

    io0("  Native: median=$(round(native_median, digits=4))s, min=$(round(native_min, digits=4))s, std=$(round(native_std, digits=4))s")
    io0("  MPI:    median=$(round(mpi_median, digits=4))s, min=$(round(mpi_min, digits=4))s, std=$(round(mpi_std, digits=4))s")
    io0("  PETSc:  median=$(round(petsc_median, digits=4))s, min=$(round(petsc_min, digits=4))s, std=$(round(petsc_std, digits=4))s")
    io0("  Speedup vs Native: MPI=$(round(mpi_vs_native, digits=2))x, PETSc=$(round(petsc_vs_native, digits=2))x")
    io0("  Speedup MPI vs PETSc: $(round(mpi_vs_petsc, digits=2))x")
end

# Summary table
io0("\n" * "="^80)
io0("Summary Table (median times)")
io0("="^80)
io0("\n  L    Native        MPI           PETSc         MPI/Native  PETSc/Native  MPI/PETSc")
io0("  -    ------        ---           -----         ----------  ------------  ---------")
for L in 1:3
    r = results[L]
    native_str = lpad(round(r.native_median, digits=3), 8) * "s"
    mpi_str = lpad(round(r.mpi_median, digits=3), 8) * "s"
    petsc_str = lpad(round(r.petsc_median, digits=3), 8) * "s"
    mpi_native_str = lpad(round(r.mpi_vs_native, digits=2), 8) * "x"
    petsc_native_str = lpad(round(r.petsc_vs_native, digits=2), 8) * "x"
    mpi_petsc_str = lpad(round(r.mpi_vs_petsc, digits=2), 8) * "x"
    io0("  $L    $native_str    $mpi_str    $petsc_str    $mpi_native_str    $petsc_native_str   $mpi_petsc_str")
end

io0("\n  Speedup > 1 means faster than baseline")
io0("  MPI/Native, PETSc/Native: >1 means Native is faster")
io0("  MPI/PETSc: >1 means MPI is faster than PETSc")
io0("="^80)
