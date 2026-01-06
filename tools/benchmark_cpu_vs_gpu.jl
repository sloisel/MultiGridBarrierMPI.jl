#!/usr/bin/env julia
#
# Benchmark: CPU vs GPU for fem2d_mpi_solve
#
# Run with:
#   mpiexec -n 1 julia --project=. tools/benchmark_cpu_vs_gpu.jl
#
# Note: Metal only supports Float32, so we use Float32 for both CPU and GPU
#       to ensure a fair comparison.

using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

println("Loading packages...")
using Metal
using MultiGridBarrierMPI
using MultiGridBarrier
using LinearAlgebraMPI
using BenchmarkTools
using Printf

println("\n" * "="^70)
println("Benchmark: fem2d_mpi_solve - CPU vs GPU")
println("  MPI ranks: $(MPI.Comm_size(comm))")
println("  Element type: Float32 (Metal requirement)")
println("  Running L = 1:7")
println("="^70)

# Store results
results = Vector{NamedTuple}()

for L in 1:7
    # Get grid size
    g = fem2d(Float32; L=L)
    n = size(g.x, 1)

    println("\n--- L = $L (n = $n) ---")

    # Benchmark CPU
    println("  Benchmarking CPU...")
    LinearAlgebraMPI.clear_plan_cache!()
    b_cpu = @benchmark fem2d_mpi_solve(Float32; L=$L, verbose=false) samples=1 evals=1
    cpu_time = median(b_cpu.times) / 1e9

    # Benchmark GPU
    println("  Benchmarking GPU...")
    LinearAlgebraMPI.clear_plan_cache!()
    b_gpu = @benchmark fem2d_mpi_solve(Float32; L=$L, backend=LinearAlgebraMPI.mtl, verbose=false) samples=1 evals=1
    gpu_time = median(b_gpu.times) / 1e9

    push!(results, (L=L, n=n, cpu=cpu_time, gpu=gpu_time))

    # Print results
    speedup = cpu_time / gpu_time
    println("  CPU:  $(round(cpu_time, digits=3))s")
    println("  GPU:  $(round(gpu_time, digits=3))s")
    if speedup > 1
        println("  Speedup: $(round(speedup, digits=2))x (GPU faster)")
    else
        println("  Slowdown: $(round(1/speedup, digits=2))x (CPU faster)")
    end
end

# Summary table
println("\n" * "="^70)
println("Summary")
println("="^70)
println("\n  L       n         CPU        GPU      Speedup")
println("  -       -         ---        ---      -------")
for r in results
    n_str = lpad(r.n, 7)
    cpu_str = @sprintf("%6.3fs", r.cpu)
    gpu_str = @sprintf("%6.3fs", r.gpu)

    speedup = r.cpu / r.gpu
    if speedup > 1
        speedup_str = @sprintf("%.2fx GPU", speedup)
    else
        speedup_str = @sprintf("%.2fx CPU", 1/speedup)
    end
    speedup_str = lpad(speedup_str, 10)

    println("  $(r.L)    $n_str    $cpu_str    $gpu_str    $speedup_str")
end

println("\n  Speedup = CPU time / GPU time (>1 means GPU is faster)")
println("="^70)
