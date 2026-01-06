#!/usr/bin/env julia
#
# Benchmark: CPU vs Auto (GPU with size threshold) for fem2d_mpi_solve
#
# Run with:
#   mpiexec -n 1 julia --project=MultiGridBarrierMPI.jl MultiGridBarrierMPI.jl/tools/benchmark_cpu_vs_gpu.jl
#
# Note: Metal only supports Float32, so we use Float32 for both CPU and GPU
#       to ensure a fair comparison.
#
# Two modes:
#   - CPU: Pure CPU (no backend parameter)
#   - Auto: Automatic GPU/CPU selection based on GPU_MIN_SIZE threshold

using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

println("Loading packages...")
using Metal
using MultiGridBarrierMPI
using MultiGridBarrier
using LinearAlgebraMPI
using LinearAlgebraMPI: GPU_MIN_SIZE
using BenchmarkTools
using Printf

# Configurable threshold for "Auto" mode
const AUTO_THRESHOLD = 1000

println("\n" * "="^70)
println("Benchmark: fem2d_mpi_solve - CPU vs Auto")
println("  MPI ranks: $(MPI.Comm_size(comm))")
println("  Element type: Float32 (Metal requirement)")
println("  Auto threshold: GPU_MIN_SIZE = $AUTO_THRESHOLD")
println("  Running L = 1:6")
println("="^70)

# Store results
results = Vector{NamedTuple}()

for L in 1:6
    # Get grid size
    g = fem2d(Float32; L=L)
    n = size(g.x, 1)

    println("\n--- L = $L (n = $n) ---")

    # Benchmark CPU (pure CPU, no backend)
    println("  Benchmarking CPU...")
    LinearAlgebraMPI.clear_plan_cache!()
    b_cpu = @benchmark fem2d_mpi_solve(Float32; L=$L, verbose=false) samples=1 evals=1
    cpu_time = median(b_cpu.times) / 1e9

    # Benchmark Auto (GPU_MIN_SIZE threshold)
    println("  Benchmarking Auto (threshold=$AUTO_THRESHOLD)...")
    LinearAlgebraMPI.clear_plan_cache!()
    GPU_MIN_SIZE[] = AUTO_THRESHOLD
    b_auto = @benchmark fem2d_mpi_solve(Float32; L=$L, backend=LinearAlgebraMPI.mtl, verbose=false) samples=1 evals=1
    auto_time = median(b_auto.times) / 1e9

    # Determine which arrays went to GPU in auto mode
    GPU_MIN_SIZE[] = AUTO_THRESHOLD
    g_test = fem2d_mpi(Float32; L=L, backend=LinearAlgebraMPI.mtl)
    auto_is_gpu = !(g_test.x.A isa Matrix)

    push!(results, (L=L, n=n, cpu=cpu_time, auto=auto_time, auto_gpu=auto_is_gpu))

    # Print results
    speedup = cpu_time / auto_time
    println("  CPU:  $(round(cpu_time, digits=3))s")
    println("  Auto: $(round(auto_time, digits=3))s [$(auto_is_gpu ? "GPU" : "CPU")]")
    if speedup > 1
        println("  Speedup: $(round(speedup, digits=2))x (Auto faster)")
    else
        println("  Speedup: $(round(1/speedup, digits=2))x (CPU faster)")
    end
end

# Summary table
println("\n" * "="^70)
println("Summary")
println("="^70)
println("\n  L       n         CPU        Auto    Speedup   Auto backend")
println("  -       -         ---        ----    -------   ------------")
for r in results
    n_str = lpad(r.n, 7)
    cpu_str = @sprintf("%6.3fs", r.cpu)
    auto_str = @sprintf("%6.3fs", r.auto)

    speedup = r.cpu / r.auto
    if speedup > 1
        speedup_str = @sprintf("%.2fx Auto", speedup)
    else
        speedup_str = @sprintf("%.2fx CPU", 1/speedup)
    end
    speedup_str = lpad(speedup_str, 10)

    auto_backend = r.auto_gpu ? "GPU" : "CPU"
    println("  $(r.L)    $n_str    $cpu_str    $auto_str    $speedup_str       $auto_backend")
end

println("\n  Auto threshold: GPU_MIN_SIZE = $AUTO_THRESHOLD")
println("  Speedup = CPU time / Auto time (>1 means Auto is faster)")
println("="^70)
