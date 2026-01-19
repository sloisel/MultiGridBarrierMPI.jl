#!/usr/bin/env julia
#
# Profile breakdown: measure time spent in different operations (CPU vs GPU)
#
# Run with:
#   mpiexec -n 1 julia --project=MultiGridBarrierMPI.jl MultiGridBarrierMPI.jl/tools/profile_breakdown.jl

using MPI
MPI.Init()

println("Loading packages...")
using Metal
using MultiGridBarrierMPI
using MultiGridBarrier
using HPCSparseArrays
using HPCSparseArrays: mtl
using LinearAlgebra
using SparseArrays: nnz
using BenchmarkTools

L = 7

println("\n" * "="^70)
println("Profile breakdown at L=$L")
println("="^70)

# Create geometries
println("\nCreating geometries...")
g_cpu = fem2d_mpi(Float32; L=L)
g_gpu = fem2d_mpi(Float32; L=L, backend=HPCSparseArrays.mtl)

n = size(g_cpu.x, 1)
println("  Problem size n=$n")

# Get a sparse matrix from the geometry
D_cpu = g_cpu.operators[:dx]
D_gpu = g_gpu.operators[:dx]

println("  Sparse matrix nnz=$(nnz(D_cpu))")

# Create test vectors
x_cpu = HPCVector(randn(Float32, n))
x_gpu = mtl(x_cpu)
y_cpu = HPCVector(randn(Float32, n))
y_gpu = mtl(y_cpu)

# Benchmark SpMV
println("\n--- SpMV Benchmark ---")
b_cpu = @benchmark $D_cpu * $x_cpu samples=20
b_gpu = @benchmark $D_gpu * $x_gpu samples=20
cpu_ms = median(b_cpu.times)/1e6
gpu_ms = median(b_gpu.times)/1e6
println("  CPU: $(round(cpu_ms, digits=3)) ms")
println("  GPU: $(round(gpu_ms, digits=3)) ms")
println("  Ratio: $(round(cpu_ms/gpu_ms, digits=2))x (>1 = GPU faster)")

# Benchmark dot product
println("\n--- Dot Product Benchmark ---")
b_cpu_dot = @benchmark dot($x_cpu, $y_cpu) samples=20
b_gpu_dot = @benchmark dot($x_gpu, $y_gpu) samples=20
cpu_ms = median(b_cpu_dot.times)/1e6
gpu_ms = median(b_gpu_dot.times)/1e6
println("  CPU: $(round(cpu_ms, digits=3)) ms")
println("  GPU: $(round(gpu_ms, digits=3)) ms")
println("  Ratio: $(round(cpu_ms/gpu_ms, digits=2))x")

# Benchmark element-wise broadcast
println("\n--- Broadcast (x .* y) Benchmark ---")
b_cpu_bc = @benchmark $x_cpu .* $y_cpu samples=20
b_gpu_bc = @benchmark $x_gpu .* $y_gpu samples=20
cpu_ms = median(b_cpu_bc.times)/1e6
gpu_ms = median(b_gpu_bc.times)/1e6
println("  CPU: $(round(cpu_ms, digits=3)) ms")
println("  GPU: $(round(gpu_ms, digits=3)) ms")
println("  Ratio: $(round(cpu_ms/gpu_ms, digits=2))x")

# Benchmark map_rows_gpu with simple function
println("\n--- map_rows_gpu (x -> x^2) Benchmark ---")
b_cpu_map = @benchmark HPCSparseArrays.map_rows_gpu(x -> x^2, $x_cpu) samples=20
b_gpu_map = @benchmark HPCSparseArrays.map_rows_gpu(x -> x^2, $x_gpu) samples=20
cpu_ms = median(b_cpu_map.times)/1e6
gpu_ms = median(b_gpu_map.times)/1e6
println("  CPU: $(round(cpu_ms, digits=3)) ms")
println("  GPU: $(round(gpu_ms, digits=3)) ms")
println("  Ratio: $(round(cpu_ms/gpu_ms, digits=2))x")

# Closures with captured data can't compile for GPU - skip this test

# Full solve comparison
println("\n--- Full Solve Benchmark ---")
println("  CPU solve (3 samples)...")
b_cpu_solve = @benchmark fem2d_mpi_solve(Float32; L=$L, verbose=false) samples=3
println("  GPU solve (3 samples)...")
b_gpu_solve = @benchmark fem2d_mpi_solve(Float32; L=$L, backend=HPCSparseArrays.mtl, verbose=false) samples=3
cpu_s = median(b_cpu_solve.times)/1e9
gpu_s = median(b_gpu_solve.times)/1e9
println("  CPU: $(round(cpu_s, digits=3)) s")
println("  GPU: $(round(gpu_s, digits=3)) s")
println("  Ratio: $(round(cpu_s/gpu_s, digits=2))x")

println("\n" * "="^70)
println("Summary")
println("="^70)
