#!/usr/bin/env julia
#
# Detailed profile of map_rows with 2 arguments
#
# Run with: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=10 mpiexec -n 1 julia --project=. tools/profile_map_rows_detail.jl
#

using MPI
MPI.Init()

using MultiGridBarrier
using HPCMultiGridBarrier
using HPCSparseArrays
using HPCSparseArrays: HPCVector, HPCMatrix, _get_row_partition, _align_to_partition, _local_rows
using LinearAlgebra

HPCMultiGridBarrier.Init()

const L = 6
println("="^70)
println("Detailed map_rows profile at L=$L")
println("="^70)

# Create geometry
g_hpc = fem2d_hpc(Float64; L=L)
x_hpc = g_hpc.x  # HPCMatrix
w_hpc = g_hpc.w  # HPCVector

println("Grid size (global): ", sum(x_hpc.row_partition))
println("Local rows: ", size(x_hpc.A, 1))

# Check partitions
println("\n--- Partition Check ---")
println("x_hpc row_partition: ", x_hpc.row_partition)
println("w_hpc partition: ", w_hpc.partition)
println("Partitions equal: ", x_hpc.row_partition == w_hpc.partition)

# Breakdown timing
f = (row_x, w) -> w * sum(row_x)

println("\n--- Timing Breakdown ---")

# 1. Get target partition
t1 = time()
for _ in 1:1000
    target_partition = _get_row_partition(x_hpc)
end
t1 = (time() - t1) / 1000
println("1. _get_row_partition: $(round(t1*1e6, digits=3)) μs")

target_partition = _get_row_partition(x_hpc)

# 2. Align arguments
t2a = time()
for _ in 1:1000
    aligned_x = _align_to_partition(x_hpc, target_partition)
end
t2a = (time() - t2a) / 1000
println("2a. _align_to_partition(x_hpc): $(round(t2a*1e6, digits=3)) μs")

t2b = time()
for _ in 1:1000
    aligned_w = _align_to_partition(w_hpc, target_partition)
end
t2b = (time() - t2b) / 1000
println("2b. _align_to_partition(w_hpc): $(round(t2b*1e6, digits=3)) μs")

aligned_x = _align_to_partition(x_hpc, target_partition)
aligned_w = _align_to_partition(w_hpc, target_partition)

# 3. Get local row iterators
t3a = time()
for _ in 1:1000
    row_iter_x = _local_rows(aligned_x)
end
t3a = (time() - t3a) / 1000
println("3a. _local_rows(x_hpc): $(round(t3a*1e6, digits=3)) μs")

t3b = time()
for _ in 1:1000
    row_iter_w = _local_rows(aligned_w)
end
t3b = (time() - t3b) / 1000
println("3b. _local_rows(w_hpc): $(round(t3b*1e6, digits=3)) μs")

row_iter_x = _local_rows(aligned_x)
row_iter_w = _local_rows(aligned_w)

# 4. Apply function with comprehension
t4 = time()
for _ in 1:100
    results = [f(rows...) for rows in zip(row_iter_x, row_iter_w)]
end
t4 = (time() - t4) / 100
println("4. Comprehension [f(rows...) for ...]: $(round(t4*1e3, digits=3)) ms")

# 5. MPI.Allgather overhead
local_info = Int32[1, 1, 1, 0]
t5 = time()
for _ in 1:1000
    all_info = MPI.Allgather(local_info, MPI.COMM_WORLD)
end
t5 = (time() - t5) / 1000
println("5. MPI.Allgather (small): $(round(t5*1e6, digits=3)) μs")

# 6. Full map_rows call
t6 = time()
for _ in 1:100
    result = MultiGridBarrier.map_rows(f, x_hpc, w_hpc)
end
t6 = (time() - t6) / 100
println("\n6. Full map_rows: $(round(t6*1e3, digits=3)) ms")

# Compare with native
x_native = g_hpc.x.A  # Get local part as native
w_native = g_hpc.w.v  # Get local part as native

t_native = time()
for _ in 1:100
    result = MultiGridBarrier.map_rows(f, x_native, w_native)
end
t_native = (time() - t_native) / 100
println("   Native map_rows: $(round(t_native*1e3, digits=3)) ms")
println("   Ratio: $(round(t6/t_native, digits=2))x")

# Now test the actual barrier function pattern
println("\n--- Actual barrier function pattern ---")

# The barrier functions do something like:
# map_rows((Dz_row, w) -> ..., Dz, w)
# where Dz is a HPCMatrix and w is a HPCVector

Dz = x_hpc  # Use x_hpc as a stand-in for Dz
w = w_hpc

barrier_f = (Dz_row, w) -> begin
    s = norm(Dz_row)^2
    return w * log(max(s, 1e-10))
end

t_barrier = time()
for _ in 1:50
    result = MultiGridBarrier.map_rows(barrier_f, Dz, w)
end
t_barrier = (time() - t_barrier) / 50
println("MPI barrier-like map_rows: $(round(t_barrier*1e3, digits=3)) ms")

# Native equivalent
t_native_barrier = time()
for _ in 1:50
    result = MultiGridBarrier.map_rows(barrier_f, Dz.A, w.v)
end
t_native_barrier = (time() - t_native_barrier) / 50
println("Native barrier-like map_rows: $(round(t_native_barrier*1e3, digits=3)) ms")
println("Ratio: $(round(t_barrier/t_native_barrier, digits=2))x")

println("\nDone.")
