#!/usr/bin/env julia
#
# Profile _local_rows behavior for HPCVector
#
# Run with: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=10 mpiexec -n 1 julia --project=. tools/profile_local_rows.jl
#

using MPI
MPI.Init()

using MultiGridBarrier
using HPCMultiGridBarrier
using HPCSparseArrays
using HPCSparseArrays: HPCVector, HPCMatrix, _local_rows
using LinearAlgebra
import Statistics: mean, median

HPCMultiGridBarrier.Init()

const L = 6
const N_ITER = 100

println("="^70)
println("Profile _local_rows for HPCVector")
println("="^70)

g_hpc = fem2d_hpc(Float64; L=L)
x_hpc = g_hpc.x
w_hpc = g_hpc.w

x_local = x_hpc.A
w_local = w_hpc.v
n = length(w_local)

println("n = $n")

# Test function
f = (row_x, w) -> w * sum(row_x)

println("\n" * "-"^70)
println("What _local_rows returns for HPCVector")
println("-"^70)

row_iter_w = _local_rows(w_hpc)
first_w = first(row_iter_w)
println("Type of element from _local_rows(HPCVector): ", typeof(first_w))
println("Value: ", first_w)
println("Is it a scalar? ", first_w isa Number)

println("\n" * "-"^70)
println("Comparing different iteration patterns")
println("-"^70)

# 1. Current MPI style: view(v, i:i) for HPCVector
t1_times = Float64[]
for _ in 1:N_ITER
    row_iter_x = eachrow(x_local)
    row_iter_w = (view(w_local, i:i) for i in 1:n)  # Current _local_rows behavior
    t = time_ns()
    results = [f(rx, rw) for (rx, rw) in zip(row_iter_x, row_iter_w)]
    t = time_ns() - t
    push!(t1_times, t)
end
println("1. view(w, i:i) (current): $(round(median(t1_times)/1000, digits=1)) μs")

# 2. Direct scalar access: w[i]
t2_times = Float64[]
for _ in 1:N_ITER
    row_iter_x = eachrow(x_local)
    t = time_ns()
    results = [f(rx, w_local[i]) for (i, rx) in enumerate(row_iter_x)]
    t = time_ns() - t
    push!(t2_times, t)
end
println("2. Direct w[i]:            $(round(median(t2_times)/1000, digits=1)) μs")

# 3. Using w_local directly in zip
t3_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    results = [f(rx, w) for (rx, w) in zip(eachrow(x_local), w_local)]
    t = time_ns() - t
    push!(t3_times, t)
end
println("3. zip(eachrow, w_local):  $(round(median(t3_times)/1000, digits=1)) μs")

# 4. Using map
t4_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    results = map(f, eachrow(x_local), w_local)
    t = time_ns() - t
    push!(t4_times, t)
end
println("4. map(f, eachrow, w):     $(round(median(t4_times)/1000, digits=1)) μs")

# 5. Modified function that handles view
f_view = (row_x, w) -> w[1] * sum(row_x)  # Access w[1] since w is a view

t5_times = Float64[]
for _ in 1:N_ITER
    row_iter_x = eachrow(x_local)
    row_iter_w = (view(w_local, i:i) for i in 1:n)
    t = time_ns()
    results = [f_view(rx, rw) for (rx, rw) in zip(row_iter_x, row_iter_w)]
    t = time_ns() - t
    push!(t5_times, t)
end
println("5. view + f_view(w[1]):    $(round(median(t5_times)/1000, digits=1)) μs")

println("\n" * "-"^70)
println("Speedup from using scalars instead of views")
println("-"^70)
println("Current (view):  $(round(median(t1_times)/1000, digits=1)) μs")
println("Best (map):      $(round(median(t4_times)/1000, digits=1)) μs")
println("Speedup:         $(round(median(t1_times) / median(t4_times), digits=1))x")

println("\n" * "-"^70)
println("What about when f returns a row vector (like f1/f2)?")
println("-"^70)

f_rowvec = (row_x, w) -> (w * row_x)'  # Returns 1x2 matrix

# Current style
t6_times = Float64[]
for _ in 1:N_ITER
    row_iter_w = (view(w_local, i:i) for i in 1:n)
    t = time_ns()
    results = [f_rowvec(rx, rw[1]) for (rx, rw) in zip(eachrow(x_local), row_iter_w)]
    t = time_ns() - t
    push!(t6_times, t)
end
println("1. view + rw[1]:  $(round(median(t6_times)/1000, digits=1)) μs")

# Direct scalar
t7_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    results = [f_rowvec(rx, w) for (rx, w) in zip(eachrow(x_local), w_local)]
    t = time_ns() - t
    push!(t7_times, t)
end
println("2. Direct scalar: $(round(median(t7_times)/1000, digits=1)) μs")

# map
t8_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    results = map(f_rowvec, eachrow(x_local), w_local)
    t = time_ns() - t
    push!(t8_times, t)
end
println("3. map:           $(round(median(t8_times)/1000, digits=1)) μs")

println("\nDone.")
