#!/usr/bin/env julia
#
# Why is [f(rows...) for rows in zip(...)] slower than vcat((f.((eachrow.(A))...))...)?
#
# Run with: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=10 mpiexec -n 1 julia --project=. tools/profile_comprehension.jl
#

using MPI
MPI.Init()

using MultiGridBarrier
using MultiGridBarrierMPI
using HPCLinearAlgebra
using HPCLinearAlgebra: HPCVector, HPCMatrix, _local_rows
using LinearAlgebra
import Statistics: mean, median

MultiGridBarrierMPI.Init()

const L = 6
const N_ITER = 100

println("="^70)
println("Comprehension vs Broadcast comparison at L=$L")
println("="^70)

# Create data
g_mpi = fem2d_mpi(Float64; L=L)
x_mpi = g_mpi.x
w_mpi = g_mpi.w

# Get local arrays
x_local = x_mpi.A  # Matrix{Float64}
w_local = w_mpi.v  # Vector{Float64}

n = size(x_local, 1)
println("Local rows: $n")

# Test function
f = (row_x, w) -> w * sum(row_x)

println("\n" * "-"^70)
println("Different iteration approaches (all on local data)")
println("-"^70)

# Method 1: Native map_rows (broadcast + vcat)
t1_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = vcat((f.((eachrow.(( x_local, w_local )))...))...)
    t = time_ns() - t
    push!(t1_times, t)
end
println("1. Native: vcat((f.((eachrow.(A))...))...)")
println("   Median: $(round(median(t1_times)/1000, digits=1)) μs")

# Method 2: MPI-style comprehension with zip and _local_rows iterators
t2_times = Float64[]
for _ in 1:N_ITER
    row_iter_x = _local_rows(x_mpi)
    row_iter_w = _local_rows(w_mpi)
    t = time_ns()
    result = [f(rows...) for rows in zip(row_iter_x, row_iter_w)]
    t = time_ns() - t
    push!(t2_times, t)
end
println("2. MPI-style: [f(rows...) for zip(_local_rows...)]")
println("   Median: $(round(median(t2_times)/1000, digits=1)) μs")

# Method 3: Simple comprehension with eachrow
t3_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = [f(row_x, w) for (row_x, w) in zip(eachrow(x_local), w_local)]
    t = time_ns() - t
    push!(t3_times, t)
end
println("3. Simple: [f(row_x, w) for (row_x, w) in zip(eachrow, w)]")
println("   Median: $(round(median(t3_times)/1000, digits=1)) μs")

# Method 4: Pre-allocated loop
t4_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        result[i] = f(view(x_local, i, :), w_local[i])
    end
    t = time_ns() - t
    push!(t4_times, t)
end
println("4. Pre-alloc loop with view")
println("   Median: $(round(median(t4_times)/1000, digits=1)) μs")

# Method 5: Inline the function
t5_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        result[i] = w_local[i] * (x_local[i,1] + x_local[i,2])
    end
    t = time_ns() - t
    push!(t5_times, t)
end
println("5. Inline loop (no function call)")
println("   Median: $(round(median(t5_times)/1000, digits=1)) μs")

# Method 6: Using map
t6_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = map(f, eachrow(x_local), w_local)
    t = time_ns() - t
    push!(t6_times, t)
end
println("6. map(f, eachrow(x), w)")
println("   Median: $(round(median(t6_times)/1000, digits=1)) μs")

# Now investigate what _local_rows returns
println("\n" * "-"^70)
println("What does _local_rows return?")
println("-"^70)

row_iter_x = _local_rows(x_mpi)
row_iter_w = _local_rows(w_mpi)

println("_local_rows(HPCMatrix) type: ", typeof(row_iter_x))
println("_local_rows(HPCVector) type: ", typeof(row_iter_w))
println("eachrow(Matrix) type: ", typeof(eachrow(x_local)))

# Test if the issue is in how _local_rows generates views
println("\n" * "-"^70)
println("Isolate _local_rows overhead")
println("-"^70)

# Time just iterating (no function call)
t_iter1 = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    s = 0.0
    for (rx, rw) in zip(eachrow(x_local), w_local)
        s += rx[1]  # minimal work
    end
    t = time_ns() - t
    push!(t_iter1, t)
end
println("Iterate with eachrow + w_local: $(round(median(t_iter1)/1000, digits=1)) μs")

t_iter2 = Float64[]
for _ in 1:N_ITER
    row_iter_x = _local_rows(x_mpi)
    row_iter_w = _local_rows(w_mpi)
    t = time_ns()
    s = 0.0
    for (rx, rw) in zip(row_iter_x, row_iter_w)
        s += rx[1]
    end
    t = time_ns() - t
    push!(t_iter2, t)
end
println("Iterate with _local_rows: $(round(median(t_iter2)/1000, digits=1)) μs")

println("\nDone.")
