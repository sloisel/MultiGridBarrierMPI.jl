#!/usr/bin/env julia
using MPI
MPI.Init()

using MultiGridBarrierMPI
using HPCLinearAlgebra
using HPCLinearAlgebra: HPCVector, HPCMatrix, _local_rows, HPCVector_local

MultiGridBarrierMPI.Init()

g = fem2d_mpi(Float64; L=6)
x = g.x
w = g.w

println("x type: ", typeof(x))
println("w type: ", typeof(w))

row_iters = (_local_rows(x), _local_rows(w))
println("\nrow_iters types:")
println("  _local_rows(x): ", typeof(row_iters[1]))
println("  _local_rows(w): ", typeof(row_iters[2]))

println("\nFirst few items from zip(row_iters...):")
for (i, items) in enumerate(zip(row_iters...))
    println("  $i: types = $(typeof.(items))")
    if i >= 3
        break
    end
end

f = (row_x, w) -> w * sum(row_x)

println("\nTiming comprehension vs loop:")

# Comprehension with zip
t1 = time_ns()
results1 = [f(rows...) for rows in zip(row_iters...)]
t1 = (time_ns() - t1) / 1000
println("Comprehension: $(round(t1, digits=1)) μs, length=$(length(results1))")

# Simple loop
row_iters2 = (_local_rows(x), _local_rows(w))
t2 = time_ns()
results2 = Float64[]
for rows in zip(row_iters2...)
    push!(results2, f(rows...))
end
t2 = (time_ns() - t2) / 1000
println("Simple loop: $(round(t2, digits=1)) μs, length=$(length(results2))")

# Direct access loop
local_x = x.A
local_w = w.v
t3 = time_ns()
results3 = Vector{Float64}(undef, size(local_x, 1))
for i in 1:size(local_x, 1)
    results3[i] = f(view(local_x, i, :), local_w[i])
end
t3 = (time_ns() - t3) / 1000
println("Direct loop: $(round(t3, digits=1)) μs, length=$(length(results3))")

# Using map with eachrow and vector
t4 = time_ns()
results4 = collect(map(f, eachrow(local_x), local_w))
t4 = (time_ns() - t4) / 1000
println("map(f, eachrow, v): $(round(t4, digits=1)) μs, length=$(length(results4))")

# What about map with _local_rows?
row_iters3 = (_local_rows(x), _local_rows(w))
t5 = time_ns()
results5 = collect(map(f, row_iters3...))
t5 = (time_ns() - t5) / 1000
println("map(f, _local_rows...): $(round(t5, digits=1)) μs, length=$(length(results5))")

# Now time HPCVector_local
println("\nTiming HPCVector_local:")
t6 = time_ns()
v = HPCVector_local(results3)
t6 = (time_ns() - t6) / 1000
println("HPCVector_local: $(round(t6, digits=1)) μs")

println("\nDone.")
