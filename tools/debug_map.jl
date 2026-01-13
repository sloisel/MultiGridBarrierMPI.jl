#!/usr/bin/env julia
using MPI
MPI.Init()

using MultiGridBarrierMPI
using HPCLinearAlgebra
using HPCLinearAlgebra: HPCVector, HPCMatrix, _local_rows

MultiGridBarrierMPI.Init()

g = fem2d_mpi(Float64; L=4)
x = g.x
w = g.w

println("x type: ", typeof(x))
println("w type: ", typeof(w))

row_iters = (_local_rows(x), _local_rows(w))
println("\nrow_iters types:")
println("  _local_rows(x): ", typeof(row_iters[1]))
println("  _local_rows(w): ", typeof(row_iters[2]))

f = (row_x, w) -> w * sum(row_x)

println("\nTrying map(f, row_iters...):")
result = collect(map(f, row_iters...))
println("Result type: ", typeof(result))
println("Result length: ", length(result))
println("First 5 elements: ", result[1:min(5, length(result))])

println("\nTrying with eachrow and vector directly:")
result2 = collect(map(f, eachrow(x.A), w.v))
println("Result2 type: ", typeof(result2))
println("Result2 length: ", length(result2))

println("\nDone.")
