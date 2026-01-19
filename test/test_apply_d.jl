using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

using MultiGridBarrierMPI
MultiGridBarrierMPI.Init()

using HPCSparseArrays
using HPCSparseArrays: HPCVector, HPCMatrix, HPCSparseMatrix, io0
using LinearAlgebra
using SparseArrays
using MultiGridBarrier

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println(io0(), "[DEBUG] Testing apply_D function")

# Create geometry
g = fem1d_mpi(Float64; L=3)

# Get operators
D_op = [g.operators[:dx], g.operators[:id]]  # dx and id operators

# Create a test vector (should be size 16)
n = length(g.w)
z_native = sin.(range(0, π, length=n))
z_mpi = HPCVector(z_native)

println(io0(), "[DEBUG] z size: $(length(z_mpi))")
println(io0(), "[DEBUG] z partition: $(z_mpi.partition)")
println(io0(), "[DEBUG] Number of D operators: $(length(D_op))")
for (i, D) in enumerate(D_op)
    println(io0(), "[DEBUG] D[$i] size: $(size(D))")
end

# Test apply_D
println(io0(), "[DEBUG] Computing apply_D(D, z)...")
Dz_mpi = hcat([D * z_mpi for D in D_op]...)
Dz_native = Matrix(Dz_mpi)

# Compare with native
D_native = [SparseMatrixCSC(D) for D in D_op]
Dz_expected = hcat([D * z_native for D in D_native]...)

println(io0(), "[DEBUG] apply_D result size: $(size(Dz_mpi))")
println(io0(), "[DEBUG] apply_D row_partition: $(Dz_mpi.row_partition)")
println(io0(), "[DEBUG] apply_D col_partition: $(Dz_mpi.col_partition)")
println(io0(), "[DEBUG] Dz expected size: $(size(Dz_expected))")
println(io0(), "[DEBUG] Dz match: $(Dz_native ≈ Dz_expected)")

@test Dz_native ≈ Dz_expected

# Test map_rows with apply_D result
println(io0(), "[DEBUG] Testing map_rows with apply_D result...")
x = g.x  # Coordinates

# Simple function: sum of squares at each row
result_mpi = MultiGridBarrier.map_rows((xi, qi) -> sum(qi.^2), x, Dz_mpi)
result_native = Vector(result_mpi)

# Compute expected result
x_native = Matrix(x)
expected = [sum(Dz_expected[i, :].^2) for i in 1:size(Dz_expected, 1)]

println(io0(), "[DEBUG] map_rows result size: $(length(result_mpi))")
println(io0(), "[DEBUG] map_rows result partition: $(result_mpi.partition)")
println(io0(), "[DEBUG] Expected: $expected")
println(io0(), "[DEBUG] Got: $result_native")
println(io0(), "[DEBUG] map_rows match: $(result_native ≈ expected)")

@test result_native ≈ expected

# Test more complex function (row vector output like F2 does)
println(io0(), "[DEBUG] Testing map_rows with row vector output...")
result2_mpi = MultiGridBarrier.map_rows((xi, qi) -> [qi[1]^2, qi[2]^2]', x, Dz_mpi)
result2_native = Matrix(result2_mpi)

# Expected: [[qi[1]^2, qi[2]^2] for each row i]
expected2 = hcat([Dz_expected[i, 1]^2 for i in 1:n], [Dz_expected[i, 2]^2 for i in 1:n])

println(io0(), "[DEBUG] map_rows (row vec) result size: $(size(result2_mpi))")
println(io0(), "[DEBUG] map_rows (row vec) row_partition: $(result2_mpi.row_partition)")
println(io0(), "[DEBUG] Expected size: $(size(expected2))")
println(io0(), "[DEBUG] match: $(result2_native ≈ expected2)")

@test result2_native ≈ expected2

println(io0(), "[DEBUG] All apply_D tests completed")
