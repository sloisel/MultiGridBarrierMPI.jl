using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

using MultiGridBarrierMPI
MultiGridBarrierMPI.Init()

using HPCLinearAlgebra
using HPCLinearAlgebra: HPCVector, HPCMatrix, io0
using LinearAlgebra
using MultiGridBarrier

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Testing map_rows function")
    flush(stdout)
end

# Test 1: Simple scalar function on HPCVector
n = 8
v_native = collect(Float64, 1:n)
v_mpi = HPCVector(v_native)

result_mpi = MultiGridBarrier.map_rows(x -> x[1]^2, v_mpi)
result_native = Vector(result_mpi)
expected = v_native .^ 2

if rank == 0
    println("[DEBUG] Test 1: Scalar function")
    println("[DEBUG] Input: $v_native")
    println("[DEBUG] Expected: $expected")
    println("[DEBUG] Got: $result_native")
    println("[DEBUG] Match: $(result_native ≈ expected)")
    flush(stdout)
end

@test result_native ≈ expected

# Test 2: Function on two HPCVectors
w_native = collect(Float64, n:-1:1)
w_mpi = HPCVector(w_native)

result2_mpi = MultiGridBarrier.map_rows((x, y) -> x[1] * y[1], v_mpi, w_mpi)
result2_native = Vector(result2_mpi)
expected2 = v_native .* w_native

if rank == 0
    println("[DEBUG] Test 2: Function on two vectors")
    println("[DEBUG] v: $v_native")
    println("[DEBUG] w: $w_native")
    println("[DEBUG] Expected: $expected2")
    println("[DEBUG] Got: $result2_native")
    println("[DEBUG] Match: $(result2_native ≈ expected2)")
    flush(stdout)
end

@test result2_native ≈ expected2

# Test 3: Function returning row vector -> HPCMatrix
result3_mpi = MultiGridBarrier.map_rows(x -> [x[1], x[1]^2, x[1]^3]', v_mpi)
result3_native = Matrix(result3_mpi)
expected3 = hcat(v_native, v_native.^2, v_native.^3)

if rank == 0
    println("[DEBUG] Test 3: Row vector output")
    println("[DEBUG] Expected:")
    println(expected3)
    println("[DEBUG] Got:")
    println(result3_native)
    println("[DEBUG] Match: $(result3_native ≈ expected3)")
    flush(stdout)
end

@test result3_native ≈ expected3

# Test 4: Function on HPCMatrix
m_native = reshape(collect(Float64, 1:16), 8, 2)
m_mpi = HPCMatrix(m_native)

result4_mpi = MultiGridBarrier.map_rows(x -> sum(x)^2, m_mpi)
result4_native = Vector(result4_mpi)
expected4 = [sum(m_native[i,:])^2 for i in 1:8]

if rank == 0
    println("[DEBUG] Test 4: Function on HPCMatrix")
    println("[DEBUG] Input matrix:")
    println(m_native)
    println("[DEBUG] Expected: $expected4")
    println("[DEBUG] Got: $result4_native")
    println("[DEBUG] Match: $(result4_native ≈ expected4)")
    flush(stdout)
end

@test result4_native ≈ expected4

if rank == 0
    println("[DEBUG] All map_rows tests completed successfully")
    flush(stdout)
end
