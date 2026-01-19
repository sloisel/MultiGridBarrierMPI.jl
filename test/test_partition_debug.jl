using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

using HPCMultiGridBarrier
HPCMultiGridBarrier.Init()

using HPCSparseArrays
using HPCSparseArrays: HPCVector, HPCMatrix, HPCSparseMatrix, io0
using LinearAlgebra
using SparseArrays
using MultiGridBarrier

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println(io0(), "[DEBUG] Partition debugging test (nranks=$nranks)")

# Create geometry
g_hpc = fem1d_hpc(Float64; L=2)
g_native = fem1d(Float64; L=2)

n = length(g_hpc.w)

# Get operators
D_dx_hpc = g_hpc.operators[:dx]
Z_mpi = HPCSparseMatrix{Float64}(spzeros(Float64, n, n))

# Create wide operator: D0_dx = [D_dx | Z] (8 x 16)
D0_dx_hpc = hcat(D_dx_hpc, Z_mpi)

println(io0(), "[DEBUG] D_dx partitions:")
println(io0(), "[DEBUG]   D_dx row_partition: $(D_dx_hpc.row_partition)")
println(io0(), "[DEBUG]   D_dx col_partition: $(D_dx_hpc.col_partition)")

println(io0(), "[DEBUG] Z partitions:")
println(io0(), "[DEBUG]   Z row_partition: $(Z_mpi.row_partition)")
println(io0(), "[DEBUG]   Z col_partition: $(Z_mpi.col_partition)")

println(io0(), "[DEBUG] D0_dx partitions (after hcat):")
println(io0(), "[DEBUG]   D0_dx row_partition: $(D0_dx_hpc.row_partition)")
println(io0(), "[DEBUG]   D0_dx col_partition: $(D0_dx_hpc.col_partition)")
println(io0(), "[DEBUG]   D0_dx size: $(size(D0_dx_hpc))")

# Check the transpose
println(io0(), "[DEBUG] Checking transpose D0_dx'...")
D0_dx_t = D0_dx_hpc'

# Is it lazy or materialized?
println(io0(), "[DEBUG]   typeof(D0_dx'): $(typeof(D0_dx_t))")

# Try materializing if it's lazy
if D0_dx_t isa LinearAlgebra.Adjoint
    println(io0(), "[DEBUG]   D0_dx' is lazy Adjoint")
    D0_dx_t_parent = parent(D0_dx_t)
    println(io0(), "[DEBUG]   Parent row_partition: $(D0_dx_t_parent.row_partition)")
    println(io0(), "[DEBUG]   Parent col_partition: $(D0_dx_t_parent.col_partition)")
end

# Create diagonal weight matrix
w_hpc = g_hpc.w
w_native = g_native.w

y11 = ones(n) * 0.5
foo_mpi = spdiagm(n, n, 0 => w_hpc .* HPCVector(y11))
foo_native = spdiagm(n, n, 0 => w_native .* y11)

println(io0(), "[DEBUG] foo (diagonal) partitions:")
println(io0(), "[DEBUG]   foo row_partition: $(foo_mpi.row_partition)")
println(io0(), "[DEBUG]   foo col_partition: $(foo_mpi.col_partition)")

# Now compute D0_dx' * foo
println(io0(), "[DEBUG] Computing D0_dx' * foo...")
tmp_mpi = D0_dx_t * foo_mpi

println(io0(), "[DEBUG] tmp = D0_dx' * foo:")
println(io0(), "[DEBUG]   tmp size: $(size(tmp_mpi))")
println(io0(), "[DEBUG]   tmp row_partition: $(tmp_mpi.row_partition)")
println(io0(), "[DEBUG]   tmp col_partition: $(tmp_mpi.col_partition)")

tmp_native = SparseMatrixCSC(tmp_mpi)
println(io0(), "[DEBUG]   tmp nnz: $(nnz(tmp_native))")

# Compare with native
D_dx_native = g_native.operators[:dx]
Z_native = spzeros(Float64, n, n)
D0_dx_native = hcat(D_dx_native, Z_native)
tmp_native_ref = D0_dx_native' * foo_native

if rank == 0
    println("[DEBUG]   Native tmp nnz: $(nnz(tmp_native_ref))")
    diff = norm(tmp_native - tmp_native_ref)
    println("[DEBUG]   tmp difference: $diff")

    if diff > 1e-10
        println("[DEBUG]   Non-zero differences in tmp:")
        for i in 1:size(tmp_native_ref, 1)
            for j in 1:size(tmp_native_ref, 2)
                mpi_val = tmp_native[i,j]
                native_val = tmp_native_ref[i,j]
                if abs(mpi_val - native_val) > 1e-14
                    println("[DEBUG]     ($i, $j): MPI=$mpi_val, Native=$native_val")
                end
            end
        end
    end
end

# Now compute tmp * D0_dx = D0_dx' * foo * D0_dx
println(io0(), "[DEBUG] Computing tmp * D0_dx...")
result_mpi = tmp_mpi * D0_dx_hpc

println(io0(), "[DEBUG] result = D0_dx' * foo * D0_dx:")
println(io0(), "[DEBUG]   result size: $(size(result_mpi))")
println(io0(), "[DEBUG]   result row_partition: $(result_mpi.row_partition)")
println(io0(), "[DEBUG]   result col_partition: $(result_mpi.col_partition)")

result_native = SparseMatrixCSC(result_mpi)
result_native_ref = D0_dx_native' * foo_native * D0_dx_native

println(io0(), "[DEBUG]   result nnz: $(nnz(result_native))")

if rank == 0
    println("[DEBUG]   Native result nnz: $(nnz(result_native_ref))")
    diff = norm(result_native - result_native_ref)
    println("[DEBUG]   result difference: $diff")

    if diff > 1e-10
        println("[DEBUG]   Non-zero differences in result:")
        for i in 1:size(result_native_ref, 1)
            for j in 1:size(result_native_ref, 2)
                mpi_val = result_native[i,j]
                native_val = result_native_ref[i,j]
                if abs(mpi_val - native_val) > 1e-14
                    println("[DEBUG]     ($i, $j): MPI=$mpi_val, Native=$native_val")
                end
            end
        end
    end
end

println(io0(), "[DEBUG] Test completed")
