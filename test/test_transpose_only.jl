using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

using MultiGridBarrierMPI
MultiGridBarrierMPI.Init()

using HPCSparseArrays
using HPCSparseArrays: HPCVector, HPCMatrix, HPCSparseMatrix, io0, materialize_transpose
using LinearAlgebra
using SparseArrays
using MultiGridBarrier

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println(io0(), "[DEBUG] Transpose-only test (nranks=$nranks)")

# Create geometry to get the D_dx operator
g_mpi = fem1d_mpi(Float64; L=2)
g_native = fem1d(Float64; L=2)

n = length(g_mpi.w)

# Get operators
D_dx_mpi = g_mpi.operators[:dx]
Z_mpi = HPCSparseMatrix{Float64}(spzeros(Float64, n, n))
D_dx_native = g_native.operators[:dx]
Z_native = spzeros(Float64, n, n)

# Test 1: Transpose of D_dx
println(io0(), "[DEBUG] Test 1: Transpose of D_dx")
D_dx_t_mpi = materialize_transpose(D_dx_mpi)
D_dx_t_native = SparseMatrixCSC(D_dx_t_mpi)

if rank == 0
    D_dx_t_ref = sparse(D_dx_native')
    diff = norm(D_dx_t_native - D_dx_t_ref)
    println("[DEBUG] D_dx' difference: $diff")
    @test diff < 1e-12
end

# Test 2: Create D0_dx = hcat(D_dx, Z) and test its transpose
println(io0(), "[DEBUG] Test 2: D0_dx = hcat(D_dx, Z) and its transpose")
D0_dx_mpi = hcat(D_dx_mpi, Z_mpi)
D0_dx_native = hcat(D_dx_native, Z_native)

println(io0(), "[DEBUG] D0_dx size: $(size(D0_dx_mpi))")
println(io0(), "[DEBUG] D0_dx row_partition: $(D0_dx_mpi.row_partition)")
println(io0(), "[DEBUG] D0_dx col_partition: $(D0_dx_mpi.col_partition)")

# Materialize the transpose
D0_dx_t_mpi = materialize_transpose(D0_dx_mpi)

println(io0(), "[DEBUG] D0_dx' size: $(size(D0_dx_t_mpi))")
println(io0(), "[DEBUG] D0_dx' row_partition: $(D0_dx_t_mpi.row_partition)")
println(io0(), "[DEBUG] D0_dx' col_partition: $(D0_dx_t_mpi.col_partition)")

D0_dx_t_mpi_native = SparseMatrixCSC(D0_dx_t_mpi)
D0_dx_t_native_ref = sparse(D0_dx_native')

if rank == 0
    println("[DEBUG] D0_dx' MPI nnz: $(nnz(D0_dx_t_mpi_native))")
    println("[DEBUG] D0_dx' native nnz: $(nnz(D0_dx_t_native_ref))")
    diff = norm(D0_dx_t_mpi_native - D0_dx_t_native_ref)
    println("[DEBUG] D0_dx' difference: $diff")

    if diff > 1e-10
        println("[DEBUG] Showing non-zero differences:")
        for i in 1:size(D0_dx_t_native_ref, 1)
            for j in 1:size(D0_dx_t_native_ref, 2)
                mpi_val = D0_dx_t_mpi_native[i,j]
                native_val = D0_dx_t_native_ref[i,j]
                if abs(mpi_val - native_val) > 1e-14
                    println("[DEBUG]   ($i, $j): MPI=$mpi_val, Native=$native_val")
                end
            end
        end
    end
    @test diff < 1e-12
end

# Test 3: Now test the multiplication D0_dx' * foo where foo is diagonal
println(io0(), "[DEBUG] Test 3: D0_dx' * foo (diagonal)")

w_mpi = g_mpi.w
w_native = g_native.w
y11 = ones(n) * 0.5
foo_mpi = spdiagm(n, n, 0 => w_mpi .* HPCVector(y11))
foo_native = spdiagm(n, n, 0 => w_native .* y11)

println(io0(), "[DEBUG] foo size: $(size(foo_mpi))")
println(io0(), "[DEBUG] foo row_partition: $(foo_mpi.row_partition)")
println(io0(), "[DEBUG] foo col_partition: $(foo_mpi.col_partition)")

# Multiply
tmp_mpi = D0_dx_t_mpi * foo_mpi
tmp_mpi_native = SparseMatrixCSC(tmp_mpi)
tmp_native_ref = D0_dx_t_native_ref * foo_native

println(io0(), "[DEBUG] tmp = D0_dx' * foo")
println(io0(), "[DEBUG] tmp size: $(size(tmp_mpi))")
println(io0(), "[DEBUG] tmp row_partition: $(tmp_mpi.row_partition)")
println(io0(), "[DEBUG] tmp col_partition: $(tmp_mpi.col_partition)")

if rank == 0
    println("[DEBUG] tmp MPI nnz: $(nnz(tmp_mpi_native))")
    println("[DEBUG] tmp native nnz: $(nnz(tmp_native_ref))")
    diff = norm(tmp_mpi_native - tmp_native_ref)
    println("[DEBUG] tmp difference: $diff")

    if diff > 1e-10
        println("[DEBUG] Showing non-zero differences in tmp:")
        for i in 1:size(tmp_native_ref, 1)
            for j in 1:size(tmp_native_ref, 2)
                mpi_val = tmp_mpi_native[i,j]
                native_val = tmp_native_ref[i,j]
                if abs(mpi_val - native_val) > 1e-14
                    println("[DEBUG]   ($i, $j): MPI=$mpi_val, Native=$native_val")
                end
            end
        end
    end
    @test diff < 1e-12
end

println(io0(), "[DEBUG] Test completed")
