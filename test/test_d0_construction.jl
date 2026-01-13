using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

using MultiGridBarrierMPI
MultiGridBarrierMPI.Init()

using HPCLinearAlgebra
using HPCLinearAlgebra: HPCVector, HPCMatrix, HPCSparseMatrix, io0
using LinearAlgebra
using SparseArrays
using MultiGridBarrier

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println(io0(), "[DEBUG] D0 matrix construction test (nranks=$nranks)")

# Create geometry
g_mpi = fem1d_mpi(Float64; L=2)
g_native = fem1d(Float64; L=2)

# Get operators and subspaces
D_dx_mpi = g_mpi.operators[:dx]
D_id_mpi = g_mpi.operators[:id]
D_dx_native = g_native.operators[:dx]
D_id_native = g_native.operators[:id]

n = size(D_dx_mpi, 1)

# Test hcat of sparse matrices
println(io0(), "[DEBUG] Testing hcat of sparse matrices...")

# Create zero matrix
Z_mpi = HPCSparseMatrix{Float64}(spzeros(Float64, n, n))
Z_native = spzeros(Float64, n, n)

# Test 1: hcat(Z, D_dx) - like foo = [Z; D_dx] in amg_helper
D0_1_mpi = hcat(Z_mpi, D_dx_mpi)
D0_1_native = hcat(Z_native, D_dx_native)

println(io0(), "[DEBUG] hcat(Z, D_dx) size: MPI=$(size(D0_1_mpi)), native=$(size(D0_1_native))")

D0_1_mpi_native = SparseMatrixCSC(D0_1_mpi)
if rank == 0
    println("[DEBUG] hcat(Z, D_dx) nnz: MPI=$(nnz(D0_1_mpi_native)), native=$(nnz(D0_1_native))")
    diff = norm(D0_1_mpi_native - D0_1_native)
    println("[DEBUG] Difference: $diff")
    @test diff < 1e-12
end

# Test 2: hcat(D_dx, Z) - operator in first position
D0_2_mpi = hcat(D_dx_mpi, Z_mpi)
D0_2_native = hcat(D_dx_native, Z_native)

D0_2_mpi_native = SparseMatrixCSC(D0_2_mpi)
if rank == 0
    println("[DEBUG] hcat(D_dx, Z) nnz: MPI=$(nnz(D0_2_mpi_native)), native=$(nnz(D0_2_native))")
    diff = norm(D0_2_mpi_native - D0_2_native)
    println("[DEBUG] Difference: $diff")
    @test diff < 1e-12
end

# Test 3: Test blockdiag for restriction matrices
println(io0(), "[DEBUG] Testing blockdiag...")

w_mpi = g_mpi.w
w_native = g_native.w

# Test 4: Now with restriction matrices (like the full Newton system)
println(io0(), "[DEBUG] Testing with restriction matrices...")

R_mpi = g_mpi.subspaces[:dirichlet][end]
R_native = g_native.subspaces[:dirichlet][end]

# Build block diagonal restriction: R_block = [R 0; 0 R]
R_block_mpi = blockdiag(R_mpi, R_mpi)
R_block_native = blockdiag(R_native, R_native)

println(io0(), "[DEBUG] R_block size: MPI=$(size(R_block_mpi)), native=$(size(R_block_native))")

# Build wide D0 that acts on the blocked state
# D0 takes [u; s] and produces [dx(u), id(s)]
D0_wide_mpi = hcat(hcat(D_dx_mpi, Z_mpi), hcat(Z_mpi, D_id_mpi))  # This isn't quite right...

# Actually, let me reconsider the structure.
# In amg_helper:
# D0[l,k] = hcat(foo...) where foo[bar[D[k,1]]] = operator
# For k=1 (dx applied to u): foo = [D_dx, Z]
# For k=2 (id applied to s): foo = [Z, D_id]

D0_dx_mpi = hcat(D_dx_mpi, Z_mpi)  # dx applied to first state variable
D0_id_mpi = hcat(Z_mpi, D_id_mpi)   # id applied to second state variable
D0_dx_native = hcat(D_dx_native, Z_native)
D0_id_native = hcat(Z_native, D_id_native)

println(io0(), "[DEBUG] D0_dx size: $(size(D0_dx_mpi)), D0_id size: $(size(D0_id_mpi))")

# Now build Hessian: sum over (j,k) of D[j]'*diag(w*y[j,k])*D[k]
# For the 2-operator case (dx, id):
# H = D_dx'*diag(w*y11)*D_dx + D_id'*diag(w*y22)*D_id + D_dx'*diag(w*y12)*D_id + D_id'*diag(w*y12)*D_dx

# Create test y values
y11 = ones(n) * 0.5
y12 = ones(n) * 0.1
y22 = ones(n) * 0.3

# MPI version
H_mpi = nothing
# j=1: dx'*diag(w*y11)*dx
foo = spdiagm(n, n, 0 => w_mpi .* HPCVector(y11))
H_mpi = D0_dx_mpi' * foo * D0_dx_mpi

# j=2: id'*diag(w*y22)*id
foo = spdiagm(n, n, 0 => w_mpi .* HPCVector(y22))
H_mpi = H_mpi + D0_id_mpi' * foo * D0_id_mpi

# Cross term: dx'*diag(w*y12)*id + id'*diag(w*y12)*dx
foo = spdiagm(n, n, 0 => w_mpi .* HPCVector(y12))
H_mpi = H_mpi + D0_dx_mpi' * foo * D0_id_mpi + D0_id_mpi' * foo * D0_dx_mpi

# Native version
foo_native = spdiagm(n, n, 0 => w_native .* y11)
H_native = D0_dx_native' * foo_native * D0_dx_native

foo_native = spdiagm(n, n, 0 => w_native .* y22)
H_native = H_native + D0_id_native' * foo_native * D0_id_native

foo_native = spdiagm(n, n, 0 => w_native .* y12)
H_native = H_native + D0_dx_native' * foo_native * D0_id_native + D0_id_native' * foo_native * D0_dx_native

H_mpi_native = SparseMatrixCSC(H_mpi)

println(io0(), "[DEBUG] Full H size: MPI=$(size(H_mpi)), native=$(size(H_native))")
if rank == 0
    println("[DEBUG] Full H nnz: MPI=$(nnz(H_mpi_native)), native=$(nnz(H_native))")
    diff = norm(H_mpi_native - H_native)
    println("[DEBUG] Difference: $diff")

    if diff > 1e-10
        println("[DEBUG] Non-zero differences:")
        for i in 1:min(20, size(H_native, 1))
            for j in 1:min(20, size(H_native, 2))
                mpi_val = H_mpi_native[i,j]
                native_val = H_native[i,j]
                if abs(mpi_val - native_val) > 1e-14
                    println("[DEBUG]   ($i, $j): MPI=$mpi_val, Native=$native_val, diff=$(mpi_val - native_val)")
                end
            end
        end
    end
    @test diff < 1e-12
end

# Final step: Apply block restriction
RHR_mpi = R_block_mpi' * H_mpi * R_block_mpi
RHR_native = R_block_native' * H_native * R_block_native

RHR_mpi_native = SparseMatrixCSC(RHR_mpi)

println(io0(), "[DEBUG] RHR size: MPI=$(size(RHR_mpi)), native=$(size(RHR_native))")
if rank == 0
    println("[DEBUG] RHR nnz: MPI=$(nnz(RHR_mpi_native)), native=$(nnz(RHR_native))")
    diff = norm(RHR_mpi_native - RHR_native)
    println("[DEBUG] RHR difference: $diff")

    if diff > 1e-10
        println("[DEBUG] RHR non-zero differences:")
        for i in 1:size(RHR_native, 1)
            for j in 1:size(RHR_native, 2)
                mpi_val = RHR_mpi_native[i,j]
                native_val = RHR_native[i,j]
                if abs(mpi_val - native_val) > 1e-14
                    println("[DEBUG]   ($i, $j): MPI=$mpi_val, Native=$native_val, diff=$(mpi_val - native_val)")
                end
            end
        end
    end
    @test diff < 1e-12
end

println(io0(), "[DEBUG] Test completed")
