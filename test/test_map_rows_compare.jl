using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

using HPCMultiGridBarrier
HPCMultiGridBarrier.Init()

using HPCSparseArrays
using HPCSparseArrays: HPCVector, HPCMatrix, io0
using LinearAlgebra
using SparseArrays
using MultiGridBarrier

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println(io0(), "[DEBUG] map_rows comparison test (nranks=$nranks)")

# Create geometries
g_hpc = fem1d_hpc(Float64; L=2)
g_native = fem1d(Float64; L=2)

println(io0(), "[DEBUG] MPI geometry x size: $(size(g_hpc.x))")
println(io0(), "[DEBUG] Native geometry x size: $(size(g_native.x))")

# Check coordinates match
x_hpc_native = Matrix(g_hpc.x)
x_diff = norm(x_hpc_native - g_native.x)
println(io0(), "[DEBUG] x difference: $x_diff")

# Check weights match
w_hpc_native = Vector(g_hpc.w)
w_diff = norm(w_hpc_native - g_native.w)
println(io0(), "[DEBUG] w difference: $w_diff")

# Get operators
D_mpi = [g_hpc.operators[:dx], g_hpc.operators[:id]]
D_native = [g_native.operators[:dx], g_native.operators[:id]]

# Create a test solution
n = length(g_hpc.w)
z_native = sin.(range(0, π, length=n))
z_hpc = HPCVector(z_native)

# Compute Dz
Dz_hpc = hcat([D * z_hpc for D in D_mpi]...)
Dz_native = hcat([D * z_native for D in D_native]...)

println(io0(), "[DEBUG] Dz_hpc size: $(size(Dz_hpc))")
println(io0(), "[DEBUG] Dz_native size: $(size(Dz_native))")

Dz_hpc_native = Matrix(Dz_hpc)
Dz_diff = norm(Dz_hpc_native - Dz_native)
println(io0(), "[DEBUG] Dz difference: $Dz_diff")

# Create test function (simplified version of F2)
function F2_test(x, q)
    # Returns a simple Hessian-like matrix
    du = q[1]  # du/dx
    u = q[2]   # u
    # 2x2 Hessian for 2 operators
    return [2*du*du   du*u;
            du*u      2*u*u]
end

# Apply map_rows
println(io0(), "[DEBUG] Testing map_rows...")
y_mpi = MultiGridBarrier.map_rows((xi, qi) -> F2_test(xi, qi)[:]', g_hpc.x, Dz_hpc)
y_native = MultiGridBarrier.map_rows((xi, qi) -> F2_test(xi, qi)[:]', g_native.x, Dz_native)

println(io0(), "[DEBUG] y_mpi size: $(size(y_mpi))")
println(io0(), "[DEBUG] y_native size: $(size(y_native))")

y_mpi_native = Matrix(y_mpi)
y_diff = norm(y_mpi_native - y_native)
println(io0(), "[DEBUG] y difference: $y_diff")

if rank == 0
    println("[DEBUG] y_mpi:")
    display(y_mpi_native)
    println()
    println("[DEBUG] y_native:")
    display(y_native)
    println()
    if y_diff > 1e-10
        println("[DEBUG] Differences:")
        for i in 1:size(y_native, 1)
            for j in 1:size(y_native, 2)
                if abs(y_mpi_native[i,j] - y_native[i,j]) > 1e-14
                    println("[DEBUG]   ($i, $j): MPI=$(y_mpi_native[i,j]), Native=$(y_native[i,j])")
                end
            end
        end
    end
end

# Now build the Hessian using the same method as f2
println(io0(), "[DEBUG] Building Hessian from map_rows output...")

w_hpc = g_hpc.w
w_native = g_native.w
n_ops = length(D_mpi)

# MPI version
local H_mpi
for j in 1:n_ops
    foo = MultiGridBarrier.amgb_diag(D_mpi[1], w_hpc .* y_mpi[:, (j-1)*n_ops + j])
    bar = D_mpi[j]' * foo * D_mpi[j]
    if j > 1
        global H_mpi = H_mpi + bar
    else
        global H_mpi = bar
    end
    for k in 1:j-1
        foo = MultiGridBarrier.amgb_diag(D_mpi[1], w_hpc .* y_mpi[:, (j-1)*n_ops + k])
        global H_mpi = H_mpi + D_mpi[j]' * foo * D_mpi[k] + D_mpi[k]' * foo * D_mpi[j]
    end
end

# Native version (on rank 0)
local H_native
for j in 1:n_ops
    foo = spdiagm(n, n, 0 => w_native .* y_native[:, (j-1)*n_ops + j])
    bar = D_native[j]' * foo * D_native[j]
    if j > 1
        global H_native = H_native + bar
    else
        global H_native = bar
    end
    for k in 1:j-1
        foo = spdiagm(n, n, 0 => w_native .* y_native[:, (j-1)*n_ops + k])
        global H_native = H_native + D_native[j]' * foo * D_native[k] + D_native[k]' * foo * D_native[j]
    end
end

H_mpi_native = SparseMatrixCSC(H_mpi)

println(io0(), "[DEBUG] H_mpi size: $(size(H_mpi)), nnz: $(nnz(H_mpi_native))")

if rank == 0
    println("[DEBUG] H_native size: $(size(H_native)), nnz: $(nnz(H_native))")
    H_diff = norm(H_mpi_native - H_native)
    println("[DEBUG] H difference: $H_diff")

    if H_diff > 1e-10
        println("[DEBUG] H differences:")
        for i in 1:size(H_native, 1)
            for j in 1:size(H_native, 2)
                mpi_val = i <= size(H_mpi_native, 1) && j <= size(H_mpi_native, 2) ? H_mpi_native[i,j] : 0.0
                native_val = H_native[i,j]
                if abs(mpi_val - native_val) > 1e-14
                    println("[DEBUG]   ($i, $j): MPI=$mpi_val, Native=$native_val")
                end
            end
        end
    end
end

# Now apply restriction
R_mpi = g_hpc.subspaces[:dirichlet][end]
R_native = g_native.subspaces[:dirichlet][end]

println(io0(), "[DEBUG] R_mpi size: $(size(R_mpi))")

RHR_mpi = R_mpi' * H_mpi * R_mpi
RHR_mpi_native = SparseMatrixCSC(RHR_mpi)

if rank == 0
    RHR_native = R_native' * H_native * R_native
    println("[DEBUG] RHR_mpi size: $(size(RHR_mpi)), nnz: $(nnz(RHR_mpi_native))")
    println("[DEBUG] RHR_native size: $(size(RHR_native)), nnz: $(nnz(RHR_native))")
    RHR_diff = norm(RHR_mpi_native - RHR_native)
    println("[DEBUG] RHR difference: $RHR_diff")
end

println(io0(), "[DEBUG] Test completed")
