using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

using MultiGridBarrierMPI
MultiGridBarrierMPI.Init()

using LinearAlgebraMPI
using LinearAlgebraMPI: VectorMPI, MatrixMPI, SparseMatrixMPI, io0
using LinearAlgebra
using SparseArrays
using MultiGridBarrier

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println(io0(), "[DEBUG] Testing AMG structure (nranks=$nranks)")

# Create geometry at L=2
g = fem1d_mpi(Float64; L=2)

println(io0(), "[DEBUG] Geometry created")
println(io0(), "[DEBUG] x size: $(size(g.x))")
println(io0(), "[DEBUG] w length: $(length(g.w))")

# Check all subspace levels
println(io0(), "[DEBUG] Dirichlet subspaces:")
if haskey(g.subspaces, :dirichlet)
    for (i, R) in enumerate(g.subspaces[:dirichlet])
        R_native = SparseMatrixCSC(R)
        println(io0(), "[DEBUG]   Level $i: R size $(size(R)), nnz $(nnz(R_native))")
        println(io0(), "[DEBUG]   Level $i: R row_partition $(R.row_partition)")
        println(io0(), "[DEBUG]   Level $i: R col_partition $(R.col_partition)")
    end
end

# Check refine matrices
println(io0(), "[DEBUG] Refine matrices:")
for (i, P) in enumerate(g.refine)
    P_native = SparseMatrixCSC(P)
    println(io0(), "[DEBUG]   Level $i: P size $(size(P)), nnz $(nnz(P_native))")
    println(io0(), "[DEBUG]   Level $i: P row_partition $(P.row_partition)")
    println(io0(), "[DEBUG]   Level $i: P col_partition $(P.col_partition)")
end

# Check coarsen matrices
println(io0(), "[DEBUG] Coarsen matrices:")
for (i, Pc) in enumerate(g.coarsen)
    Pc_native = SparseMatrixCSC(Pc)
    println(io0(), "[DEBUG]   Level $i: Pc size $(size(Pc)), nnz $(nnz(Pc_native))")
    println(io0(), "[DEBUG]   Level $i: Pc row_partition $(Pc.row_partition)")
    println(io0(), "[DEBUG]   Level $i: Pc col_partition $(Pc.col_partition)")
end

# Create AMG structure using MultiGridBarrier's internal function
println(io0(), "[DEBUG] \n--- Creating AMG structure ---")

# Check the D operators
D_op = g.operators[:dx]
ID_op = g.operators[:id]
println(io0(), "[DEBUG] dx size: $(size(D_op)), row_part $(D_op.row_partition), col_part $(D_op.col_partition)")
println(io0(), "[DEBUG] id size: $(size(ID_op)), row_part $(ID_op.row_partition), col_part $(ID_op.col_partition)")

# Test map_rows on coordinates
println(io0(), "[DEBUG] \n--- Testing map_rows on x (coordinates) ---")
x = g.x
println(io0(), "[DEBUG] x size: $(size(x))")
println(io0(), "[DEBUG] x row_partition: $(x.row_partition)")
println(io0(), "[DEBUG] x col_partition: $(x.col_partition)")

# Test creating a column matrix from x
result = MultiGridBarrier.map_rows((xi,) -> [xi[1]^2]', x)
println(io0(), "[DEBUG] map_rows result size: $(size(result))")
println(io0(), "[DEBUG] map_rows result row_partition: $(result.row_partition)")
println(io0(), "[DEBUG] map_rows result col_partition: $(result.col_partition)")

# Test on fine-level D operator
println(io0(), "[DEBUG] \n--- Testing D operations at finest level ---")
R_finest = g.subspaces[:dirichlet][end]
println(io0(), "[DEBUG] R_finest size: $(size(R_finest))")

# Create a test vector with size ncols(R_finest)
n_coarse = size(R_finest, 2)
z_test = VectorMPI(ones(n_coarse))
println(io0(), "[DEBUG] z_test partition: $(z_test.partition)")

# Compute R * z
Rz = R_finest * z_test
println(io0(), "[DEBUG] R*z partition: $(Rz.partition)")
println(io0(), "[DEBUG] R*z size: $(length(Rz))")

# Compute D * (R * z)
DRz = D_op * Rz
println(io0(), "[DEBUG] D*R*z partition: $(DRz.partition)")

# Now test the full system
println(io0(), "[DEBUG] \n--- Testing full Hessian assembly ---")

# The actual Hessian in amgb involves: R' * D' * diag(w * something) * D * R
# Let's simulate this
w = g.w
w_scaled = 2.0 .* Vector(w)  # Example scaling
W_diag = spdiagm(length(w), length(w), 0 => VectorMPI(w_scaled))

println(io0(), "[DEBUG] W_diag size: $(size(W_diag))")
println(io0(), "[DEBUG] W_diag row_partition: $(W_diag.row_partition)")
println(io0(), "[DEBUG] W_diag col_partition: $(W_diag.col_partition)")

# Full Hessian: R' * D' * W * D * R
D_R = D_op * R_finest
println(io0(), "[DEBUG] D*R size: $(size(D_R))")
println(io0(), "[DEBUG] D*R row_partition: $(D_R.row_partition)")
println(io0(), "[DEBUG] D*R col_partition: $(D_R.col_partition)")

W_D_R = W_diag * D_R
println(io0(), "[DEBUG] W*D*R size: $(size(W_D_R))")
println(io0(), "[DEBUG] W*D*R row_partition: $(W_D_R.row_partition)")
println(io0(), "[DEBUG] W*D*R col_partition: $(W_D_R.col_partition)")

Dt_W_D_R = D_op' * W_D_R
println(io0(), "[DEBUG] D'*W*D*R size: $(size(Dt_W_D_R))")
println(io0(), "[DEBUG] D'*W*D*R row_partition: $(Dt_W_D_R.row_partition)")
println(io0(), "[DEBUG] D'*W*D*R col_partition: $(Dt_W_D_R.col_partition)")

Rt_Dt_W_D_R = R_finest' * Dt_W_D_R
println(io0(), "[DEBUG] R'*D'*W*D*R size: $(size(Rt_Dt_W_D_R))")
println(io0(), "[DEBUG] R'*D'*W*D*R row_partition: $(Rt_Dt_W_D_R.row_partition)")
println(io0(), "[DEBUG] R'*D'*W*D*R col_partition: $(Rt_Dt_W_D_R.col_partition)")

# Convert to native and examine
H_native = SparseMatrixCSC(Rt_Dt_W_D_R)
if rank == 0
    println("[DEBUG] Hessian nnz: $(nnz(H_native))")
    println("[DEBUG] Hessian diagonal: $(diag(H_native))")

    H_dense = Matrix(H_native)
    if size(H_dense, 1) <= 20
        eigs = eigvals(Symmetric(H_dense))
        println("[DEBUG] Hessian eigenvalues: $eigs")
        println("[DEBUG] Min eigenvalue: $(minimum(eigs))")
    end
end

# Try factorization
println(io0(), "[DEBUG] Attempting to factorize...")
b_test = VectorMPI(ones(size(Rt_Dt_W_D_R, 1)))
try
    x_test = Rt_Dt_W_D_R \ b_test
    println(io0(), "[DEBUG] Factorization succeeded!")
catch e
    println(io0(), "[DEBUG] Factorization FAILED: $e")
end

println(io0(), "[DEBUG] AMG structure test completed")
