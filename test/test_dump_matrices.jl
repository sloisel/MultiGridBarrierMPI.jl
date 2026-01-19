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

println(io0(), "[DEBUG] Dumping matrices that trigger the bug (nranks=$nranks)")

# Create geometry - same as the failing test
g = fem1d_mpi(Float64; L=2)

# Get D operators
D_op = [g.operators[:dx], g.operators[:id]]

# Create test data
n = length(g.w)
y_vals = zeros(n, 4)
y_vals[:, 1] = ones(n) .* 0.5
y_vals[:, 2] = ones(n) .* 0.1
y_vals[:, 3] = ones(n) .* 0.1
y_vals[:, 4] = ones(n) .* 0.3

y_mpi = HPCMatrix(y_vals)
w_mpi = g.w

# Build up to the failing point
foo = MultiGridBarrier.amgb_diag(D_op[1], w_mpi .* y_mpi[:, 1])
bar = D_op[1]' * foo * D_op[1]
ret_mpi = bar

foo = MultiGridBarrier.amgb_diag(D_op[1], w_mpi .* y_mpi[:, 4])
bar = D_op[2]' * foo * D_op[2]
ret_mpi = ret_mpi + bar

# Now compute the cross term matrices - these are the ones that fail when added
foo = MultiGridBarrier.amgb_diag(D_op[1], w_mpi .* y_mpi[:, 3])

# These two matrices, when added, cause the BoundsError
M1 = D_op[2]' * foo * D_op[1]  # id' * foo * dx
M2 = D_op[1]' * foo * D_op[2]  # dx' * foo * id

# Dump the matrices
println(io0(), "\n=== MATRIX M1 (id' * foo * dx) ===")
println(io0(), "Global size: $(size(M1))")
println(io0(), "row_partition: $(M1.row_partition)")
println(io0(), "col_partition: $(M1.col_partition)")
println(io0(), "col_indices: $(M1.col_indices)")

# Dump local CSC data for each rank
println("Rank $rank M1 local:")
println("  A.parent.m (local ncols): $(M1.A.parent.m)")
println("  A.parent.n (local nrows): $(M1.A.parent.n)")
println("  A.parent.colptr: $(M1.A.parent.colptr)")
println("  A.parent.rowval: $(M1.A.parent.rowval)")
println("  A.parent.nzval: $(M1.A.parent.nzval)")
println("  nzval length: $(length(M1.A.parent.nzval))")

println(io0(), "\n=== MATRIX M2 (dx' * foo * id) ===")
println(io0(), "Global size: $(size(M2))")
println(io0(), "row_partition: $(M2.row_partition)")
println(io0(), "col_partition: $(M2.col_partition)")
println(io0(), "col_indices: $(M2.col_indices)")

println("Rank $rank M2 local:")
println("  A.parent.m (local ncols): $(M2.A.parent.m)")
println("  A.parent.n (local nrows): $(M2.A.parent.n)")
println("  A.parent.colptr: $(M2.A.parent.colptr)")
println("  A.parent.rowval: $(M2.A.parent.rowval)")
println("  A.parent.nzval: $(M2.A.parent.nzval)")
println("  nzval length: $(length(M2.A.parent.nzval))")

# Also dump the native versions for comparison
M1_native = SparseMatrixCSC(M1)
M2_native = SparseMatrixCSC(M2)

if rank == 0
    println("\n=== NATIVE M1 ===")
    println("size: $(size(M1_native))")
    println("nnz: $(nnz(M1_native))")
    println("colptr: $(M1_native.colptr)")
    println("rowval: $(M1_native.rowval)")
    println("nzval: $(M1_native.nzval)")

    println("\n=== NATIVE M2 ===")
    println("size: $(size(M2_native))")
    println("nnz: $(nnz(M2_native))")
    println("colptr: $(M2_native.colptr)")
    println("rowval: $(M2_native.rowval)")
    println("nzval: $(M2_native.nzval)")
end

# Now try the addition (this should fail without cache clearing)
println(io0(), "\n=== ATTEMPTING M1 + M2 ===")
try
    result = M1 + M2
    println(io0(), "SUCCESS!")
    result_native = SparseMatrixCSC(result)
    println(io0(), "Result nnz: $(nnz(result_native))")
catch e
    println(io0(), "FAILED: $e")
    # Print more info about the error
    for (ex, bt) in current_exceptions()
        showerror(stdout, ex, bt)
        println()
    end
end

println(io0(), "\n[DEBUG] Dump completed")
