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

println(io0(), "[DEBUG] Testing column extraction and element-wise ops (nranks=$nranks)")

# Create geometry
g = fem1d_mpi(Float64; L=2)

# Get a sample matrix from map_rows
D_op = [g.operators[:dx], g.operators[:id]]
n = length(g.w)
z_native = sin.(range(0, π, length=n))
z_mpi = HPCVector(z_native)

# Apply D operators to get matrix
Dz_mpi = hcat([D * z_mpi for D in D_op]...)

println(io0(), "[DEBUG] Dz_mpi size: $(size(Dz_mpi))")
println(io0(), "[DEBUG] Dz_mpi row_partition: $(Dz_mpi.row_partition)")
println(io0(), "[DEBUG] Dz_mpi col_partition: $(Dz_mpi.col_partition)")

# Now use map_rows to create a matrix like F2 does
x = g.x
y_mpi = MultiGridBarrier.map_rows((xi, qi) -> [qi[1]^2, qi[2]^2, qi[1]*qi[2], qi[1]*qi[2]]', x, Dz_mpi)

println(io0(), "[DEBUG] y_mpi (from map_rows) size: $(size(y_mpi))")
println(io0(), "[DEBUG] y_mpi row_partition: $(y_mpi.row_partition)")
println(io0(), "[DEBUG] y_mpi col_partition: $(y_mpi.col_partition)")

# Test column extraction
println(io0(), "[DEBUG] Testing column extraction y[:, 1]...")
col1 = y_mpi[:, 1]
println(io0(), "[DEBUG] col1 type: $(typeof(col1))")
println(io0(), "[DEBUG] col1 size: $(size(col1))")
if col1 isa HPCVector
    println(io0(), "[DEBUG] col1 partition: $(col1.partition)")
else
    println(io0(), "[DEBUG] col1 is NOT a HPCVector!")
end

# Test element-wise multiplication
println(io0(), "[DEBUG] Testing w .* col...")
w = g.w
println(io0(), "[DEBUG] w partition: $(w.partition)")

try
    w_col = w .* col1
    println(io0(), "[DEBUG] w .* col1 type: $(typeof(w_col))")
    if w_col isa HPCVector
        println(io0(), "[DEBUG] w .* col1 partition: $(w_col.partition)")
    end
    println(io0(), "[DEBUG] Element-wise multiplication succeeded!")
catch e
    println(io0(), "[DEBUG] Element-wise multiplication FAILED: $e")
end

# Test with amgb_diag
println(io0(), "[DEBUG] Testing amgb_diag...")
D = g.operators[:dx]
try
    # This is what f2 does
    foo = MultiGridBarrier.amgb_diag(D, w .* col1)
    println(io0(), "[DEBUG] amgb_diag result type: $(typeof(foo))")
    println(io0(), "[DEBUG] amgb_diag result size: $(size(foo))")
    if foo isa HPCSparseMatrix
        println(io0(), "[DEBUG] amgb_diag row_partition: $(foo.row_partition)")
        println(io0(), "[DEBUG] amgb_diag col_partition: $(foo.col_partition)")
    end
    println(io0(), "[DEBUG] amgb_diag succeeded!")
catch e
    println(io0(), "[DEBUG] amgb_diag FAILED: $e")
    for (ex, bt) in current_exceptions()
        showerror(stdout, ex, bt)
        println()
    end
end

# Test the full chain: D' * foo * D
println(io0(), "[DEBUG] Testing D' * amgb_diag(D, w.*col) * D...")
try
    foo = MultiGridBarrier.amgb_diag(D, w .* col1)
    bar = D' * foo * D
    println(io0(), "[DEBUG] D'*foo*D size: $(size(bar))")
    if bar isa HPCSparseMatrix
        println(io0(), "[DEBUG] D'*foo*D row_partition: $(bar.row_partition)")
        println(io0(), "[DEBUG] D'*foo*D col_partition: $(bar.col_partition)")
    end
    println(io0(), "[DEBUG] Full chain succeeded!")

    # Convert to native and check
    bar_native = SparseMatrixCSC(bar)
    if rank == 0
        println("[DEBUG] D'*foo*D nnz: $(nnz(bar_native))")
    end
catch e
    println(io0(), "[DEBUG] Full chain FAILED: $e")
    for (ex, bt) in current_exceptions()
        showerror(stdout, ex, bt)
        println()
    end
end

# Compare with native
println(io0(), "[DEBUG] \n--- Comparing with native ---")
g_native = fem1d(Float64; L=2)
D_native = [g_native.operators[:dx], g_native.operators[:id]]
Dz_native = hcat([D * z_native for D in D_native]...)
x_native = g_native.x
y_native = MultiGridBarrier.map_rows((xi, qi) -> [qi[1]^2, qi[2]^2, qi[1]*qi[2], qi[1]*qi[2]]', x_native, Dz_native)

if rank == 0
    # Check the matrices match
    y_mpi_native = Matrix(y_mpi)
    println("[DEBUG] y difference (MPI vs native): $(norm(y_mpi_native - y_native))")

    col1_native = y_native[:, 1]
    col1_mpi_native = col1 isa HPCVector ? Vector(col1) : col1
    println("[DEBUG] col1 difference: $(norm(col1_mpi_native - col1_native))")
end

println(io0(), "[DEBUG] Column extraction test completed")
