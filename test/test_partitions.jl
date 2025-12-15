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

if rank == 0
    println("[DEBUG] Testing partition consistency")
    flush(stdout)
end

# Create geometry
g = fem1d_mpi(Float64; L=3)

# Get operators
D = g.operators[:dx]
id = g.operators[:id]
w = g.w

if rank == 0
    println("[DEBUG] D row_partition: $(D.row_partition)")
    println("[DEBUG] D col_partition: $(D.col_partition)")
    println("[DEBUG] id row_partition: $(id.row_partition)")
    println("[DEBUG] w partition: $(w.partition)")
    println("[DEBUG] D size: $(size(D))")
    println("[DEBUG] id size: $(size(id))")
    flush(stdout)
end

# Create diagonal matrix from w
n = length(w)
diag_from_w = spdiagm(n, n, 0 => w)

if rank == 0
    println("[DEBUG] diag_from_w row_partition: $(diag_from_w.row_partition)")
    println("[DEBUG] diag_from_w col_partition: $(diag_from_w.col_partition)")
    println("[DEBUG] diag_from_w size: $(size(diag_from_w))")
    flush(stdout)
end

# Check if partitions match
if rank == 0
    println("[DEBUG] D.row_partition == diag_from_w.row_partition? $(D.row_partition == diag_from_w.row_partition)")
    println("[DEBUG] D.col_partition == diag_from_w.col_partition? $(D.col_partition == diag_from_w.col_partition)")
    flush(stdout)
end

# Try computing D' * diag(w) * D
if rank == 0
    println("[DEBUG] Computing D' * diag(w) * D...")
    flush(stdout)
end

try
    # First do diag(w) * D
    wD = diag_from_w * D

    if rank == 0
        println("[DEBUG] wD computed successfully")
        println("[DEBUG] wD size: $(size(wD))")
        println("[DEBUG] wD row_partition: $(wD.row_partition)")
        flush(stdout)
    end

    # Then D' * wD
    DtwD = D' * wD

    if rank == 0
        println("[DEBUG] D'*w*D computed successfully")
        println("[DEBUG] DtwD size: $(size(DtwD))")
        println("[DEBUG] DtwD row_partition: $(DtwD.row_partition)")
        flush(stdout)
    end

    # Convert to native and compare with expected
    DtwD_native = SparseMatrixCSC(DtwD)
    D_native = SparseMatrixCSC(D)
    w_native = Vector(w)
    DtwD_expected = D_native' * Diagonal(w_native) * D_native

    if rank == 0
        println("[DEBUG] Native comparison:")
        println("[DEBUG] Expected DtwD diag: $(diag(DtwD_expected))")
        println("[DEBUG] Got DtwD diag: $(diag(DtwD_native))")
        println("[DEBUG] Difference: $(norm(DtwD_native - DtwD_expected))")
        flush(stdout)
    end

    @test norm(DtwD_native - DtwD_expected) < 1e-10

catch e
    if rank == 0
        println("[DEBUG] ERROR: $e")
        for (ex, bt) in current_exceptions()
            showerror(stdout, ex, bt)
            println()
        end
        flush(stdout)
    end
end

if rank == 0
    println("[DEBUG] Test completed")
    flush(stdout)
end
