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
    println("[DEBUG] Testing solve with detailed diagnostics")
    flush(stdout)
end

# Override solve to add diagnostics
original_solve = MultiGridBarrier.solve
function debug_solve(A::SparseMatrixMPI{T}, b::VectorMPI{T}) where T
    if rank == 0
        println("[DEBUG-SOLVE] Matrix size: $(size(A))")
        println("[DEBUG-SOLVE] Vector length: $(length(b))")
        flush(stdout)
    end

    # Gather matrix to rank 0 for inspection
    A_native = SparseMatrixCSC(A)
    b_native = Vector(b)

    if rank == 0
        println("[DEBUG-SOLVE] Matrix nnz: $(nnz(A_native))")

        # Check for zero rows
        zero_rows = Int[]
        for i in 1:size(A_native, 1)
            row_nnz = A_native[i, :] |> nnz
            if row_nnz == 0
                push!(zero_rows, i)
            end
        end
        if !isempty(zero_rows)
            println("[DEBUG-SOLVE] WARNING: Zero rows found: $zero_rows")
        else
            println("[DEBUG-SOLVE] No zero rows found")
        end

        # Check for zero columns
        zero_cols = Int[]
        for j in 1:size(A_native, 2)
            col_nnz = A_native[:, j] |> nnz
            if col_nnz == 0
                push!(zero_cols, j)
            end
        end
        if !isempty(zero_cols)
            println("[DEBUG-SOLVE] WARNING: Zero columns found: $zero_cols")
        else
            println("[DEBUG-SOLVE] No zero columns found")
        end

        # Check diagonal
        diag_vals = diag(A_native)
        zero_diag = findall(iszero, diag_vals)
        if !isempty(zero_diag)
            println("[DEBUG-SOLVE] WARNING: Zero diagonal entries at: $zero_diag")
        else
            println("[DEBUG-SOLVE] No zero diagonal entries")
        end

        # Check symmetry
        diff_sym = norm(A_native - A_native', Inf)
        println("[DEBUG-SOLVE] Symmetry check (||A - A'||_inf): $diff_sym")

        # Try local solve on rank 0
        println("[DEBUG-SOLVE] Attempting local solve...")
        try
            x_local = A_native \ b_native
            println("[DEBUG-SOLVE] Local solve succeeded!")
            residual = norm(A_native * x_local - b_native)
            println("[DEBUG-SOLVE] Local residual: $residual")
        catch e
            println("[DEBUG-SOLVE] Local solve failed: $e")
        end

        flush(stdout)
    end

    # Now try the distributed solve
    if rank == 0
        println("[DEBUG-SOLVE] Attempting distributed solve...")
        flush(stdout)
    end

    result = A \ b

    if rank == 0
        println("[DEBUG-SOLVE] Distributed solve succeeded!")
        flush(stdout)
    end

    return result
end

# Monkey-patch (not ideal but for debugging)
# Instead, let's just run amgb and catch the error

if rank == 0
    println("[DEBUG] Creating geometry...")
    flush(stdout)
end

g = fem1d_mpi(Float64; L=3)

if rank == 0
    println("[DEBUG] Geometry created, x size: $(size(g.x))")
    flush(stdout)
end

# Try with verbose to see more
if rank == 0
    println("[DEBUG] Starting solve with verbose=true...")
    flush(stdout)
end

try
    sol = fem1d_mpi_solve(Float64; L=3, p=1.0, verbose=true)
    if rank == 0
        println("[DEBUG] Solve succeeded!")
    end
catch e
    if rank == 0
        println("[DEBUG] Solve failed: $e")
        println(stacktrace(catch_backtrace())[1:min(10, end)])
    end
end

if rank == 0
    println("[DEBUG] Test completed")
    flush(stdout)
end
