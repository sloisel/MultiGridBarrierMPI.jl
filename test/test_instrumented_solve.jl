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

println(io0(), "[DEBUG] Instrumented solve test (nranks=$nranks)")

# Monkey-patch the solve function to capture the matrix
original_solve = MultiGridBarrier.solve
solve_count = Ref(0)

function instrumented_solve(A::SparseMatrixMPI{T}, b::VectorMPI{T}) where {T}
    solve_count[] += 1

    println(io0(), "[SOLVE #$(solve_count[])] Matrix size: $(size(A))")
    println(io0(), "[SOLVE #$(solve_count[])] Matrix row_partition: $(A.row_partition)")
    println(io0(), "[SOLVE #$(solve_count[])] Matrix col_partition: $(A.col_partition)")
    println(io0(), "[SOLVE #$(solve_count[])] RHS partition: $(b.partition)")

    # Gather the matrix to check properties
    A_native = SparseMatrixCSC(A)
    b_native = Vector(b)

    if rank == 0
        println("[SOLVE #$(solve_count[])] Matrix nnz: $(nnz(A_native))")
        println("[SOLVE #$(solve_count[])] Matrix diagonal min: $(minimum(abs.(diag(A_native))))")
        println("[SOLVE #$(solve_count[])] Matrix diagonal max: $(maximum(abs.(diag(A_native))))")

        # Check for zero rows
        n = size(A_native, 1)
        zero_rows = Int[]
        for i in 1:n
            row_start = A_native.colptr[1]
            row_norm = 0.0
            for j in 1:n
                row_norm += abs(A_native[i, j])^2
            end
            if sqrt(row_norm) < 1e-14
                push!(zero_rows, i)
            end
        end
        if !isempty(zero_rows)
            println("[SOLVE #$(solve_count[])] WARNING: Zero rows detected at: $zero_rows")
        end

        # Check eigenvalues for small matrices
        if n <= 30
            A_dense = Matrix(A_native)
            try
                eigs = eigvals(Symmetric(A_dense))
                println("[SOLVE #$(solve_count[])] Eigenvalues: $eigs")
                println("[SOLVE #$(solve_count[])] Min eigenvalue: $(minimum(eigs))")
                if minimum(eigs) < 1e-10
                    println("[SOLVE #$(solve_count[])] WARNING: Near-singular eigenvalue detected!")
                end
            catch e
                println("[SOLVE #$(solve_count[])] Could not compute eigenvalues: $e")
            end
        end
    end

    # Call original
    try
        result = original_solve(A, b)
        println(io0(), "[SOLVE #$(solve_count[])] Success!")
        return result
    catch e
        println(io0(), "[SOLVE #$(solve_count[])] FAILED: $e")

        # Save the matrix for inspection
        if rank == 0
            println("[SOLVE #$(solve_count[])] Full matrix dump:")
            A_dense = Matrix(A_native)
            display(A_dense)
            println()
        end

        rethrow()
    end
end

# Override solve
MultiGridBarrier.solve(A::SparseMatrixMPI{T}, b::VectorMPI{T}) where {T} = instrumented_solve(A, b)

# Now run the actual solve
println(io0(), "[DEBUG] Starting fem1d_mpi_solve...")

try
    sol = fem1d_mpi_solve(Float64; L=2, p=1.0, verbose=false)
    println(io0(), "[DEBUG] Solve succeeded!")
    sol_native = mpi_to_native(sol)
    println(io0(), "[DEBUG] Solution z norm: $(norm(sol_native.z))")
catch e
    println(io0(), "[DEBUG] Solve FAILED: $e")
end

println(io0(), "[DEBUG] Total solve calls: $(solve_count[])")
println(io0(), "[DEBUG] Test completed")
