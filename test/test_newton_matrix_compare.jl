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

println(io0(), "[DEBUG] Newton matrix comparison test (nranks=$nranks)")

# Storage for captured matrices
mpi_matrices = Dict{Int, Tuple{SparseMatrixCSC{Float64,Int}, Vector{Float64}}}()
native_matrices = Dict{Int, Tuple{SparseMatrixCSC{Float64,Int}, Vector{Float64}}}()

# Counter for solve calls
mpi_solve_count = Ref(0)
native_solve_count = Ref(0)

# Override solve for MPI version
function instrumented_mpi_solve(A::HPCSparseMatrix{T}, b::HPCVector{T}) where {T}
    mpi_solve_count[] += 1
    A_native = SparseMatrixCSC(A)
    b_native = Vector(b)
    mpi_matrices[mpi_solve_count[]] = (A_native, b_native)
    return A \ b
end

# Override solve for native version (only on rank 0)
original_native_solve = MultiGridBarrier.solve

function instrumented_native_solve(A::SparseMatrixCSC{T,Int}, b::Vector{T}) where {T}
    native_solve_count[] += 1
    native_matrices[native_solve_count[]] = (A, b)
    return original_native_solve(A, b)
end

# Override solve
MultiGridBarrier.solve(A::HPCSparseMatrix{T}, b::HPCVector{T}) where {T} = instrumented_mpi_solve(A, b)

println(io0(), "[DEBUG] Running MPI solve...")

# Run MPI solver and capture matrices
try
    sol_mpi = fem1d_mpi_solve(Float64; L=2, p=1.0, verbose=false)
    println(io0(), "[DEBUG] MPI solve succeeded with $(mpi_solve_count[]) iterations")
catch e
    println(io0(), "[DEBUG] MPI solve failed after $(mpi_solve_count[]) iterations: $e")
end

# Now run native solve on rank 0 to compare
if rank == 0
    # Override native solve
    MultiGridBarrier.solve(A::SparseMatrixCSC{T,Int}, b::Vector{T}) where {T} = instrumented_native_solve(A, b)

    println("[DEBUG] Running native solve...")
    try
        sol_native = fem1d_solve(Float64; L=2, p=1.0, verbose=false)
        println("[DEBUG] Native solve succeeded with $(native_solve_count[]) iterations")
    catch e
        println("[DEBUG] Native solve failed after $(native_solve_count[]) iterations: $e")
    end

    # Compare first matrices
    println("\n[DEBUG] === Matrix Comparison ===")

    min_count = min(mpi_solve_count[], native_solve_count[])
    if min_count > 0
        for i in 1:min(3, min_count)  # Compare first 3 iterations
            println("\n[DEBUG] Iteration $i:")

            if haskey(mpi_matrices, i) && haskey(native_matrices, i)
                A_mpi, b_mpi = mpi_matrices[i]
                A_native, b_native = native_matrices[i]

                println("[DEBUG] MPI matrix size: $(size(A_mpi)), nnz: $(nnz(A_mpi))")
                println("[DEBUG] Native matrix size: $(size(A_native)), nnz: $(nnz(A_native))")

                if size(A_mpi) == size(A_native)
                    diff = norm(A_mpi - A_native)
                    println("[DEBUG] Matrix difference (Frobenius): $diff")

                    # Check diagonal
                    diag_mpi = diag(A_mpi)
                    diag_native = diag(A_native)
                    diag_diff = norm(diag_mpi - diag_native)
                    println("[DEBUG] Diagonal difference: $diag_diff")

                    # Show where differences are
                    if diff > 1e-10
                        println("[DEBUG] Non-zero differences:")
                        D_diff = A_mpi - A_native
                        I, J, V = findnz(D_diff)
                        for idx in 1:min(10, length(V))
                            if abs(V[idx]) > 1e-14
                                println("[DEBUG]   ($(I[idx]), $(J[idx])): MPI=$(A_mpi[I[idx],J[idx]]), Native=$(A_native[I[idx],J[idx]]), diff=$(V[idx])")
                            end
                        end
                    end

                    # Check eigenvalues
                    eigs_mpi = eigvals(Symmetric(Matrix(A_mpi)))
                    eigs_native = eigvals(Symmetric(Matrix(A_native)))
                    println("[DEBUG] MPI eigenvalues: $eigs_mpi")
                    println("[DEBUG] Native eigenvalues: $eigs_native")
                else
                    println("[DEBUG] Size mismatch - cannot compare directly")
                end
            else
                println("[DEBUG] Missing matrix data for iteration $i")
            end
        end
    end
end

println(io0(), "[DEBUG] Test completed")
