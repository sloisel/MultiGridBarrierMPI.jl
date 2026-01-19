using Test
using MPI

# Initialize MPI first
if !MPI.Initialized()
    MPI.Init()
end

using HPCMultiGridBarrier
HPCMultiGridBarrier.Init()

using HPCSparseArrays
using HPCSparseArrays: HPCVector, HPCMatrix, HPCSparseMatrix, io0
using LinearAlgebra
using SparseArrays
using MultiGridBarrier

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Testing geometry and Hessian construction")
    flush(stdout)
end

# Create geometry (same as test_quick.jl)
g = fem1d_hpc(Float64; L=3)

if rank == 0
    println("[DEBUG] Geometry created")
    println("[DEBUG] x size: $(size(g.x))")
    println("[DEBUG] w length: $(length(g.w))")
    println("[DEBUG] operators keys: $(keys(g.operators))")
    flush(stdout)
end

# Get the D operator (differentiation)
D = g.operators[:dx]

if rank == 0
    println("[DEBUG] D size: $(size(D))")
    flush(stdout)
end

# Test D' * D (this is part of the Hessian construction)
DtD = D' * D
DtD_native = SparseMatrixCSC(DtD)

if rank == 0
    println("[DEBUG] D'D size: $(size(DtD))")
    println("[DEBUG] D'D nnz: $(nnz(DtD_native))")
    flush(stdout)
end

# Try to factorize D'D (should work for a well-posed problem)
w_native = Vector(g.w)
w_diag = spdiagm(0 => w_native)
WD = HPCSparseMatrix{Float64}(w_diag) * D

if rank == 0
    println("[DEBUG] W*D size: $(size(WD))")
    flush(stdout)
end

DtWD = D' * HPCSparseMatrix{Float64}(w_diag) * D
DtWD_native = SparseMatrixCSC(DtWD)

if rank == 0
    println("[DEBUG] D'WD size: $(size(DtWD))")
    println("[DEBUG] D'WD nnz: $(nnz(DtWD_native))")

    # Check if it's positive definite by looking at eigenvalues
    DtWD_dense = Matrix(DtWD_native)
    if size(DtWD_dense, 1) <= 20
        eigs = eigvals(Symmetric(DtWD_dense))
        println("[DEBUG] D'WD eigenvalues: $eigs")
        println("[DEBUG] Positive definite: $(all(eigs .> 0))")
    end
    flush(stdout)
end

# Check the R_coarse matrix from the multigrid hierarchy
if haskey(g.subspaces, :dirichlet) && length(g.subspaces[:dirichlet]) > 0
    R = g.subspaces[:dirichlet][1]  # Coarsest level restriction
    if rank == 0
        println("[DEBUG] R size: $(size(R))")
        println("[DEBUG] R nnz: $(nnz(SparseMatrixCSC(R)))")
    end

    # Test R' * D' * W * D * R (the form that appears in the Hessian)
    RtDtWDR = R' * DtWD * R
    RtDtWDR_native = SparseMatrixCSC(RtDtWDR)

    if rank == 0
        println("[DEBUG] R'D'WDR size: $(size(RtDtWDR))")
        println("[DEBUG] R'D'WDR nnz: $(nnz(RtDtWDR_native))")

        # Check if it's positive definite
        RtDtWDR_dense = Matrix(RtDtWDR_native)
        if size(RtDtWDR_dense, 1) <= 20
            eigs = eigvals(Symmetric(RtDtWDR_dense))
            println("[DEBUG] R'D'WDR eigenvalues: $eigs")
            println("[DEBUG] Positive definite: $(all(eigs .> 0))")
        end
        flush(stdout)
    end

    # Try to solve with this matrix
    n_small = size(RtDtWDR, 1)
    b_mpi = HPCVector(ones(n_small))

    if rank == 0
        println("[DEBUG] Attempting to solve R'D'WDR \\ b...")
        flush(stdout)
    end

    try
        x_hpc = RtDtWDR \ b_mpi
        x_native = Vector(x_hpc)
        if rank == 0
            println("[DEBUG] Solve succeeded!")
            println("[DEBUG] Solution: $x_native")
        end
    catch e
        if rank == 0
            println("[DEBUG] Solve failed: $e")
        end
    end
end

if rank == 0
    println("[DEBUG] Hessian test completed")
    flush(stdout)
end
