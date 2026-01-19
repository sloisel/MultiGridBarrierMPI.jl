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

println(io0(), "[DEBUG] Testing subspace structures")

# Create geometry
g = fem1d_hpc(Float64; L=3)

println(io0(), "[DEBUG] Geometry created")
println(io0(), "[DEBUG] x size: $(size(g.x))")
println(io0(), "[DEBUG] w length: $(length(g.w))")
println(io0(), "[DEBUG] subspaces keys: $(keys(g.subspaces))")
println(io0(), "[DEBUG] refine length: $(length(g.refine))")
println(io0(), "[DEBUG] coarsen length: $(length(g.coarsen))")

# Check subspace matrices (restriction matrices)
# All operations must be run by all ranks (collective)
if haskey(g.subspaces, :dirichlet)
    dirichlet_subspaces = g.subspaces[:dirichlet]
    println(io0(), "[DEBUG] Dirichlet subspaces: $(length(dirichlet_subspaces))")
    for i in 1:length(dirichlet_subspaces)
        R = dirichlet_subspaces[i]
        R_native = SparseMatrixCSC(R)  # Collective operation
        println(io0(), "[DEBUG]   Level $i: R size $(size(R)), nnz $(nnz(R_native))")
        println(io0(), "[DEBUG]   Level $i: R row_partition $(R.row_partition)")
        println(io0(), "[DEBUG]   Level $i: R col_partition $(R.col_partition)")
    end
end

# Check refine matrices
println(io0(), "[DEBUG] Refine matrices:")
for i in 1:length(g.refine)
    P = g.refine[i]
    P_native = SparseMatrixCSC(P)  # Collective operation
    println(io0(), "[DEBUG]   Level $i: P size $(size(P)), nnz $(nnz(P_native))")
end

# Check coarsen matrices
println(io0(), "[DEBUG] Coarsen matrices:")
for i in 1:length(g.coarsen)
    Pc = g.coarsen[i]
    Pc_native = SparseMatrixCSC(Pc)  # Collective operation
    println(io0(), "[DEBUG]   Level $i: Pc size $(size(Pc)), nnz $(nnz(Pc_native))")
end

# The key test: At coarsest level, try to build and solve a small system
if haskey(g.subspaces, :dirichlet) && length(g.subspaces[:dirichlet]) > 0
    R = g.subspaces[:dirichlet][1]  # Coarsest level
    D = g.operators[:dx]
    w = g.w

    println(io0(), "[DEBUG] Building coarse level system: R'*D'*diag(w)*D*R")

    # Build the Hessian-like matrix
    try
        n_w = length(w)
        w_diag = spdiagm(n_w, n_w, 0 => w)  # Collective
        DwD = D' * w_diag * D

        println(io0(), "[DEBUG] D'*diag(w)*D size: $(size(DwD))")

        RtDwDR = R' * DwD * R

        println(io0(), "[DEBUG] R'*D'*diag(w)*D*R size: $(size(RtDwDR))")
        println(io0(), "[DEBUG] R'*D'*diag(w)*D*R row_partition: $(RtDwDR.row_partition)")

        # Convert to native and check (collective)
        RtDwDR_native = SparseMatrixCSC(RtDwDR)

        println(io0(), "[DEBUG] R'*D'*diag(w)*D*R nnz: $(nnz(RtDwDR_native))")

        # Only do local analysis on rank 0 after gathering
        if rank == 0
            println("[DEBUG] R'*D'*diag(w)*D*R diagonal: $(diag(RtDwDR_native))")

            # Check eigenvalues
            RtDwDR_dense = Matrix(RtDwDR_native)
            if size(RtDwDR_dense, 1) <= 20
                eigs = eigvals(Symmetric(RtDwDR_dense))
                println("[DEBUG] Eigenvalues: $eigs")
            end
        end

        # Try to solve (collective)
        n_coarse = size(RtDwDR, 1)
        b_coarse = HPCVector(ones(n_coarse))

        println(io0(), "[DEBUG] Attempting to solve coarse system...")

        x_coarse = RtDwDR \ b_coarse

        println(io0(), "[DEBUG] Coarse solve succeeded!")
        x_native = Vector(x_coarse)  # Collective
        println(io0(), "[DEBUG] Solution: $x_native")

    catch e
        println(io0(), "[DEBUG] ERROR: $e")
        if rank == 0
            for (ex, bt) in current_exceptions()
                showerror(stdout, ex, bt)
                println()
            end
        end
    end
end

println(io0(), "[DEBUG] Test completed")
