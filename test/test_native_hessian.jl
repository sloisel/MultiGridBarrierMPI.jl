using LinearAlgebra
using SparseArrays
using MultiGridBarrier

println("[DEBUG] Testing native Hessian eigenvalues")

# Monkey-patch the solve function to capture the matrix
original_solve = MultiGridBarrier.solve
solve_count = Ref(0)

function instrumented_solve(A::SparseMatrixCSC{T}, b::Vector{T}) where {T}
    solve_count[] += 1

    println("[SOLVE #$(solve_count[])] Matrix size: $(size(A))")
    println("[SOLVE #$(solve_count[])] Matrix nnz: $(nnz(A))")
    println("[SOLVE #$(solve_count[])] Matrix diagonal min: $(minimum(abs.(diag(A))))")
    println("[SOLVE #$(solve_count[])] Matrix diagonal max: $(maximum(abs.(diag(A))))")

    # Check eigenvalues for small matrices
    n = size(A, 1)
    if n <= 30
        A_dense = Matrix(A)
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

    # Call original
    result = original_solve(A, b)
    println("[SOLVE #$(solve_count[])] Success!")
    return result
end

MultiGridBarrier.solve(A::SparseMatrixCSC{T}, b::Vector{T}) where {T} = instrumented_solve(A, b)

# Now run the actual solve
println("[DEBUG] Starting fem1d_solve...")

try
    sol = fem1d_solve(Float64; L=2, p=1.0, verbose=false)
    println("[DEBUG] Solve succeeded!")
    println("[DEBUG] Solution z norm: $(norm(sol.z))")
catch e
    println("[DEBUG] Solve FAILED: $e")
    for (ex, bt) in current_exceptions()
        showerror(stdout, ex, bt)
        println()
    end
end

println("[DEBUG] Total solve calls: $(solve_count[])")
println("[DEBUG] Test completed")
