#!/usr/bin/env julia
#
# Basic Solve Example for MultiGridBarrierMPI.jl
#
# This example demonstrates the simplest workflow:
# 1. Initialize MPI
# 2. Solve with MPI distributed types
# 3. Convert solution to native types
# 4. Display results
#
# Run with: mpiexec -n 4 julia --project examples/basic_solve.jl
#

using MPI
MPI.Init()

using MultiGridBarrierMPI
using LinearAlgebraMPI
using MultiGridBarrier
using LinearAlgebra

println(io0(), "="^70)
println(io0(), "Basic Solve Example - MultiGridBarrierMPI.jl")
println(io0(), "="^70)

# Get MPI information
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)
println(io0(), "Running on $nranks MPI ranks\n")

# Problem parameters
L = 2          # Refinement levels (L=2 is fast for demonstration)
p = 1.0        # Barrier power parameter

println(io0(), "Problem Parameters:")
println(io0(), "  Refinement levels (L): $L")
println(io0(), "  Barrier parameter (p): $p")
println(io0(), "")

# Solve with MPI distributed types (collective operation)
println(io0(), "Solving with MPI distributed types...")
sol_mpi = fem2d_mpi_solve(Float64; L=L, p=p, verbose=true)

# Convert solution to native Julia types (collective operation)
println(io0(), "\nConverting solution to native types...")
sol_native = mpi_to_native(sol_mpi)

# Display results (only on rank 0)
println(io0(), "")
println(io0(), "="^70)
println(io0(), "Solution Summary:")
println(io0(), "="^70)

# Solution dimensions
z_size = size(sol_native.z)
println(io0(), "Solution matrix size: $(z_size[1]) x $(z_size[2])")

# Convergence information
n_newton_steps = sum(sol_native.SOL_main.its)
println(io0(), "Total Newton steps: $n_newton_steps")
println(io0(), "Elapsed time: $(sol_native.SOL_main.t_elapsed) seconds")

# Solution statistics
z_norm = norm(sol_native.z)
z_min = minimum(sol_native.z)
z_max = maximum(sol_native.z)
println(io0(), "Solution norm: $z_norm")
println(io0(), "Solution range: [$z_min, $z_max]")

println(io0(), "")
println(io0(), "="^70)
println(io0(), "Example completed successfully!")
println(io0(), "="^70)

# Optional: Save solution for visualization
if rank == 0
    println("\nTo visualize this solution, use:")
    println("  using MultiGridBarrier, PyPlot")
    println("  plot(sol_native)")
    println("  savefig(\"basic_solve_solution.png\")")
end
