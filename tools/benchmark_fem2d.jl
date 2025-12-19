#!/usr/bin/env julia
#
# Benchmark: MultiGridBarrier.jl (Native) vs MultiGridBarrierMPI
#
# Runs fem2d_solve for L = 1:8
#
# Run with:
#   export OMP_NUM_THREADS=1
#   export OPENBLAS_NUM_THREADS=10  # or your number of CPU cores
#   mpiexec -n 1 julia --project=. tools/benchmark_fem2d.jl
#

using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Only rank 0 prints output
io0(args...) = rank == 0 && println(args...)

io0("Loading packages...")
using MultiGridBarrier
using MultiGridBarrierMPI
using Dates
using LinearAlgebra
using Printf


io0("\n" * "="^70)
io0("Benchmark: fem2d_solve - Native vs MPI")
io0("  MPI ranks: $nranks, BLAS threads: $(LinearAlgebra.BLAS.get_num_threads())")
io0("  Running L = 1:5")
io0("="^70)

# Store results
results = Vector{NamedTuple}()

for L in 1:8

    # Get grid size
    g = fem2d(Float64; L=L)
    n = size(g.x, 1)

    io0("\n--- L = $L (n = $n) ---")

    # Native: warmup and timed run on rank 0 only
    native_time = 0.0
    native_sol = nothing
    if rank == 0
        io0("  Warmup Native...")
        fem2d_solve(Float64; L=L, verbose=false)
        io0("  Timed Native...")
        native_sol = fem2d_solve(Float64; L=L, verbose=false)
        native_time = @elapsed fem2d_solve(Float64; L=L, verbose=false)
    end

    # Broadcast native_time to all ranks
    native_time = MPI.Bcast(native_time, 0, comm)

    # MPI: warmup and timed run on all ranks
    io0("  Warmup MPI...")
    fem2d_mpi_solve(Float64; L=L, verbose=false)
    io0("  Timed MPI...")
    mpi_sol = fem2d_mpi_solve(Float64; L=L, verbose=false)
    mpi_time = @elapsed fem2d_mpi_solve(Float64; L=L, verbose=false)

    # Calculate ratio (MPI time / Native time)
    ratio = mpi_time / native_time

    # Compute sup norm of difference (on rank 0)
    sup_diff = 0.0
    if rank == 0
        mpi_native = mpi_to_native(mpi_sol)
        sup_diff = norm(native_sol.z - mpi_native.z, Inf)
    end

    push!(results, (L=L, n=n, native=native_time, mpi=mpi_time, ratio=ratio, diff=sup_diff))

    io0("  Native: $(round(native_time, digits=3))s")
    io0("  MPI:    $(round(mpi_time, digits=3))s")
    io0("  Ratio:  $(round(ratio, digits=2))x")
    io0("  Diff:   $(sup_diff)")

end

# Summary table
io0("\n" * "="^70)
io0("Summary")
io0("="^70)
io0("\n  L       n       Native          MPI             Ratio       Diff")
io0("  -       -       ------          ---             -----       ----")
for r in results
    n_str = lpad(r.n, 7)
    native_str = lpad(round(r.native, digits=3), 10) * "s"
    mpi_str = lpad(round(r.mpi, digits=3), 10) * "s"
    ratio_str = lpad(round(r.ratio, digits=2), 8) * "x"
    diff_str = @sprintf("%.2e", r.diff)
    io0("  $(r.L)    $n_str    $native_str    $mpi_str    $ratio_str    $diff_str")
end

io0("\n  Ratio = MPI time / Native time (lower is better)")
io0("="^70)

# Save HTML results (rank 0 only)
if rank == 0
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>fem2d Benchmark Results</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
        h1 { color: #333; }
        .info { background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: right; }
        th { background: #4a90d9; color: white; }
        td:first-child { text-align: center; }
        tr:nth-child(even) { background: #f9f9f9; }
        .faster { color: green; font-weight: bold; }
        .slower { color: #c00; }
    </style>
</head>
<body>
    <h1>fem2d Benchmark: Native vs MPI</h1>
    <div class="info">
        <strong>Configuration:</strong><br>
        MPI ranks: $nranks<br>
        BLAS threads: $(LinearAlgebra.BLAS.get_num_threads())<br>
        Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    </div>
    <table>
        <tr>
            <th>L</th>
            <th>n (grid points)</th>
            <th>Native (s)</th>
            <th>MPI (s)</th>
            <th>Ratio</th>
        </tr>
"""
    for r in results
        ratio_class = r.ratio < 1 ? "faster" : "slower"
        global html *= """
        <tr>
            <td>$(r.L)</td>
            <td>$(r.n)</td>
            <td>$(round(r.native, digits=3))</td>
            <td>$(round(r.mpi, digits=3))</td>
            <td class="$ratio_class">$(round(r.ratio, digits=2))x</td>
        </tr>
"""
    end
    html *= """
    </table>
    <p><em>Ratio = MPI time / Native time (lower is better)</em></p>
</body>
</html>
"""

    open("tools/benchmark_fem2d_results.html", "w") do f
        write(f, html)
    end
    io0("\nResults saved to tools/benchmark_fem2d_results.html")
end
