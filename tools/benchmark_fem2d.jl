#!/usr/bin/env julia
#
# Benchmark: MultiGridBarrier.jl (Native) vs MultiGridBarrierMPI
#
# Runs fem2d_solve for increasing L until runtime exceeds 30s
#
# Run with: mpiexec -n 4 julia -t 2 --project=. tools/benchmark_fem2d.jl
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

# Initialize MPI package
MultiGridBarrierMPI.Init()

io0("\n" * "="^70)
io0("Benchmark: fem2d_solve - Native vs MPI")
io0("  MPI ranks: $nranks, Julia threads: $(Threads.nthreads())")
io0("  Stop when runtime > 30s")
io0("="^70)

# Store results
results = Vector{NamedTuple}()

L = 0
while true
    global L += 1

    # Get grid size
    g = fem2d(Float64; L=L)
    n = size(g.x, 1)

    io0("\n--- L = $L (n = $n) ---")

    # Native: warmup and timed run on rank 0 only
    native_time = 0.0
    if rank == 0
        io0("  Warmup Native...")
        fem2d_solve(Float64; L=L, verbose=false)
        io0("  Timed Native...")
        native_time = @elapsed fem2d_solve(Float64; L=L, verbose=false)
    end

    # Broadcast native_time to all ranks
    native_time = MPI.Bcast(native_time, 0, comm)

    # MPI: warmup and timed run on all ranks
    io0("  Warmup MPI...")
    fem2d_mpi_solve(Float64; L=L, verbose=false)
    io0("  Timed MPI...")
    mpi_time = @elapsed fem2d_mpi_solve(Float64; L=L, verbose=false)

    # Calculate speedup
    speedup = native_time / mpi_time

    push!(results, (L=L, n=n, native=native_time, mpi=mpi_time, speedup=speedup))

    io0("  Native: $(round(native_time, digits=3))s")
    io0("  MPI:    $(round(mpi_time, digits=3))s")
    io0("  Speedup: $(round(speedup, digits=2))x")

    # Stop if either time exceeds 30s
    if native_time > 30 || mpi_time > 30
        io0("\n  Stopping: runtime exceeded 30s")
        break
    end
end

# Summary table
io0("\n" * "="^70)
io0("Summary")
io0("="^70)
io0("\n  L       n       Native          MPI             Speedup")
io0("  -       -       ------          ---             -------")
for r in results
    n_str = lpad(r.n, 7)
    native_str = lpad(round(r.native, digits=3), 10) * "s"
    mpi_str = lpad(round(r.mpi, digits=3), 10) * "s"
    speedup_str = lpad(round(r.speedup, digits=2), 8) * "x"
    io0("  $(r.L)    $n_str    $native_str    $mpi_str    $speedup_str")
end

io0("\n  Speedup > 1 means MPI is faster than Native")
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
        Julia threads: $(Threads.nthreads())<br>
        Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    </div>
    <table>
        <tr>
            <th>L</th>
            <th>n (grid points)</th>
            <th>Native (s)</th>
            <th>MPI (s)</th>
            <th>Speedup</th>
        </tr>
"""
    for r in results
        speedup_class = r.speedup > 1 ? "faster" : "slower"
        global html *= """
        <tr>
            <td>$(r.L)</td>
            <td>$(r.n)</td>
            <td>$(round(r.native, digits=3))</td>
            <td>$(round(r.mpi, digits=3))</td>
            <td class="$speedup_class">$(round(r.speedup, digits=2))x</td>
        </tr>
"""
    end
    html *= """
    </table>
    <p><em>Speedup &gt; 1 means MPI is faster than Native</em></p>
</body>
</html>
"""

    open("tools/benchmark_fem2d_results.html", "w") do f
        write(f, html)
    end
    io0("\nResults saved to tools/benchmark_fem2d_results.html")
end
