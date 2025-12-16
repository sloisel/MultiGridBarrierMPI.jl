#!/usr/bin/env julia
#
# Benchmark: MultiGridBarrier.jl (Native) vs MultiGridBarrierMPI
#
# Compares fem2d_solve performance for L=1:6
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

# Initialize MPI package
MultiGridBarrierMPI.Init()

io0("\n" * "="^70)
io0("Benchmark: fem2d_solve - Native vs MPI")
io0("  MPI ranks: $nranks, Julia threads: $(Threads.nthreads())")
io0("="^70)

# Store results
results = Dict{Int, NamedTuple}()

for L in 1:6
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

    # Synchronize before MPI benchmark
    MPI.Allreduce(0, MPI.SUM, comm)

    # MPI: warmup and timed run on all ranks
    io0("  Warmup MPI...")
    fem2d_mpi_solve(Float64; L=L, verbose=false)
    io0("  Timed MPI...")
    mpi_time = @elapsed fem2d_mpi_solve(Float64; L=L, verbose=false)

    # Calculate speedup
    speedup = rank == 0 ? native_time / mpi_time : 0.0

    results[L] = (n=n, native=native_time, mpi=mpi_time, speedup=speedup)

    io0("  Native: $(round(native_time, digits=3))s")
    io0("  MPI:    $(round(mpi_time, digits=3))s")
    io0("  Speedup: $(round(speedup, digits=2))x")
end

# Summary table
io0("\n" * "="^70)
io0("Summary")
io0("="^70)
io0("\n  L       n       Native          MPI             Speedup")
io0("  -       -       ------          ---             -------")
for L in 1:6
    r = results[L]
    n_str = lpad(r.n, 7)
    native_str = lpad(round(r.native, digits=3), 10) * "s"
    mpi_str = lpad(round(r.mpi, digits=3), 10) * "s"
    speedup_str = lpad(round(r.speedup, digits=2), 8) * "x"
    io0("  $L    $n_str    $native_str    $mpi_str    $speedup_str")
end

io0("\n  Speedup > 1 means MPI is faster than Native")
io0("="^70)
