#!/usr/bin/env julia
#
# Detailed step-by-step timing of map_rows
#
# Run with: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=10 mpiexec -n 1 julia --project=. tools/profile_map_rows_steps.jl
#

using MPI
MPI.Init()

using MultiGridBarrier
using MultiGridBarrierMPI
using HPCSparseArrays
using HPCSparseArrays: HPCVector, HPCMatrix, HPCVector_local, HPCMatrix_local
using HPCSparseArrays: _get_row_partition, _align_to_partition, _local_rows
using LinearAlgebra
import Statistics: mean, median

MultiGridBarrierMPI.Init()

const L = 6
const N_ITER = 100  # Number of iterations for timing

println("="^70)
println("Step-by-step map_rows timing at L=$L")
println("="^70)

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Create geometry
g_mpi = fem2d_mpi(Float64; L=L)
x_mpi = g_mpi.x  # HPCMatrix (n x 2)
w_mpi = g_mpi.w  # HPCVector (n)

println("Grid size (global): ", sum(x_mpi.row_partition))
println("Local rows: ", size(x_mpi.A, 1))
println("MPI ranks: ", nranks)

# Test function (typical barrier pattern)
f = (row_x, w) -> w * sum(row_x)

println("\n" * "-"^70)
println("Timing each step of map_rows (averaged over $N_ITER iterations)")
println("-"^70)

# Collect timings
times = Dict{String, Vector{Float64}}()
for key in ["total", "get_partition", "align", "local_rows", "comprehension",
            "type_detect", "allgather", "result_build"]
    times[key] = Float64[]
end

for iter in 1:N_ITER
    A = (x_mpi, w_mpi)

    t_total_start = time_ns()

    # Step 1: Get target partition
    t1 = time_ns()
    target_partition = _get_row_partition(A[1])
    t1 = time_ns() - t1
    push!(times["get_partition"], t1)

    # Step 2: Align all arguments
    t2 = time_ns()
    aligned = map(a -> _align_to_partition(a, target_partition), A)
    t2 = time_ns() - t2
    push!(times["align"], t2)

    # Step 3: Get local row iterators
    t3 = time_ns()
    row_iters = map(_local_rows, aligned)
    t3 = time_ns() - t3
    push!(times["local_rows"], t3)

    # Step 4: Apply function with comprehension
    t4 = time_ns()
    results = [f(rows...) for rows in zip(row_iters...)]
    t4 = time_ns() - t4
    push!(times["comprehension"], t4)

    # Step 5: Type detection (local)
    t5 = time_ns()
    local_info = if isempty(results)
        Int32[0, 0, 0, 0]
    else
        first_result = results[1]
        kind = if first_result isa Number
            Int32(1)
        elseif first_result isa AbstractVector
            Int32(2)
        elseif first_result isa AbstractMatrix
            Int32(3)
        else
            Int32(0)
        end
        T = first_result isa Number ? typeof(first_result) : eltype(first_result)
        eltype_code = if T == Float64
            Int32(1)
        elseif T == ComplexF64
            Int32(2)
        elseif T <: Integer
            Int32(3)
        else
            Int32(4)
        end
        ncols = first_result isa AbstractMatrix ? Int32(size(first_result, 2)) : Int32(0)
        Int32[1, kind, eltype_code, ncols]
    end
    t5 = time_ns() - t5
    push!(times["type_detect"], t5)

    # Step 6: MPI.Allgather
    t6 = time_ns()
    all_info = MPI.Allgather(local_info, comm)
    t6 = time_ns() - t6
    push!(times["allgather"], t6)

    # Step 7: Build result (scalar case - most common for barrier)
    t7 = time_ns()
    # Determine result type from all_info
    result_kind = Int32(0)
    eltype_code = Int32(1)
    ncols_result = Int32(0)
    for r in 0:(nranks-1)
        idx = r * 4
        if all_info[idx + 1] == 1
            result_kind = all_info[idx + 2]
            eltype_code = all_info[idx + 3]
            ncols_result = all_info[idx + 4]
            break
        end
    end

    # Build HPCVector for scalar results
    if result_kind == 1
        local_v = Vector{Float64}(undef, length(results))
        for (i, r) in enumerate(results)
            local_v[i] = r
        end
        result = HPCVector_local(local_v)
    end
    t7 = time_ns() - t7
    push!(times["result_build"], t7)

    t_total = time_ns() - t_total_start
    push!(times["total"], t_total)
end

# Print results
println("\nStep                    Median (μs)    Mean (μs)    % of total")
println("-"^70)

total_median = median(times["total"]) / 1000  # convert to μs

for (name, label) in [
    ("get_partition", "1. Get partition"),
    ("align", "2. Align to partition"),
    ("local_rows", "3. Get local row iters"),
    ("comprehension", "4. Comprehension [f...]"),
    ("type_detect", "5. Type detection"),
    ("allgather", "6. MPI.Allgather"),
    ("result_build", "7. Build result"),
]
    med = median(times[name]) / 1000  # μs
    avg = mean(times[name]) / 1000
    pct = 100 * med / total_median
    println("$(rpad(label, 24)) $(lpad(round(med, digits=1), 10))   $(lpad(round(avg, digits=1), 10))   $(lpad(round(pct, digits=1), 6))%")
end
println("-"^70)
println("$(rpad("TOTAL", 24)) $(lpad(round(total_median, digits=1), 10))   $(lpad(round(mean(times["total"])/1000, digits=1), 10))   100.0%")

# Now compare with native
println("\n" * "-"^70)
println("Comparison: Native map_rows")
println("-"^70)

x_native = x_mpi.A
w_native = w_mpi.v

native_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = MultiGridBarrier.map_rows(f, x_native, w_native)
    t = time_ns() - t
    push!(native_times, t)
end

native_median = median(native_times) / 1000
println("Native map_rows median: $(round(native_median, digits=1)) μs")
println("MPI map_rows median:    $(round(total_median, digits=1)) μs")
println("Overhead ratio:         $(round(total_median / native_median, digits=2))x")

# Now test with a row-vector returning function (like barrier f1/f2)
println("\n" * "-"^70)
println("Test with row-vector returning function (like f1)")
println("-"^70)

f_rowvec = (row_x, w) -> (w * row_x)'  # Returns 1x2 row vector

# MPI version
mpi_rowvec_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = MultiGridBarrier.map_rows(f_rowvec, x_mpi, w_mpi)
    t = time_ns() - t
    push!(mpi_rowvec_times, t)
end

# Native version
native_rowvec_times = Float64[]
for _ in 1:N_ITER
    t = time_ns()
    result = MultiGridBarrier.map_rows(f_rowvec, x_native, w_native)
    t = time_ns() - t
    push!(native_rowvec_times, t)
end

println("Native (row-vec): $(round(median(native_rowvec_times)/1000, digits=1)) μs")
println("MPI (row-vec):    $(round(median(mpi_rowvec_times)/1000, digits=1)) μs")
println("Overhead ratio:   $(round(median(mpi_rowvec_times) / median(native_rowvec_times), digits=2))x")

println("\nDone.")
