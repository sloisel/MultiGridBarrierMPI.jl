#!/usr/bin/env julia
#
# Profile individual operations to identify bottlenecks
#

using MPI
MPI.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
io0(args...) = rank == 0 && println(args...)

io0("Loading packages...")
using MultiGridBarrier
using MultiGridBarrierMPI
using HPCLinearAlgebra
using LinearAlgebra
using SparseArrays
using Printf

L = 5

io0("\n" * "="^70)
io0("Operation timing at L=$L")
io0("="^70)

# Create geometries
io0("\nCreating geometries...")
g_mpi = fem2d_mpi(Float64; L=L)
g_native = rank == 0 ? fem2d(Float64; L=L) : nothing

n = size(g_mpi.x, 1)
io0("  n = $n")

# Get some operators
Dx_mpi = g_mpi.operators[:dx]
Dy_mpi = g_mpi.operators[:dy]
Id_mpi = g_mpi.operators[:id]

if rank == 0
    Dx_native = g_native.operators[:dx]
    Dy_native = g_native.operators[:dy]
    Id_native = g_native.operators[:id]
end

# Time various operations
io0("\n--- Sparse Matrix Operations ---")

# Transpose multiply: Dx' * Dy
io0("\nDx' * Dy (10 times each):")
for _ in 1:3  # warmup
    _ = Dx_mpi' * Dy_mpi
end
t_mpi = @elapsed for _ in 1:10
    _ = Dx_mpi' * Dy_mpi
end
io0("  MPI:    $(round(t_mpi/10*1000, digits=2)) ms per op")

if rank == 0
    for _ in 1:3
        _ = Dx_native' * Dy_native
    end
    t_native = @elapsed for _ in 1:10
        _ = Dx_native' * Dy_native
    end
    io0("  Native: $(round(t_native/10*1000, digits=2)) ms per op")
    io0("  Ratio:  $(round(t_mpi/t_native, digits=2))x")
end

# map_rows timing
io0("\nmap_rows (barrier Hessian-like, 10 times each):")
x_mpi = g_mpi.x
f_hess = x -> [sum(x.^2), prod(x)]'

for _ in 1:3
    _ = HPCLinearAlgebra.map_rows(f_hess, x_mpi)
end
t_mpi = @elapsed for _ in 1:10
    _ = HPCLinearAlgebra.map_rows(f_hess, x_mpi)
end
io0("  MPI:    $(round(t_mpi/10*1000, digits=2)) ms per op")

if rank == 0
    x_native = g_native.x
    for _ in 1:3
        _ = [f_hess(x_native[i,:])' for i in 1:size(x_native,1)]
    end
    t_native = @elapsed for _ in 1:10
        _ = vcat([f_hess(x_native[i,:])' for i in 1:size(x_native,1)]...)
    end
    io0("  Native: $(round(t_native/10*1000, digits=2)) ms per op")
    io0("  Ratio:  $(round(t_mpi/t_native, digits=2))x")
end

# Stiffness-like matrix: Dx' * Dx + Dy' * Dy
io0("\nDx'*Dx + Dy'*Dy (stiffness-like, 10 times each):")
for _ in 1:3
    _ = Dx_mpi' * Dx_mpi + Dy_mpi' * Dy_mpi
end
t_mpi = @elapsed for _ in 1:10
    _ = Dx_mpi' * Dx_mpi + Dy_mpi' * Dy_mpi
end
io0("  MPI:    $(round(t_mpi/10*1000, digits=2)) ms per op")

if rank == 0
    for _ in 1:3
        _ = Dx_native' * Dx_native + Dy_native' * Dy_native
    end
    t_native = @elapsed for _ in 1:10
        _ = Dx_native' * Dx_native + Dy_native' * Dy_native
    end
    io0("  Native: $(round(t_native/10*1000, digits=2)) ms per op")
    io0("  Ratio:  $(round(t_mpi/t_native, digits=2))x")
end

# Linear solve timing with stiffness matrix
io0("\nLinear solve with stiffness (3 times each):")
K_mpi = Dx_mpi' * Dx_mpi + Dy_mpi' * Dy_mpi + 0.01 * Id_mpi
b_mpi = HPCVector(randn(n))

for _ in 1:2
    _ = K_mpi \ b_mpi
end
t_mpi = @elapsed for _ in 1:3
    _ = K_mpi \ b_mpi
end
io0("  MPI (MUMPS): $(round(t_mpi/3*1000, digits=2)) ms per solve")

if rank == 0
    K_native = Dx_native' * Dx_native + Dy_native' * Dy_native + 0.01 * Id_native
    b_native = Vector(b_mpi)
    for _ in 1:2
        _ = K_native \ b_native
    end
    t_native = @elapsed for _ in 1:3
        _ = K_native \ b_native
    end
    io0("  Native:      $(round(t_native/3*1000, digits=2)) ms per solve")
    io0("  Ratio:       $(round(t_mpi/t_native, digits=2))x")
end

io0("\n" * "="^70)
