# ============================================================
# system.jl
#
# Framework layer — atomistic system construction.
#
# Provides the data structures and utilities used to build the
# physical structure (lattice + atoms) that any ETB model operates on.
# Models never construct these themselves; they only *read* from a
# System instance handed to them by the framework.
# ============================================================

"""
    Atom

A single atom in the system.

- `species`   : chemical species / site label (e.g. :A, :Fe, :C1)
- `position`  : Cartesian coordinates (Angstrom)
"""
struct Atom
    species::Symbol
    position::Vector{Float64}
end

"""
    Lattice

Periodic lattice vectors, stored column-wise (`vectors[:, i]` is the i-th
primitive vector). Works for 1D/2D/3D periodicity by padding with large
vacuum vectors in non-periodic directions (a common convention, e.g. QuantumATK-style).
"""
struct Lattice
    a1::Union{Vector{Float64},Nothing}
    a2::Union{Vector{Float64},Nothing}
    a3::Union{Vector{Float64},Nothing}
end

function get_lattice_params(lattice::Lattice)
    valid_vecs = Vector{Float64}[]
    valid_indices = Int[]

    if !isnothing(lattice.a1)
        push!(valid_vecs, lattice.a1)
        push!(valid_indices, 1)
    end
    if !isnothing(lattice.a2)
        push!(valid_vecs, lattice.a2)
        push!(valid_indices, 2)
    end
    if !isnothing(lattice.a3)
        push!(valid_vecs, lattice.a3)
        push!(valid_indices, 3)
    end

    dim = length(valid_vecs)
    if dim == 0
        return zeros(Float64, 0, 3), zeros(Float64, 0), valid_indices
    end

    A = hcat(valid_vecs...) # 3 x dim
    pseudo_inv = inv(A' * A) * A' # dim x 3
    row_norms = [norm(pseudo_inv[k, :]) for k in 1:dim]
    return pseudo_inv, row_norms, valid_indices
end

function get_dR(lattice::Lattice, R::Vector{Int})
    v = zeros(Float64, 3)
    if !isnothing(lattice.a1)
        v += R[1] * lattice.a1
    end
    if !isnothing(lattice.a2)
        v += R[2] * lattice.a2
    end
    if !isnothing(lattice.a3)
        v += R[3] * lattice.a3
    end
    return v
end

"""
    System

The full atomistic system: a lattice plus a collection of atoms in the
unit cell. This is the single object passed from the framework into a
model's `parse_parameters` / initialization routine.
"""
mutable struct System
    lattice::Lattice
    atoms::Vector{Atom}
    dim::Int
end

"""
    System(lattice::Lattice)

Construct an empty system with the given lattice, ready for atoms to be added.
"""
function System(lattice::Lattice)
    dim = (!isnothing(lattice.a1) ? 1 : 0) +
          (!isnothing(lattice.a2) ? 1 : 0) +
          (!isnothing(lattice.a3) ? 1 : 0)
    return System(lattice, Atom[], dim)
end

"""
    add_atom!(system::System, species::Symbol, position::Vector{Float64})

Add an atom to the system.
"""
function add_atom!(system::System, species::Symbol, position::Vector{Float64})
    push!(system.atoms, Atom(species, position))
    return system
end

"""
    unitcell_volume(lattice::Lattice) -> Float64
    unitcell_volume(system::System) -> Float64

Calculates the volume (or area/length for lower dimensions) of the unit cell.
"""
function unitcell_volume(lattice::Lattice)
    valid_vecs = Vector{Float64}[]
    if !isnothing(lattice.a1)
        push!(valid_vecs, lattice.a1)
    end
    if !isnothing(lattice.a2)
        push!(valid_vecs, lattice.a2)
    end
    if !isnothing(lattice.a3)
        push!(valid_vecs, lattice.a3)
    end

    dim = length(valid_vecs)
    if dim == 0
        return 0.0
    elseif dim == 1
        return norm(valid_vecs[1])
    elseif dim == 2
        return norm(cross(valid_vecs[1], valid_vecs[2]))
    else
        return abs(dot(valid_vecs[1], cross(valid_vecs[2], valid_vecs[3])))
    end
end
unitcell_volume(system::System) = unitcell_volume(system.lattice)

"""
    reciprocal_lattice(lattice::Lattice) -> Tuple
    reciprocal_lattice(system::System) -> Tuple

Returns the reciprocal lattice vectors (b1, b2, b3). If a real-space lattice
vector is nothing, the corresponding reciprocal vector is also nothing.
Satisfies a_i ⋅ b_j = 2π δ_ij.
"""
function reciprocal_lattice(lattice::Lattice)
    pseudo_inv, _, valid_indices = get_lattice_params(lattice)
    b1 = b2 = b3 = nothing

    if !isempty(valid_indices)
        B_T = 2 * pi * pseudo_inv
        for (i, v_idx) in enumerate(valid_indices)
            b = B_T[i, :]
            if v_idx == 1
                b1 = b
            end
            if v_idx == 2
                b2 = b
            end
            if v_idx == 3
                b3 = b
            end
        end
    end
    return b1, b2, b3
end
reciprocal_lattice(system::System) = reciprocal_lattice(system.lattice)

"""
    num_atoms(system::System) -> Int
"""
num_atoms(system::System) = length(system.atoms)


