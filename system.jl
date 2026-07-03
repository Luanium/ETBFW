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
    a1::Union{Vector{Float64}, Nothing}
    a2::Union{Vector{Float64}, Nothing}
    a3::Union{Vector{Float64}, Nothing}
end

function get_lattice_params(lattice::Lattice)
    valid_vecs = Vector{Float64}[]
    valid_indices = Int[]
    
    if !isnothing(lattice.a1); push!(valid_vecs, lattice.a1); push!(valid_indices, 1); end
    if !isnothing(lattice.a2); push!(valid_vecs, lattice.a2); push!(valid_indices, 2); end
    if !isnothing(lattice.a3); push!(valid_vecs, lattice.a3); push!(valid_indices, 3); end
    
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
    if !isnothing(lattice.a1); v += R[1] * lattice.a1; end
    if !isnothing(lattice.a2); v += R[2] * lattice.a2; end
    if !isnothing(lattice.a3); v += R[3] * lattice.a3; end
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
    num_atoms(system::System) -> Int
"""
num_atoms(system::System) = length(system.atoms)


