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
- `orbitals`  : list of orbital labels living on this atom (e.g. ["s"], ["px","py","pz"])
"""
struct Atom
    species::Symbol
    position::Vector{Float64}
    orbitals::Vector{String}
end

"""
    Lattice

Periodic lattice vectors, stored column-wise (`vectors[:, i]` is the i-th
primitive vector). Works for 1D/2D/3D periodicity by padding with large
vacuum vectors in non-periodic directions (a common convention, e.g. QuantumATK-style).
"""
struct Lattice
    vectors::Matrix{Float64}
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
end

"""
    System(lattice::Lattice)

Construct an empty system with the given lattice, ready for atoms to be added.
"""
System(lattice::Lattice) = System(lattice, Atom[])

"""
    add_atom!(system::System, species::Symbol, position::Vector{Float64}, orbitals::Vector{String}=["s"])

Add an atom to the system.
"""
function add_atom!(system::System, species::Symbol, position::Vector{Float64}, orbitals::Vector{String}=["s"])
    push!(system.atoms, Atom(species, position, orbitals))
    return system
end

"""
    num_atoms(system::System) -> Int
"""
num_atoms(system::System) = length(system.atoms)

"""
    num_orbitals(system::System) -> Int

Total number of orbitals in the system (i.e. the Hamiltonian matrix dimension).
"""
num_orbitals(system::System) = sum(length(a.orbitals) for a in system.atoms; init=0)

"""
    orbital_index_map(system::System) -> Dict{Tuple{Int,Int},Int}

Maps (atom_index, local_orbital_index) -> global basis index (1-based),
in atom order. This is a framework utility so that every model builds
its Hamiltonian with a consistent, shared indexing convention.
"""
function orbital_index_map(system::System)
    idx_map = Dict{Tuple{Int,Int},Int}()
    idx = 1
    for (i, atom) in enumerate(system.atoms)
        for j in 1:length(atom.orbitals)
            idx_map[(i, j)] = idx
            idx += 1
        end
    end
    return idx_map
end
