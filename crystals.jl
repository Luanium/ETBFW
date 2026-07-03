# ============================================================
# crystals.jl
#
# Utility templates for common crystalline structures.
# ============================================================

"""
    make_sc(a::Float64, species::Symbol) -> System

Create a Simple Cubic (SC) structure with lattice constant `a`.
"""
function make_sc(a::Float64, species::Symbol)
    lat = Lattice([a, 0.0, 0.0],
                  [0.0, a, 0.0],
                  [0.0, 0.0, a])
    sys = System(lat)
    add_atom!(sys, species, [0.0, 0.0, 0.0])
    return sys
end

"""
    make_bcc(a::Float64, species::Symbol) -> System

Create a Body-Centered Cubic (BCC) structure with lattice constant `a`.
(Uses primitive lattice vectors).
"""
function make_bcc(a::Float64, species::Symbol)
    lat = Lattice((a / 2.0) * [-1.0, 1.0, 1.0],
                  (a / 2.0) * [ 1.0,-1.0, 1.0],
                  (a / 2.0) * [ 1.0, 1.0,-1.0])
    sys = System(lat)
    add_atom!(sys, species, [0.0, 0.0, 0.0])
    return sys
end

"""
    make_fcc(a::Float64, species::Symbol) -> System

Create a Face-Centered Cubic (FCC) structure with lattice constant `a`.
(Uses primitive lattice vectors).
"""
function make_fcc(a::Float64, species::Symbol)
    lat = Lattice((a / 2.0) * [0.0 1.0 1.0;
                               1.0 0.0 1.0;
                               1.0 1.0 0.0])
    sys = System(lat)
    add_atom!(sys, species, [0.0, 0.0, 0.0])
    return sys
end

"""
    make_diamond(a::Float64, species::Symbol) -> System

Create a Diamond structure with lattice constant `a`.
(Uses FCC primitive lattice vectors with a two-atom basis).
"""
function make_diamond(a::Float64, species::Symbol)
    lat = Lattice((a / 2.0) * [0.0, 1.0, 1.0],
                  (a / 2.0) * [1.0, 0.0, 1.0],
                  (a / 2.0) * [1.0, 1.0, 0.0])
    sys = System(lat)
    add_atom!(sys, species, [0.0, 0.0, 0.0])
    add_atom!(sys, species, (a / 4.0) * [1.0, 1.0, 1.0])
    return sys
end

"""
    make_zincblende(a::Float64, species1::Symbol, species2::Symbol) -> System

Create a Zincblende structure with lattice constant `a`.
(Uses FCC primitive lattice vectors with a two-atom basis of different species).
"""
function make_zincblende(a::Float64, species1::Symbol, species2::Symbol)
    lat = Lattice((a / 2.0) * [0.0, 1.0, 1.0],
                  (a / 2.0) * [1.0, 0.0, 1.0],
                  (a / 2.0) * [1.0, 1.0, 0.0])
    sys = System(lat)
    add_atom!(sys, species1, [0.0, 0.0, 0.0])
    add_atom!(sys, species2, (a / 4.0) * [1.0, 1.0, 1.0])
    return sys
end

"""
    make_graphene(a::Float64, species::Symbol=:C) -> System

Create a Graphene (2D hexagonal) structure with lattice constant `a`.
(Lattice vectors in xy plane, large vacuum in z).
"""
function make_graphene(a::Float64, species::Symbol=:C)
    lat = Lattice([a, 0.0, 0.0],
                  [a/2.0, a*sqrt(3)/2, 0.0],
                  nothing)
    sys = System(lat)
    add_atom!(sys, species, [0.0, 0.0, 0.0])
    add_atom!(sys, species, [a/2.0, a/(2.0*sqrt(3)), 0.0])
    return sys
end

"""
    make_1d_chain(a::Float64, species::Symbol) -> System

Create a 1D chain of atoms with lattice constant `a` (along x-axis).
"""
function make_1d_chain(a::Float64, species::Symbol)
    lat = Lattice([a, 0.0, 0.0], nothing, nothing)
    sys = System(lat)
    add_atom!(sys, species, [0.0, 0.0, 0.0])
    return sys
end
