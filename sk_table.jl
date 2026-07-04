# ============================================================
# sk_table.jl
#
# Framework layer — general Slater-Koster two-center integral table.
#
# Orbitals and bond types are plain Symbols (e.g. :s, :px, :dxy,
# :sigma, :pi, :delta). There is no fixed enum or basis-set struct:
# any orbital symbol recognized below can be used by any model. A
# model only needs to declare which orbitals/bonds are used and
# supply the corresponding integral values in a Dict{Symbol,Float64}
# (or Dict{Tuple{Symbol,Symbol,Symbol},Float64} keyed by
# (orb_kind_a, orb_kind_b, bond_type) — see `sk_integral_key`).
#
# This is model-agnostic: any model (Jancu1998, orthogonal sp3, ...)
# can call `sk_block` / `sk_matrix_element` to build its Hamiltonian
# without re-deriving the Slater-Koster angular formulas itself.
#
# Reference: J.C. Slater and G.F. Koster, Phys. Rev. 94, 1498 (1954).
# ============================================================

const SQRT3 = sqrt(3.0)

"""
    ORBITAL_KIND::Dict{Symbol,Symbol}

Maps an orbital symbol to its angular-momentum family (`:s`, `:p`, or
`:d`). New orbitals can be added here without touching any model file.
"""
const ORBITAL_KIND = Dict{Symbol,Symbol}(
    :s => :s, :e => :s,
    :px => :p, :py => :p, :pz => :p,
    :dxy => :d, :dyz => :d, :dzx => :d, :dx2y2 => :d, :dz2 => :d,
)

orbital_kind(orb::Symbol) = ORBITAL_KIND[orb]

"""
    available_orbitals() -> Dict{Symbol,Symbol}

Dictionary of all orbitals recognized by the SK table: orbital symbol
-> angular-momentum kind (:s, :p, :d).
"""
available_orbitals() = ORBITAL_KIND

"""
    required_couplings(orbitals::Vector{Symbol}) -> Set{Symbol}

Given a basis set of orbitals, returns the set of integral keys
(e.g. :sss, :sps, :pps, :ppp, :pds, :dps, ...) required to build all
matrix elements among them.
"""
function required_couplings(orbitals::Vector{Symbol})
    fields = Set{Symbol}()
    letter(orb) = orb == :e ? "e" : orbital_kind(orb) == :s ? "s" : orbital_kind(orb) == :p ? "p" : "d"
    for o1 in orbitals, o2 in orbitals
        k1, k2 = orbital_kind(o1), orbital_kind(o2)
        l1, l2 = letter(o1), letter(o2)
        if k1 == :p && k2 == :p
            push!(fields, :pps)
            push!(fields, :ppp)
        elseif k1 == :d && k2 == :d
            push!(fields, :dds)
            push!(fields, :ddp)
            push!(fields, :ddd)
        elseif (k1 == :p && k2 == :d) || (k1 == :d && k2 == :p)
            push!(fields, Symbol(l1 * l2 * "s"))
            push!(fields, Symbol(l1 * l2 * "p"))
        else
            push!(fields, Symbol(l1 * l2 * "s"))
        end
    end
    return fields
end

_z(integ::Dict{Symbol,Float64}, key::Symbol) = get(integ, key, 0.0)

"""
    sk_integral_key(orb_a, orb_b, bond) -> Symbol

Builds the canonical integral key for orbital kinds `orb_a`, `orb_b`
(e.g. `:s`, `:p`, `:d`, or `:e`) and bond type (`:sigma`, `:pi`,
`:delta`), e.g. (:s, :p, :sigma) -> :sps, (:p, :p, :pi) -> :ppp.
The bond-type is encoded as its first letter: sigma->s, pi->p, delta->d.
"""
function sk_integral_key(kind_a::Symbol, kind_b::Symbol, bond::Symbol)
    letter(k) = k == :s ? "s" : k == :e ? "e" : k == :p ? "p" : k == :d ? "d" : error("Unknown orbital kind $(k)")
    bl = bond == :sigma ? "s" : bond == :pi ? "p" : bond == :delta ? "d" : error("Unknown bond type $(bond)")
    return Symbol(letter(kind_a) * letter(kind_b) * bl)
end

"""
    sk_matrix_element(orb_a::Symbol, orb_b::Symbol, l, m, n, integ::Dict{Symbol,Float64}) -> Float64

Universal Slater-Koster two-center matrix element <orb_a| H |orb_b>
for a bond with direction cosines (l, m, n) pointing from the source
atom (orbital `orb_a`) to the target atom (orbital `orb_b`), using the
supplied two-center integrals dict `integ` (keys are Symbols like
`:sss`, `:sps`, `:pps`, `:ppp`, `:sds`, `:pds`, `:pdp`, `:dds`, `:ddp`,
`:ddd`, and s*-related keys `:ees`, `:ses`, `:ess`, `:eps`, `:pes`,
`:eds`, `:des`). Missing keys default to zero.
"""
function sk_matrix_element(orb_a::Symbol, orb_b::Symbol, l::Float64, m::Float64, n::Float64, integ::Dict{Symbol,Float64})
    ka, kb = orbital_kind(orb_a), orbital_kind(orb_b)

    # ---------------- s / s* block ----------------
    if ka == :s && kb == :s
        a_star, b_star = orb_a == :e, orb_b == :e
        if !a_star && !b_star
            return _z(integ, :sss)
        elseif a_star && b_star
            return _z(integ, :ess)
        elseif a_star && !b_star
            return _z(integ, :ses)
        else
            return _z(integ, :ses) # s(a)-s*(b): use "sses"-style key if provided
        end
    end

    # ---------------- s / s* <-> p ----------------
    if ka == :s && kb == :p
        dc = orb_b == :px ? l : orb_b == :py ? m : n
        coeff = orb_a == :e ? _z(integ, :eps) : _z(integ, :sps)
        return dc * coeff
    end
    if ka == :p && kb == :s
        dc = orb_a == :px ? l : orb_a == :py ? m : n
        coeff = orb_b == :e ? _z(integ, :pes) : _z(integ, :pss)
        return dc * coeff
    end

    # ---------------- p / p ----------------
    if ka == :p && kb == :p
        return _pp_matrix_element(orb_a, orb_b, l, m, n, _z(integ, :pps), _z(integ, :ppp))
    end

    # ---------------- s / s* <-> d ----------------
    if ka == :s && kb == :d
        coeff = orb_a == :e ? _z(integ, :eds) : _z(integ, :sds)
        return _sd_angular(orb_b, l, m, n) * coeff
    end
    if ka == :d && kb == :s
        coeff = orb_b == :e ? _z(integ, :des) : _z(integ, :dss)
        return _sd_angular(orb_a, l, m, n) * coeff
    end

    # ---------------- p <-> d ----------------
    if ka == :p && kb == :d
        return _pd_angular(orb_a, orb_b, l, m, n, _z(integ, :pds), _z(integ, :pdp))
    end
    if ka == :d && kb == :p
        pds_ = haskey(integ, :dps) ? integ[:dps] : _z(integ, :pds)
        pdp_ = haskey(integ, :dpp) ? integ[:dpp] : _z(integ, :pdp)
        return _pd_angular(orb_b, orb_a, l, m, n, pds_, pdp_)
    end

    # ---------------- d / d ----------------
    if ka == :d && kb == :d
        return _dd_angular(orb_a, orb_b, l, m, n, _z(integ, :dds), _z(integ, :ddp), _z(integ, :ddd))
    end

    error("Unhandled SK orbital pair ($(orb_a), $(orb_b)).")
end

function _pp_angular(orb_a::Symbol, orb_b::Symbol, l::Float64, m::Float64, n::Float64)
    da = orb_a == :px ? l : orb_a == :py ? m : n
    db = orb_b == :px ? l : orb_b == :py ? m : n
    return da, db
end

function _sd_angular(d_orb::Symbol, l::Float64, m::Float64, n::Float64)
    if d_orb == :dxy
        return SQRT3 * l * m
    elseif d_orb == :dyz
        return SQRT3 * m * n
    elseif d_orb == :dzx
        return SQRT3 * n * l
    elseif d_orb == :dx2y2
        return (SQRT3 / 2) * (l^2 - m^2)
    elseif d_orb == :dz2
        return n^2 - (l^2 + m^2) / 2
    end
    error("Unknown d orbital $(d_orb).")
end

function _pd_angular(p_orb::Symbol, d_orb::Symbol, l::Float64, m::Float64, n::Float64, pds::Float64, pdp::Float64)
    if d_orb == :dxy
        if p_orb == :px
            return SQRT3 * l^2 * m * pds + m * (1 - 2 * l^2) * pdp
        elseif p_orb == :py
            return SQRT3 * m^2 * l * pds + l * (1 - 2 * m^2) * pdp
        else
            return SQRT3 * l * m * n * pds - 2 * l * m * n * pdp
        end
    elseif d_orb == :dyz
        if p_orb == :px
            return SQRT3 * l * m * n * pds - 2 * l * m * n * pdp
        elseif p_orb == :py
            return SQRT3 * m^2 * n * pds + n * (1 - 2 * m^2) * pdp
        else
            return SQRT3 * n^2 * m * pds + m * (1 - 2 * n^2) * pdp
        end
    elseif d_orb == :dzx
        if p_orb == :px
            return SQRT3 * l^2 * n * pds + n * (1 - 2 * l^2) * pdp
        elseif p_orb == :py
            return SQRT3 * l * m * n * pds - 2 * l * m * n * pdp
        else
            return SQRT3 * n^2 * l * pds + l * (1 - 2 * n^2) * pdp
        end
    elseif d_orb == :dx2y2
        if p_orb == :px
            return (SQRT3 / 2) * l * (l^2 - m^2) * pds + l * (1 - l^2 + m^2) * pdp
        elseif p_orb == :py
            return (SQRT3 / 2) * m * (l^2 - m^2) * pds - m * (1 + l^2 - m^2) * pdp
        else
            return (SQRT3 / 2) * n * (l^2 - m^2) * pds - n * (l^2 - m^2) * pdp
        end
    elseif d_orb == :dz2
        if p_orb == :px
            return l * (n^2 - (l^2 + m^2) / 2) * pds - SQRT3 * l * n^2 * pdp
        elseif p_orb == :py
            return m * (n^2 - (l^2 + m^2) / 2) * pds - SQRT3 * m * n^2 * pdp
        else
            return n * (n^2 - (l^2 + m^2) / 2) * pds + SQRT3 * n * (l^2 + m^2) * pdp
        end
    end
    error("Unknown d orbital $(d_orb).")
end

function _dd_angular(orb_a::Symbol, orb_b::Symbol, l::Float64, m::Float64, n::Float64, dds::Float64, ddp::Float64, ddd::Float64)
    a, b = orb_a, orb_b
    if a == :dxy && b == :dxy
        return 3 * l^2 * m^2 * dds + (l^2 + m^2 - 4 * l^2 * m^2) * ddp + (n^2 + l^2 * m^2) * ddd
    elseif (a == :dxy && b == :dyz) || (a == :dyz && b == :dxy)
        return 3 * l * m^2 * n * dds + l * n * (1 - 4 * m^2) * ddp + l * n * (m^2 - 1) * ddd
    elseif (a == :dxy && b == :dzx) || (a == :dzx && b == :dxy)
        return 3 * l^2 * m * n * dds + m * n * (1 - 4 * l^2) * ddp + m * n * (l^2 - 1) * ddd
    elseif a == :dyz && b == :dyz
        return 3 * m^2 * n^2 * dds + (m^2 + n^2 - 4 * m^2 * n^2) * ddp + (l^2 + m^2 * n^2) * ddd
    elseif (a == :dyz && b == :dzx) || (a == :dzx && b == :dyz)
        return 3 * m * n^2 * l * dds + m * l * (1 - 4 * n^2) * ddp + m * l * (n^2 - 1) * ddd
    elseif a == :dzx && b == :dzx
        return 3 * n^2 * l^2 * dds + (n^2 + l^2 - 4 * n^2 * l^2) * ddp + (m^2 + n^2 * l^2) * ddd
    elseif (a == :dxy && b == :dx2y2) || (a == :dx2y2 && b == :dxy)
        return (3.0 / 2) * l * m * (l^2 - m^2) * dds + 2 * l * m * (m^2 - l^2) * ddp + (l * m * (l^2 - m^2) / 2) * ddd
    elseif (a == :dyz && b == :dx2y2) || (a == :dx2y2 && b == :dyz)
        return (3.0 / 2) * m * n * (l^2 - m^2) * dds - m * n * (1 + 2 * (l^2 - m^2)) * ddp + m * n * (1 + (l^2 - m^2) / 2) * ddd
    elseif (a == :dzx && b == :dx2y2) || (a == :dx2y2 && b == :dzx)
        return (3.0 / 2) * n * l * (l^2 - m^2) * dds + n * l * (1 - 2 * (l^2 - m^2)) * ddp - n * l * (1 - (l^2 - m^2) / 2) * ddd
    elseif a == :dx2y2 && b == :dx2y2
        return (3.0 / 4) * (l^2 - m^2)^2 * dds + (l^2 + m^2 - (l^2 - m^2)^2) * ddp + (n^2 + (l^2 - m^2)^2 / 4) * ddd
    elseif (a == :dxy && b == :dz2) || (a == :dz2 && b == :dxy)
        return SQRT3 * l * m * (n^2 - (l^2 + m^2) / 2) * dds - 2 * SQRT3 * l * m * n^2 * ddp + (SQRT3 / 2) * l * m * (1 + n^2) * ddd
    elseif (a == :dyz && b == :dz2) || (a == :dz2 && b == :dyz)
        return SQRT3 * m * n * (n^2 - (l^2 + m^2) / 2) * dds + SQRT3 * m * n * (l^2 + m^2 - n^2) * ddp - (SQRT3 / 2) * m * n * (l^2 + m^2) * ddd
    elseif (a == :dzx && b == :dz2) || (a == :dz2 && b == :dzx)
        return SQRT3 * n * l * (n^2 - (l^2 + m^2) / 2) * dds + SQRT3 * n * l * (l^2 + m^2 - n^2) * ddp - (SQRT3 / 2) * n * l * (l^2 + m^2) * ddd
    elseif (a == :dx2y2 && b == :dz2) || (a == :dz2 && b == :dx2y2)
        return (SQRT3 / 2) * (l^2 - m^2) * (n^2 - (l^2 + m^2) / 2) * dds + SQRT3 * n^2 * (m^2 - l^2) * ddp + (SQRT3 / 4) * (1 + n^2) * (l^2 - m^2) * ddd
    elseif a == :dz2 && b == :dz2
        return (n^2 - (l^2 + m^2) / 2)^2 * dds + 3 * n^2 * (l^2 + m^2) * ddp + (3.0 / 4) * (l^2 + m^2)^2 * ddd
    end
    error("Unknown d-d orbital pair ($(orb_a), $(orb_b)).")
end

function _pp_matrix_element(orb_a::Symbol, orb_b::Symbol, l::Float64, m::Float64, n::Float64, pps::Float64, ppp::Float64)
    da, db = _pp_angular(orb_a, orb_b, l, m, n)
    return da * db * pps + (Float64(orb_a == orb_b) - da * db) * ppp
end

"""
    ORBITAL_GROUPS::Dict{Symbol,Vector{Symbol}}

Maps a group label (e.g. :p, :d) to its member orbitals, in canonical
order. Used by `shop_onsite` to expand group-level onsite fields.
"""
const ORBITAL_GROUPS = Dict{Symbol,Vector{Symbol}}(
    :s => [:s],
    :e => [:e],
    :p => [:px, :py, :pz],
    :d => [:dxy, :dyz, :dzx, :dx2y2, :dz2],
)

"""
    shop_onsite(raw::Dict, species::String, param_file::String) -> (orbitals::Vector{Symbol}, energies::Dict{Symbol,Float64})

Generic onsite shopping utility. `raw` is the onsite sub-block for one
species (plain field=>value pairs, e.g. from TOML). Recognizes both
group-level keys (`"s"`, `"p"`, `"d"`, `"e"`) and orbital-specific keys
(`"px"`, `"py"`, `"pz"`, `"dxy"`, ...):

- A group key (e.g. `"p" = 1.2345`) sets the same value for every
  orbital in that group and includes all of them in the basis.
- A specific key (e.g. `"px" = 2.3456`) overrides the value for that
  one orbital only. If the group key was absent, the specific key
  alone includes just that orbital in the basis.
- Unrecognized keys (not in `available_orbitals()`/`ORBITAL_GROUPS`)
  trigger a warning identifying `species` and `param_file`, and are skipped.

Returns the basis orbitals (in first-encountered order) and their
energies.
"""
function shop_onsite(raw::Dict, species::String, param_file::String)
    known = available_orbitals()
    energies = Dict{Symbol,Float64}()
    orbitals = Symbol[]

    _add(orb, val) = begin
        if orb in orbitals
            energies[orb] = val
        else
            push!(orbitals, orb)
            energies[orb] = val
        end
    end

    for (field, val) in raw
        sym = Symbol(field)
        if haskey(ORBITAL_GROUPS, sym) && !haskey(known, sym)
            for orb in ORBITAL_GROUPS[sym]
                _add(orb, Float64(val))
            end
        elseif haskey(known, sym)
            _add(sym, Float64(val))
        else
            @warn "orbital $(field) of atom $(species) in parameter file $(param_file) is not recognized in SK table"
        end
    end

    return orbitals, energies
end

"""
    swap_coupling_key(field::Symbol) -> Symbol

Given an integral key (e.g. :sps, :pds, :eps), returns the key for the
same physical integral with source/target orbitals swapped (e.g.
:pss, :dps, :pes). Generic: works from the two orbital-kind letters
and bond-type letter encoded in the 3-character key.
"""
function swap_coupling_key(field::Symbol)
    s = string(field)
    length(s) == 3 || error("swap_coupling_key: malformed integral key $(field).")
    a, b, bond = s[1], s[2], s[3]
    return Symbol(string(b, a, bond))
end

"""
    sk_block(orbitals_a::Vector{Symbol}, orbitals_b::Vector{Symbol},
             d::Vector{Float64}, integ::Dict{Symbol,Float64}) -> Matrix{Float64}

Builds the full Slater-Koster hopping block between two orbital sets
(e.g. all orbitals on atom a and all orbitals on atom b) for a bond
vector `d` (Cartesian, pointing from a to b). Internally normalizes
`d` to direction cosines. Returns a `length(orbitals_a) x
length(orbitals_b)` real matrix.
"""
function sk_block(orbitals_a::Vector{Symbol}, orbitals_b::Vector{Symbol}, d::Vector{Float64}, integ::Dict{Symbol,Float64})
    dist = sqrt(sum(abs2, d))
    dist > 1e-10 || error("sk_block: zero-length bond vector.")
    l, m, n = d[1] / dist, d[2] / dist, d[3] / dist

    block = zeros(Float64, length(orbitals_a), length(orbitals_b))
    for (ia, oa) in enumerate(orbitals_a), (ib, ob) in enumerate(orbitals_b)
        block[ia, ib] = sk_matrix_element(oa, ob, l, m, n, integ)
    end
    return block
end