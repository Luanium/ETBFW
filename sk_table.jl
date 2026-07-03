# ============================================================
# sk_table.jl
#
# Framework layer — general Slater-Koster two-center integral table.
#
# Provides the universal direction-cosine algebra for building
# two-center hopping matrix elements between atomic orbitals
# {s, s*, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2} given the bond
# direction cosines (l, m, n) and the relevant two-center integrals
# (sss, sps, pps, ppp, sds, pds, pdp, dds, ddp, ddd, ...).
#
# This is model-agnostic: any model (Jancu1998, orthogonal sp3, ...)
# can call `sk_block` / `sk_matrix_element` to build its Hamiltonian
# without re-deriving the Slater-Koster angular formulas itself.
#
# Reference: J.C. Slater and G.F. Koster, Phys. Rev. 94, 1498 (1954).
# ============================================================

"""
    SKOrbital

Enumerates the atomic orbitals supported by the general Slater-Koster
table, in the canonical order used to index SK blocks.
"""
@enum SKOrbital begin
    SK_S
    SK_SSTAR
    SK_PX
    SK_PY
    SK_PZ
    SK_DXY
    SK_DYZ
    SK_DZX
    SK_DX2Y2
    SK_DZ2
end

"""
    orbital_kind(orb::SKOrbital) -> Symbol

Returns the angular-momentum family (`:s`, `:p`, or `:d`) of an orbital,
used to look up the correct two-center integral name.
"""
function orbital_kind(orb::SKOrbital)
    if orb == SK_S || orb == SK_SSTAR
        return :s
    elseif orb == SK_PX || orb == SK_PY || orb == SK_PZ
        return :p
    else
        return :d
    end
end

"""
    orbital_from_label(label::String) -> SKOrbital

Parses a string orbital label ("s", "s*", "px", "py", "pz", "dxy",
"dyz", "dzx", "dx2-y2", "dz2") into an `SKOrbital`.
"""
function orbital_from_label(label::String)
    mapping = Dict(
        "s" => SK_S, "s*" => SK_SSTAR,
        "px" => SK_PX, "py" => SK_PY, "pz" => SK_PZ,
        "dxy" => SK_DXY, "dyz" => SK_DYZ, "dzx" => SK_DZX,
        "dx2-y2" => SK_DX2Y2, "dz2" => SK_DZ2,
    )
    haskey(mapping, label) || error("Unknown SK orbital label \"$(label)\".")
    return mapping[label]
end

"""
    SKIntegrals

Two-center Slater-Koster integrals for a given (species_a, species_b)
directed pair (a: source species, b: target species; asymmetric for
polar bonds). Any field left as `nothing` is treated as zero.
Field names follow the Slater-Koster/Jancu convention where the first
orbital letter belongs to the source atom and the second to the target
atom, e.g. `sps` = <s_a| H |p_b> (sσ), `pds` = <p_a| H |d_b> (pdσ).
"""
Base.@kwdef struct SKIntegrals
    sss::Union{Float64,Nothing}     = nothing
    sps::Union{Float64,Nothing}     = nothing   # <s_a|H|p_b>  sp sigma
    pss::Union{Float64,Nothing}     = nothing   # <p_a|H|s_b>  sp sigma (reverse direction)
    sSss::Union{Float64,Nothing}    = nothing   # <s*_a|H|s_b>
    sssS::Union{Float64,Nothing}    = nothing   # <s_a|H|s*_b>
    sSsSs::Union{Float64,Nothing}   = nothing   # <s*_a|H|s*_b>
    sSps::Union{Float64,Nothing}    = nothing   # <s*_a|H|p_b>
    psSs::Union{Float64,Nothing}    = nothing   # <p_a|H|s*_b>
    pps::Union{Float64,Nothing}     = nothing
    ppp::Union{Float64,Nothing}     = nothing
    sds::Union{Float64,Nothing}     = nothing   # <s_a|H|d_b>
    dss::Union{Float64,Nothing}     = nothing   # <d_a|H|s_b>
    sSds::Union{Float64,Nothing}    = nothing   # <s*_a|H|d_b>
    dsSs::Union{Float64,Nothing}    = nothing   # <d_a|H|s*_b>
    pds::Union{Float64,Nothing}     = nothing   # <p_a|H|d_b>
    dps::Union{Float64,Nothing}     = nothing   # <d_a|H|p_b>
    pdp::Union{Float64,Nothing}     = nothing
    dpp::Union{Float64,Nothing}     = nothing
    dds::Union{Float64,Nothing}     = nothing
    ddp::Union{Float64,Nothing}     = nothing
    ddd::Union{Float64,Nothing}     = nothing
end

_z(x) = x === nothing ? 0.0 : x

# ------------------------------------------------------------
# Angular factors for d orbitals (Slater-Koster 1954, Table I-III).
# l, m, n are direction cosines of the bond vector (from source to target).
# ------------------------------------------------------------

const SQRT3 = sqrt(3.0)

"""
    sk_matrix_element(orb_a::SKOrbital, orb_b::SKOrbital, l, m, n, integrals::SKIntegrals) -> Float64

Universal Slater-Koster two-center matrix element <orb_a| H |orb_b>
for a bond with direction cosines (l, m, n) pointing from the source
atom (orbital `orb_a`) to the target atom (orbital `orb_b`), using the
supplied two-center `integrals`. Implements the full s/s*/p/d table
(Slater & Koster, 1954).
"""
function sk_matrix_element(orb_a::SKOrbital, orb_b::SKOrbital, l::Float64, m::Float64, n::Float64, integ::SKIntegrals)
    ka, kb = orbital_kind(orb_a), orbital_kind(orb_b)

    # ---------------- s / s* block ----------------
    if ka == :s && kb == :s
        if orb_a == SK_S && orb_b == SK_S
            return _z(integ.sss)
        elseif orb_a == SK_SSTAR && orb_b == SK_SSTAR
            return _z(integ.sSsSs)
        elseif orb_a == SK_SSTAR && orb_b == SK_S
            return _z(integ.sSss)
        elseif orb_a == SK_S && orb_b == SK_SSTAR
            return _z(integ.sssS)
        end
    end

    # ---------------- s / s* <-> p ----------------
    if ka == :s && kb == :p
        dc = orb_b == SK_PX ? l : orb_b == SK_PY ? m : n
        coeff = orb_a == SK_S ? _z(integ.sps) : _z(integ.sSps)
        return dc * coeff
    end
    if ka == :p && kb == :s
        dc = orb_a == SK_PX ? l : orb_a == SK_PY ? m : n
        coeff = orb_b == SK_S ? _z(integ.pss) : _z(integ.psSs)
        return dc * coeff
    end

    # ---------------- p / p ----------------
    if ka == :p && kb == :p
        return _pp_matrix_element(orb_a, orb_b, l, m, n, _z(integ.pps), _z(integ.ppp))
    end

    # ---------------- s / s* <-> d ----------------
    if ka == :s && kb == :d
        coeff = orb_a == SK_S ? _z(integ.sds) : _z(integ.sSds)
        return _sd_angular(orb_b, l, m, n) * coeff
    end
    if ka == :d && kb == :s
        coeff = orb_b == SK_S ? _z(integ.dss) : _z(integ.dsSs)
        return _sd_angular(orb_a, l, m, n) * coeff
    end

    # ---------------- p <-> d ----------------
    if ka == :p && kb == :d
        return _pd_angular(orb_a, orb_b, l, m, n, _z(integ.pds), _z(integ.pdp))
    end
    if ka == :d && kb == :p
        # <d|H|p> = <p|H|d> with source/target swapped; SK matrix elements
        # for real orbitals are symmetric under simultaneous orbital swap
        # and bond reversal (l,m,n)->(l,m,n) since d is even, p is odd:
        # use dps/dpp integrals if provided, else reciprocal identity.
        pds_ = integ.dps !== nothing ? _z(integ.dps) : _z(integ.pds)
        pdp_ = integ.dpp !== nothing ? _z(integ.dpp) : _z(integ.pdp)
        return _pd_angular(orb_b, orb_a, l, m, n, pds_, pdp_)
    end

    # ---------------- d / d ----------------
    if ka == :d && kb == :d
        return _dd_angular(orb_a, orb_b, l, m, n, _z(integ.dds), _z(integ.ddp), _z(integ.ddd))
    end

    error("Unhandled SK orbital pair ($(orb_a), $(orb_b)).")
end

# p-p angular factor (clean, correct implementation; replaces the
# placeholder pipe above which is dead code by construction order).
function _pp_angular(orb_a::SKOrbital, orb_b::SKOrbital, l::Float64, m::Float64, n::Float64)
    da = orb_a == SK_PX ? l : orb_a == SK_PY ? m : n
    db = orb_b == SK_PX ? l : orb_b == SK_PY ? m : n
    return da, db
end

function _sd_angular(d_orb::SKOrbital, l::Float64, m::Float64, n::Float64)
    if d_orb == SK_DXY
        return SQRT3 * l * m
    elseif d_orb == SK_DYZ
        return SQRT3 * m * n
    elseif d_orb == SK_DZX
        return SQRT3 * n * l
    elseif d_orb == SK_DX2Y2
        return (SQRT3 / 2) * (l^2 - m^2)
    elseif d_orb == SK_DZ2
        return n^2 - (l^2 + m^2) / 2
    end
    error("Unknown d orbital $(d_orb).")
end

function _pd_angular(p_orb::SKOrbital, d_orb::SKOrbital, l::Float64, m::Float64, n::Float64, pds::Float64, pdp::Float64)
    dc = p_orb == SK_PX ? l : p_orb == SK_PY ? m : n
    others = p_orb == SK_PX ? (m, n) : p_orb == SK_PY ? (l, n) : (l, m)
    a, b = others

    if d_orb == SK_DXY
        if p_orb == SK_PX
            return SQRT3 * l^2 * m * pds + m * (1 - 2 * l^2) * pdp
        elseif p_orb == SK_PY
            return SQRT3 * m^2 * l * pds + l * (1 - 2 * m^2) * pdp
        else
            return SQRT3 * l * m * n * pds - 2 * l * m * n * pdp
        end
    elseif d_orb == SK_DYZ
        if p_orb == SK_PX
            return SQRT3 * l * m * n * pds - 2 * l * m * n * pdp
        elseif p_orb == SK_PY
            return SQRT3 * m^2 * n * pds + n * (1 - 2 * m^2) * pdp
        else
            return SQRT3 * n^2 * m * pds + m * (1 - 2 * n^2) * pdp
        end
    elseif d_orb == SK_DZX
        if p_orb == SK_PX
            return SQRT3 * l^2 * n * pds + n * (1 - 2 * l^2) * pdp
        elseif p_orb == SK_PY
            return SQRT3 * l * m * n * pds - 2 * l * m * n * pdp
        else
            return SQRT3 * n^2 * l * pds + l * (1 - 2 * n^2) * pdp
        end
    elseif d_orb == SK_DX2Y2
        if p_orb == SK_PX
            return (SQRT3 / 2) * l * (l^2 - m^2) * pds + l * (1 - l^2 + m^2) * pdp
        elseif p_orb == SK_PY
            return (SQRT3 / 2) * m * (l^2 - m^2) * pds - m * (1 + l^2 - m^2) * pdp
        else
            return (SQRT3 / 2) * n * (l^2 - m^2) * pds - n * (l^2 - m^2) * pdp
        end
    elseif d_orb == SK_DZ2
        if p_orb == SK_PX
            return l * (n^2 - (l^2 + m^2) / 2) * pds - SQRT3 * l * n^2 * pdp
        elseif p_orb == SK_PY
            return m * (n^2 - (l^2 + m^2) / 2) * pds - SQRT3 * m * n^2 * pdp
        else
            return n * (n^2 - (l^2 + m^2) / 2) * pds + SQRT3 * n * (l^2 + m^2) * pdp
        end
    end
    error("Unknown d orbital $(d_orb).")
end

function _dd_angular(orb_a::SKOrbital, orb_b::SKOrbital, l::Float64, m::Float64, n::Float64, dds::Float64, ddp::Float64, ddd::Float64)
    a, b = orb_a, orb_b
    # Ensure symmetric handling regardless of argument order for identical-kind pairs
    if a == SK_DXY && b == SK_DXY
        return 3 * l^2 * m^2 * dds + (l^2 + m^2 - 4 * l^2 * m^2) * ddp + (n^2 + l^2 * m^2) * ddd
    elseif (a == SK_DXY && b == SK_DYZ) || (a == SK_DYZ && b == SK_DXY)
        return 3 * l * m^2 * n * dds + l * n * (1 - 4 * m^2) * ddp + l * n * (m^2 - 1) * ddd
    elseif (a == SK_DXY && b == SK_DZX) || (a == SK_DZX && b == SK_DXY)
        return 3 * l^2 * m * n * dds + m * n * (1 - 4 * l^2) * ddp + m * n * (l^2 - 1) * ddd
    elseif a == SK_DYZ && b == SK_DYZ
        return 3 * m^2 * n^2 * dds + (m^2 + n^2 - 4 * m^2 * n^2) * ddp + (l^2 + m^2 * n^2) * ddd
    elseif (a == SK_DYZ && b == SK_DZX) || (a == SK_DZX && b == SK_DYZ)
        return 3 * m * n^2 * l * dds + m * l * (1 - 4 * n^2) * ddp + m * l * (n^2 - 1) * ddd
    elseif a == SK_DZX && b == SK_DZX
        return 3 * n^2 * l^2 * dds + (n^2 + l^2 - 4 * n^2 * l^2) * ddp + (m^2 + n^2 * l^2) * ddd
    elseif (a == SK_DXY && b == SK_DX2Y2) || (a == SK_DX2Y2 && b == SK_DXY)
        return (3.0 / 2) * l * m * (l^2 - m^2) * dds + 2 * l * m * (m^2 - l^2) * ddp + (l * m * (l^2 - m^2) / 2) * ddd
    elseif (a == SK_DYZ && b == SK_DX2Y2) || (a == SK_DX2Y2 && b == SK_DYZ)
        return (3.0 / 2) * m * n * (l^2 - m^2) * dds - m * n * (1 + 2 * (l^2 - m^2)) * ddp + m * n * (1 + (l^2 - m^2) / 2) * ddd
    elseif (a == SK_DZX && b == SK_DX2Y2) || (a == SK_DX2Y2 && b == SK_DZX)
        return (3.0 / 2) * n * l * (l^2 - m^2) * dds + n * l * (1 - 2 * (l^2 - m^2)) * ddp - n * l * (1 - (l^2 - m^2) / 2) * ddd
    elseif a == SK_DX2Y2 && b == SK_DX2Y2
        return (3.0 / 4) * (l^2 - m^2)^2 * dds + (l^2 + m^2 - (l^2 - m^2)^2) * ddp + (n^2 + (l^2 - m^2)^2 / 4) * ddd
    elseif (a == SK_DXY && b == SK_DZ2) || (a == SK_DZ2 && b == SK_DXY)
        return SQRT3 * l * m * (n^2 - (l^2 + m^2) / 2) * dds - 2 * SQRT3 * l * m * n^2 * ddp + (SQRT3 / 2) * l * m * (1 + n^2) * ddd
    elseif (a == SK_DYZ && b == SK_DZ2) || (a == SK_DZ2 && b == SK_DYZ)
        return SQRT3 * m * n * (n^2 - (l^2 + m^2) / 2) * dds + SQRT3 * m * n * (l^2 + m^2 - n^2) * ddp - (SQRT3 / 2) * m * n * (l^2 + m^2) * ddd
    elseif (a == SK_DZX && b == SK_DZ2) || (a == SK_DZ2 && b == SK_DZX)
        return SQRT3 * n * l * (n^2 - (l^2 + m^2) / 2) * dds + SQRT3 * n * l * (l^2 + m^2 - n^2) * ddp - (SQRT3 / 2) * n * l * (l^2 + m^2) * ddd
    elseif (a == SK_DX2Y2 && b == SK_DZ2) || (a == SK_DZ2 && b == SK_DX2Y2)
        return (SQRT3 / 2) * (l^2 - m^2) * (n^2 - (l^2 + m^2) / 2) * dds + SQRT3 * n^2 * (m^2 - l^2) * ddp + (SQRT3 / 4) * (1 + n^2) * (l^2 - m^2) * ddd
    elseif a == SK_DZ2 && b == SK_DZ2
        return (n^2 - (l^2 + m^2) / 2)^2 * dds + 3 * n^2 * (l^2 + m^2) * ddp + (3.0 / 4) * (l^2 + m^2)^2 * ddd
    end
    error("Unknown d-d orbital pair ($(orb_a), $(orb_b)).")
end

function _pp_matrix_element(orb_a::SKOrbital, orb_b::SKOrbital, l::Float64, m::Float64, n::Float64, pps::Float64, ppp::Float64)
    da, db = _pp_angular(orb_a, orb_b, l, m, n)
    return da * db * pps + (Float64(orb_a == orb_b) - da * db) * ppp
end

"""
    sk_block(orbitals_a::Vector{SKOrbital}, orbitals_b::Vector{SKOrbital},
             d::Vector{Float64}, integrals::SKIntegrals) -> Matrix{Float64}

Builds the full Slater-Koster hopping block between two orbital sets
(e.g. all orbitals on atom a and all orbitals on atom b) for a bond
vector `d` (Cartesian, pointing from a to b). Internally normalizes
`d` to direction cosines. Returns a `length(orbitals_a) x
length(orbitals_b)` real matrix.
"""
function sk_block(orbitals_a::Vector{SKOrbital}, orbitals_b::Vector{SKOrbital}, d::Vector{Float64}, integrals::SKIntegrals)
    dist = sqrt(sum(abs2, d))
    dist > 1e-10 || error("sk_block: zero-length bond vector.")
    l, m, n = d[1] / dist, d[2] / dist, d[3] / dist

    block = zeros(Float64, length(orbitals_a), length(orbitals_b))
    for (ia, oa) in enumerate(orbitals_a), (ib, ob) in enumerate(orbitals_b)
        block[ia, ib] = sk_matrix_element(oa, ob, l, m, n, integrals)
    end
    return block
end
