# ============================================================
# models/jancu1998_model.jl
#
# Model layer — J.-M. Jancu, R. Scholz, F. Beltram, F. Bassani,
# "Empirical spds* tight-binding calculation for cubic semiconductor
# structures", Phys. Rev. B 57, 6493 (1998).
#
# sp3d5s* nearest-neighbor empirical TB model for zinc-blende /
# diamond structures, with two sublattices ("a" = cation/atom-1,
# "c" = anion/atom-2), spin-orbit coupling on the p states, and
# on-site energies + two-center SK integrals as tabulated in the
# paper (Tables II, III).
#
# 10 spatial orbitals per site: s, s*, px, py, pz, dxy, dyz, dzx,
# dx2-y2, dz2. With spin, the full Hamiltonian per k-point is 40x40
# (2 atoms x 10 orbitals x 2 spins). Nearest-neighbor only (4 bonds
# per atom in the diamond/zinc-blende lattice).
#
# This model only uses the framework interface (model_interface.jl)
# and the general Slater-Koster utility (sk_table.jl); it builds no
# angular-momentum algebra of its own.
# ============================================================

using TOML
using LinearAlgebra

# Canonical 10-orbital basis order used throughout this model.
const JANCU_ORBITALS = [SK_S, SK_SSTAR, SK_PX, SK_PY, SK_PZ, SK_DXY, SK_DYZ, SK_DZX, SK_DX2Y2, SK_DZ2]
const JANCU_NORB = length(JANCU_ORBITALS)   # 10

struct JancuOnsite
    Es::Float64
    Ep::Float64
    Ed::Float64
    Esstar::Float64
    delta3::Float64   # spin-orbit parameter Δ/3 for the p shell on this sublattice
end

"""
    JancuParams

Parsed sp3d5s* parameters for a two-sublattice (a, c) zinc-blende /
diamond material: lattice constant, energy unit, on-site energies per
sublattice, and the directed two-center SK integrals for the a->c bond
(the c->a integrals are obtained via the reversed-bond SK convention).
"""
struct JancuParams
    a_lattice::Float64
    onsite_a::JancuOnsite
    onsite_c::JancuOnsite
    integrals_ac::SKIntegrals   # source = a, target = c
    cutoff::Float64
end

struct JancuBond
    R::Vector{Int}          # lattice translation (integers, in units of conventional lattice vectors)
    d::Vector{Float64}      # Cartesian bond vector from atom a to atom c (periodic image)
end

struct Jancu1998Model <: AbstractETBModel
    system::System
    params::JancuParams
    bonds::Vector{JancuBond}   # the 4 nearest-neighbor a->c bonds (zinc-blende/diamond)
end

# ------------------------------------------------------------
# Parameter parsing
# ------------------------------------------------------------

function _parse_onsite(raw::Dict)
    JancuOnsite(
        Float64(raw["Es"]), Float64(raw["Ep"]), Float64(raw["Ed"]),
        Float64(raw["Esstar"]), Float64(get(raw, "delta3", 0.0)),
    )
end

function _parse_integrals(raw::Dict)
    SKIntegrals(
        sss   = get(raw, "sss", nothing),
        sps   = get(raw, "sps", nothing),      # <s_a|H|p_c>
        pss   = get(raw, "pss", nothing),      # <p_a|H|s_c>
        sSss  = get(raw, "sSss", nothing),     # <s*_a|H|s_c>
        sssS  = get(raw, "sssS", nothing),     # <s_a|H|s*_c>
        sSsSs = get(raw, "sSsSs", nothing),
        sSps  = get(raw, "sSps", nothing),     # <s*_a|H|p_c>
        psSs  = get(raw, "psSs", nothing),     # <p_a|H|s*_c>
        pps   = get(raw, "pps", nothing),
        ppp   = get(raw, "ppp", nothing),
        sds   = get(raw, "sds", nothing),      # <s_a|H|d_c>
        dss   = get(raw, "dss", nothing),      # <d_a|H|s_c>
        sSds  = get(raw, "sSds", nothing),     # <s*_a|H|d_c>
        dsSs  = get(raw, "dsSs", nothing),     # <d_a|H|s*_c>
        pds   = get(raw, "pds", nothing),      # <p_a|H|d_c>
        dps   = get(raw, "dps", nothing),      # <d_a|H|p_c>
        pdp   = get(raw, "pdp", nothing),
        dpp   = get(raw, "dpp", nothing),
        dds   = get(raw, "dds", nothing),
        ddp   = get(raw, "ddp", nothing),
        ddd   = get(raw, "ddd", nothing),
    )
end

"""
    parse_parameters(::Type{Jancu1998Model}, system::System, param_file::String) -> JancuParams
"""
function parse_parameters(::Type{Jancu1998Model}, system::System, param_file::String)
    if !isfile(param_file)
        error("Parameter file \"$(param_file)\" not found.")
    end
    raw = TOML.parsefile(param_file)

    model_type = get(raw, "model_type", nothing)
    if model_type != "jancu1998"
        error("Parameter file incompatible. Expected model_type = 'jancu1998', found '$(model_type === nothing ? "missing" : model_type)'.")
    end

    a_lattice = Float64(raw["lattice_constant"])
    onsite_a = _parse_onsite(raw["onsite"]["a"])
    onsite_c = _parse_onsite(raw["onsite"]["c"])
    integrals_ac = _parse_integrals(raw["hopping"])
    cutoff = Float64(get(raw, "cutoff", 0.45 * a_lattice))

    return JancuParams(a_lattice, onsite_a, onsite_c, integrals_ac, cutoff)
end

# ------------------------------------------------------------
# Nearest-neighbor bonds (zinc-blende / diamond, 4 bonds a->c)
# ------------------------------------------------------------

"""
    _find_ac_bonds(system::System, cutoff::Float64) -> Vector{JancuBond}

Uses the framework's generic `find_neighbors` to discover the a->c
(atom 1 -> atom 2) nearest-neighbor bonds. Assumes `system` has exactly
two atoms per unit cell (the zinc-blende / diamond basis), consistent
with the two-sublattice Jancu1998 model.
"""
function _find_ac_bonds(system::System, cutoff::Float64)
    length(system.atoms) == 2 || error("Jancu1998Model requires a two-atom (zinc-blende/diamond) basis; got $(length(system.atoms)) atoms.")

    neighbor_list = find_neighbors(system, cutoff, canonical=false)
    bonds = JancuBond[]
    for nb in neighbor_list
        if nb.i == 1 && nb.j == 2
            push!(bonds, JancuBond(nb.R, nb.d))
        end
    end
    isempty(bonds) && error("No a->c nearest neighbors found within cutoff=$(cutoff); check system geometry.")
    return bonds
end

# ------------------------------------------------------------
# Interface implementation
# ------------------------------------------------------------

"""
    build_model(::Type{Jancu1998Model}, system::System, param_file::String) -> Jancu1998Model
"""
function build_model(::Type{Jancu1998Model}, system::System, param_file::String)
    params = parse_parameters(Jancu1998Model, system, param_file)
    bonds = _find_ac_bonds(system, params.cutoff)
    return Jancu1998Model(system, params, bonds)
end

"""
    num_bands(model::Jancu1998Model) -> Int

40 = 2 atoms x 10 orbitals (s,s*,px,py,pz,dxy,dyz,dzx,dx2-y2,dz2) x 2 spins.
"""
num_bands(model::Jancu1998Model) = 2 * JANCU_NORB * 2

# Spin-orbit L.S matrix on the {px,py,pz} block (real orbital basis),
# block-diagonal-free 6x6 form ordered (px↑,py↑,pz↑,px↓,py↓,pz↓).
# Standard p-orbital spin-orbit coupling matrix (Chadi 1977 convention).
function _p_soc_block(delta3::Float64)
    # λ = Δ/3 already in the parameter convention used by Jancu (Table II/III: "D/3").
    λ = delta3
    M = zeros(ComplexF64, 6, 6)
    # order: 1=px↑ 2=py↑ 3=pz↑ 4=px↓ 5=py↓ 6=pz↓
    M[1, 2] = -1im * λ; M[2, 1] = 1im * λ
    M[1, 6] = λ;        M[6, 1] = λ
    M[2, 6] = -1im * λ; M[6, 2] = 1im * λ
    M[3, 4] = -λ;       M[4, 3] = -λ
    M[3, 5] = 1im * λ;  M[5, 3] = -1im * λ
    M[4, 5] = 1im * λ;  M[5, 4] = -1im * λ
    return M
end

"""
    build_hamiltonian(model::Jancu1998Model, k::Vector{Float64}) -> Matrix{ComplexF64}

Assembles the 40x40 Bloch Hamiltonian: on-site blocks (10x10 per atom,
per spin, spin-degenerate except for the p-shell spin-orbit term) plus
nearest-neighbor a<->c hopping blocks built via the framework's
`sk_block` utility, Bloch-summed over the 4 nearest-neighbor bonds.
"""
function build_hamiltonian(model::Jancu1998Model, k::Vector{Float64})
    p = model.params
    N = JANCU_NORB
    Nspin = 2 * N
    H = zeros(ComplexF64, 2 * Nspin, 2 * Nspin)  # [atom][spin][orbital] blocks: a-up,a-dn,c-up,c-dn each N

    # Spinless on-site blocks (diagonal in orbital)
    function onsite_diag(os::JancuOnsite)
        d = zeros(Float64, N)
        d[1] = os.Es
        d[2] = os.Esstar
        d[3] = d[4] = d[5] = os.Ep
        d[6] = d[7] = d[8] = d[9] = d[10] = os.Ed
        return d
    end

    diag_a = onsite_diag(p.onsite_a)
    diag_c = onsite_diag(p.onsite_c)

    # Block layout: indices 1:N = atom a spin up, N+1:2N = atom a spin down,
    # 2N+1:3N = atom c spin up, 3N+1:4N = atom c spin down.
    idx_a_up = 1:N
    idx_a_dn = (N+1):(2N)
    idx_c_up = (2N+1):(3N)
    idx_c_dn = (3N+1):(4N)

    for (r, idx) in ((diag_a, idx_a_up), (diag_a, idx_a_dn), (diag_c, idx_c_up), (diag_c, idx_c_dn))
        for (loc, gi) in enumerate(idx)
            H[gi, gi] += r[loc]
        end
    end

    # Spin-orbit coupling on p-shell (orbitals 3,4,5 = px,py,pz), atom a and atom c independently.
    function apply_soc!(H, idx_up, idx_dn, delta3::Float64)
        delta3 == 0.0 && return
        soc = _p_soc_block(delta3)
        p_up = [idx_up[3], idx_up[4], idx_up[5]]
        p_dn = [idx_dn[3], idx_dn[4], idx_dn[5]]
        gidx = vcat(p_up, p_dn)  # matches soc's (px↑,py↑,pz↑,px↓,py↓,pz↓) ordering
        for (a, ga) in enumerate(gidx), (b, gb) in enumerate(gidx)
            H[ga, gb] += soc[a, b]
        end
    end
    apply_soc!(H, idx_a_up, idx_a_dn, p.onsite_a.delta3)
    apply_soc!(H, idx_c_up, idx_c_dn, p.onsite_c.delta3)

    # Nearest-neighbor a<->c hopping, Bloch-summed over the 4 bonds.
    # Forward direction a->c uses integrals_ac directly via sk_block.
    # Reverse direction c->a uses the SK bond-reversal identity: for a
    # bond vector -d (c->a), s/p/d orbitals transform with definite
    # parity, so <c|H|a>(-d) = <a|H|c>(d)^T with p-type source/target
    # orbitals picking up a sign from direction cosines already handled
    # internally by sk_block since it re-derives cosines from -d.
    Hac_up = zeros(ComplexF64, N, N)
    Hac_dn = zeros(ComplexF64, N, N)
    for bond in model.bonds
        phase = exp(1im * dot(k, bond.d))
        block = sk_block(JANCU_ORBITALS, JANCU_ORBITALS, bond.d, p.integrals_ac)
        Hac_up .+= block .* phase
        Hac_dn .+= block .* phase
    end

    H[idx_a_up, idx_c_up] .+= Hac_up
    H[idx_a_dn, idx_c_dn] .+= Hac_dn
    H[idx_c_up, idx_a_up] .+= Hac_up'
    H[idx_c_dn, idx_a_dn] .+= Hac_dn'

    return H
end

"""
    show_params(model::Jancu1998Model)
"""
function show_params(model::Jancu1998Model)
    p = model.params
    println("=== Jancu 1998 spds* Model Parameters ===")
    println("Lattice constant a = $(p.a_lattice) Å, cutoff = $(p.cutoff) Å")
    println("Onsite (a): Es=$(p.onsite_a.Es) Ep=$(p.onsite_a.Ep) Ed=$(p.onsite_a.Ed) Es*=$(p.onsite_a.Esstar) Δ/3=$(p.onsite_a.delta3)")
    println("Onsite (c): Es=$(p.onsite_c.Es) Ep=$(p.onsite_c.Ep) Ed=$(p.onsite_c.Ed) Es*=$(p.onsite_c.Esstar) Δ/3=$(p.onsite_c.delta3)")
    println("Two-center integrals (a->c):")
    for f in fieldnames(SKIntegrals)
        v = getfield(p.integrals_ac, f)
        v !== nothing && println("  $(f) = $(v)")
    end
    println("Number of nearest-neighbor bonds: $(length(model.bonds))")
    println("===========================================")
end

# NOTE: this model registers itself with the framework in
# models_registry.jl (included once at startup), not here — this file
# is include()'d lazily by load_etb() on first use.
