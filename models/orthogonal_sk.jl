# ============================================================
# models/orthogonal_sk_model.jl
#
# Model layer — example concrete ETB model.
#
# A generic orthogonal empirical tight-binding model: arbitrary
# species and orbitals (as declared in the System), onsite energies
# per (species, orbital), and pairwise hoppings per
# (species, orbital, species, orbital) that decay exponentially with
# distance:
#
#       t(d) = t0 * exp( -beta * (d - d0) )
#
# This mirrors the common empirical-TB convention (e.g. Slater-Koster
# two-center integrals with Harrison-type distance scaling) without
# committing to any specific material or orbital symmetry beyond what
# is declared in the parameter file. It only implements the framework
# interface defined in model_interface.jl and never reaches outside it.
# ============================================================

using TOML

struct HoppingRule
    t0::Float64
    d0::Float64
    beta::Float64
end

struct OrthogonalSKParams
    species_orbitals::Dict{Symbol,Vector{String}}
    onsite::Dict{Tuple{Symbol,String},Float64}
    hopping_rules::Dict{Tuple{Symbol,String,Symbol,String},HoppingRule}
    cutoff::Float64
end

struct PreparedHopping
    atom_from::Int
    atom_to::Int
    gi::Int
    gj::Int
    R::Vector{Int}
    t::ComplexF64
end

struct OrthogonalSKModel <: AbstractETBModel
    system::System
    params::OrthogonalSKParams
    num_bands::Int
    onsite_energies::Dict{Int,Float64}   # gi -> energy
    hoppings::Vector{PreparedHopping}
end

num_bands(model::OrthogonalSKModel) = model.num_bands

# ------------------------------------------------------------
# Parameter parsing
# ------------------------------------------------------------

"""
    parse_parameters(::Type{OrthogonalSKModel}, system::System, param_file::String) -> OrthogonalSKParams

Reads the TOML parameter file and extracts only the onsite/hopping
entries relevant to the species and orbitals actually present in
`system`. Errors if required entries are missing.
"""
function parse_parameters(::Type{OrthogonalSKModel}, system::System, param_file::String)
    if !isfile(param_file)
        error("Parameter file \"$(param_file)\" not found.")
    end
    raw = TOML.parsefile(param_file)

    model_type = get(raw, "model_type", nothing)
    if model_type != "orthogonal_sk"
        error("Parameter file incompatible. Expected model_type = 'orthogonal_sk', found '$(model_type === nothing ? "missing" : model_type)'.")
    end

    raw_onsite = get(raw, "onsite", Dict())
    species_orbitals = Dict{Symbol,Vector{String}}()
    for atom in system.atoms
        species = atom.species
        if !haskey(species_orbitals, species)
            key = string(species)
            if !haskey(raw_onsite, key)
                error("Onsite block for species :$(species) not found in parameter file.")
            end
            species_orbitals[species] = sort(collect(keys(raw_onsite[key])))
        end
    end
    present_species = collect(keys(species_orbitals))

    onsite_dict = Dict{Tuple{Symbol,String},Float64}()
    for (species, orbs) in species_orbitals
        key = string(species)
        for orb in orbs
            onsite_dict[(species, orb)] = Float64(raw_onsite[key][orb])
        end
    end

    hopping_dict = Dict{Tuple{Symbol,String,Symbol,String},HoppingRule}()
    raw_hopping = get(raw, "hopping", Dict())
    cutoff = Float64(get(raw_hopping, "cutoff", 0.0))

    species_list = collect(present_species)
    for a in 1:length(species_list), b in a:length(species_list)
        sp_i, sp_j = species_list[a], species_list[b]
        pair_key, rev_key = "$(sp_i)_$(sp_j)", "$(sp_j)_$(sp_i)"

        pair_data, reversed = nothing, false
        if haskey(raw_hopping, pair_key)
            pair_data = raw_hopping[pair_key]
        elseif haskey(raw_hopping, rev_key)
            pair_data, reversed = raw_hopping[rev_key], true
        else
            error("Hopping block for species pair :$(sp_i)-:$(sp_j) not found.")
        end

        for o1 in species_orbitals[sp_i], o2 in species_orbitals[sp_j]
            primary = reversed ? "$(o2)-$(o1)" : "$(o1)-$(o2)"
            fallback = reversed ? "$(o1)-$(o2)" : "$(o2)-$(o1)"

            rule_data = if haskey(pair_data, primary)
                pair_data[primary]
            elseif sp_i == sp_j && haskey(pair_data, fallback)
                pair_data[fallback]
            else
                error("Hopping rule for orbitals \"$(o1)\"-\"$(o2)\" between :$(sp_i)-:$(sp_j) not found.")
            end

            rule = HoppingRule(Float64(rule_data["t0"]), Float64(rule_data["d0"]), Float64(get(rule_data, "beta", 0.0)))
            hopping_dict[(sp_i, o1, sp_j, o2)] = rule
            hopping_dict[(sp_j, o2, sp_i, o1)] = rule
        end
    end

    return OrthogonalSKParams(species_orbitals, onsite_dict, hopping_dict, cutoff)
end

# ------------------------------------------------------------
# Interface implementation
# ------------------------------------------------------------

"""
    build_model(::Type{OrthogonalSKModel}, system::System, param_file::String) -> OrthogonalSKModel

Implements the framework's `build_model` interface: parses parameters,
uses the framework's `find_neighbors` utility to discover in-range atom
pairs, and precomputes hopping matrix elements via exponential distance
scaling.
"""
function build_model(::Type{OrthogonalSKModel}, system::System, param_file::String)
    params = parse_parameters(OrthogonalSKModel, system, param_file)

    idx_map = Dict{Tuple{Int,String},Int}()
    idx = 1
    for (i, atom) in enumerate(system.atoms)
        for orb in params.species_orbitals[atom.species]
            idx_map[(i, orb)] = idx
            idx += 1
        end
    end
    num_bands_val = idx - 1

    onsite_energies = Dict{Int,Float64}()
    for (i, atom) in enumerate(system.atoms)
        for orb in params.species_orbitals[atom.species]
            gi = idx_map[(i, orb)]
            onsite_energies[gi] = params.onsite[(atom.species, orb)]
        end
    end

    hoppings = PreparedHopping[]
    neighbor_list = find_neighbors(system, params.cutoff, canonical=true)

    for nb in neighbor_list
        atom_i, atom_j = system.atoms[nb.i], system.atoms[nb.j]
        for orb_i in params.species_orbitals[atom_i.species], orb_j in params.species_orbitals[atom_j.species]
            rule = params.hopping_rules[(atom_i.species, orb_i, atom_j.species, orb_j)]
            t = rule.t0 * exp(-rule.beta * (nb.distance - rule.d0))
            gi = idx_map[(nb.i, orb_i)]
            gj = idx_map[(nb.j, orb_j)]
            push!(hoppings, PreparedHopping(nb.i, nb.j, gi, gj, nb.R, ComplexF64(t)))
        end
    end

    return OrthogonalSKModel(system, params, num_bands_val, onsite_energies, hoppings)
end

"""
    build_hamiltonian(model::OrthogonalSKModel, k::Vector{Float64}) -> Matrix{ComplexF64}

Implements the framework's `build_hamiltonian` interface.
"""
function build_hamiltonian(model::OrthogonalSKModel, k::Vector{Float64})
    system = model.system
    N = model.num_bands
    H = zeros(ComplexF64, N, N)

    for (gi, E) in model.onsite_energies
        H[gi, gi] += E
    end

    for hop in model.hoppings
        gi, gj = hop.gi, hop.gj

        dR = get_dR(system.lattice, hop.R)
        pos_from = system.atoms[hop.atom_from].position
        pos_to = system.atoms[hop.atom_to].position
        d = dR + (pos_to - pos_from)

        phase = exp(1im * dot(k, d))
        H[gi, gj] += hop.t * phase

        if gi != gj || hop.R != zeros(Int, length(hop.R))
            H[gj, gi] += conj(hop.t * phase)
        end
    end

    return H
end

"""
    show_params(model::OrthogonalSKModel)

Implements the optional `show_params` interface hook.
"""
function show_params(model::OrthogonalSKModel)
    p = model.params
    println("=== Orthogonal SK Model Parameters ===")
    println("Onsite energies:")
    for (k, v) in p.onsite
        println("  species=:$(k[1]) orbital=$(k[2]) => $(v) eV")
    end
    println("Hopping rules (cutoff = $(p.cutoff) Å):")
    printed = Set{Tuple{Symbol,String,Symbol,String}}()
    for (key, rule) in p.hopping_rules
        rev = (key[3], key[4], key[1], key[2])
        if !(key in printed) && !(rev in printed)
            println("  :$(key[1]).$(key[2]) <-> :$(key[3]).$(key[4]) : t0=$(rule.t0), d0=$(rule.d0), beta=$(rule.beta)")
            push!(printed, key)
        end
    end
    println("========================================")
end

# NOTE: this model registers itself with the framework in
# models_registry.jl (included once at startup), not here — this file
# is include()'d lazily by load_etb() on first use, and we want
# registration metadata available before that happens.
