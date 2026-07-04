# ============================================================
# models/jancu1998_model.jl
# Constructor-based usage: model = Jancu1998Model(system, param_file; soc=true)
# Orbitals are plain Symbols (:s, :e, :px, :py, :pz, :dxy, :dyz,
# :dzx, :dx2y2, :dz2), shopped dynamically against sk_table.jl's
# available_orbitals()/required_couplings() functions.
# ============================================================

using TOML
using LinearAlgebra

struct JancuHopEntry
    V0::Float64
    eta::Float64
end

struct JancuOnsite
    species::String
    orbitals::Vector{Symbol}
    energies::Dict{Symbol,Float64}
    delta3::Union{Float64,Nothing}
end

struct JancuPair
    species1::String
    species2::String
    a_lattice::Float64
    d0::Float64
    cutoff::Float64
    onsite1::JancuOnsite
    onsite2::JancuOnsite
    hopping::Dict{Symbol,JancuHopEntry}
end

struct JancuParams
    pairs::Dict{Tuple{String,String},JancuPair}
end

struct Jancu1998Model <: AbstractETBModel
    system::System
    params::JancuParams
    atom_species::Vector{String}
    atom_orbitals::Vector{Vector{Symbol}}
    n_bonds::Int
    soc::Bool
end

# ------------------------------------------------------------
# Parameter parsing
# ------------------------------------------------------------

function _require(raw::Dict, key::String, context::String)
    haskey(raw, key) || error("Jancu1998 parameter file: missing required field \"$(key)\" in $(context).")
    return raw[key]
end

"""
    _parse_onsite(species, raw, context, param_file) -> JancuOnsite

Delegates basis-set and onsite-energy shopping to the framework's
`shop_onsite` (sk_table.jl), which handles group-level (`p`, `d`)
and orbital-specific (`px`, `dxy`, ...) keys generically.
"""
function _parse_onsite(species::String, raw::Dict, context::String, param_file::String)
    onsite_raw = Dict(k => v for (k, v) in raw if k != "delta3")
    orbitals, energies = shop_onsite(onsite_raw, species, param_file)
    isempty(orbitals) && error("Jancu1998 parameter file: species '$(species)' in $(context) has no recognized onsite energy fields.")
    delta3 = haskey(raw, "delta3") ? Float64(raw["delta3"]) : nothing
    return JancuOnsite(species, orbitals, energies, delta3)
end

"""
    _parse_hopping_table(raw, needed, context) -> Dict{Symbol,JancuHopEntry}

Shops each required coupling key from the parameter file. Every value
(V0 and eta) must be supplied by the file; missing entries or missing
fields raise an error. No exponent or value is assumed by the model.
"""
function _parse_hopping_table(raw::Dict, needed::Set{Symbol}, context::String)
    hopping = Dict{Symbol,JancuHopEntry}()
    for field in needed
        field_str = string(field)
        haskey(raw, field_str) || error("$(context): hopping entry \"$(field_str)\" is required by the declared orbitals but is not listed in the parameter file.")
        entry = raw[field_str]
        haskey(entry, "V0") || error("$(context): hopping entry \"$(field_str)\" is missing V0.")
        haskey(entry, "eta") || error("$(context): hopping entry \"$(field_str)\" is missing eta.")
        hopping[field] = JancuHopEntry(Float64(entry["V0"]), Float64(entry["eta"]))
    end
    return hopping
end

function _parse_jancu_parameters(param_file::String)
    if !isfile(param_file)
        error("Parameter file \"$(param_file)\" not found.")
    end
    raw = TOML.parsefile(param_file)

    model_type = get(raw, "model_type", nothing)
    if model_type != "jancu1998"
        error("Parameter file incompatible. Expected model_type = 'jancu1998', found '$(model_type === nothing ? "missing" : model_type)'.")
    end

    pairs = Dict{Tuple{String,String},JancuPair}()
    for (key, block) in raw
        key == "model_type" && continue
        !(block isa Dict) && continue
        parts = split(key, "_")
        length(parts) == 2 || error("Top-level parameter block \"$(key)\" must be named as Species1_Species2.")
        sp1, sp2 = String(parts[1]), String(parts[2])

        a_lattice = Float64(_require(block, "lattice_constant", key))
        d0 = a_lattice * sqrt(3.0) / 4.0
        cutoff = Float64(get(block, "cutoff", 0.45 * a_lattice))

        onsite1 = _parse_onsite(sp1, _require(block, sp1, key), "$(key).$(sp1)", param_file)
        onsite2 = _parse_onsite(sp2, _require(block, sp2, key), "$(key).$(sp2)", param_file)

        hopping_block = _require(block, "hopping", key)
        needed = required_couplings(vcat(onsite1.orbitals, onsite2.orbitals))
        hopping = _parse_hopping_table(hopping_block, needed, "$(key).hopping")

        pairs[(sp1, sp2)] = JancuPair(sp1, sp2, a_lattice, d0, cutoff, onsite1, onsite2, hopping)
    end

    isempty(pairs) && error("Parameter file contains no valid Species1_Species2 pair blocks.")
    return JancuParams(pairs)
end

function _get_pair(params::JancuParams, sp1::String, sp2::String)
    if haskey(params.pairs, (sp1, sp2))
        return params.pairs[(sp1, sp2)], true
    elseif haskey(params.pairs, (sp2, sp1))
        return params.pairs[(sp2, sp1)], false
    else
        error("No parameters found for species pair '$(sp1)'-'$(sp2)'.")
    end
end

function _onsite_for(pair::JancuPair, species::String)
    if pair.species1 == species
        return pair.onsite1
    elseif pair.species2 == species
        return pair.onsite2
    else
        error("Species '$(species)' does not belong to pair $(pair.species1)_$(pair.species2).")
    end
end

function _mixed_onsite(params::JancuParams, sp_i::String, neighbor_species::Vector{String})
    counts = Dict{String,Int}()
    for sp_j in neighbor_species
        counts[sp_j] = get(counts, sp_j, 0) + 1
    end
    N = length(neighbor_species)

    orbitals = nothing
    delta3_sum = 0.0
    any_delta3 = false
    energies = Dict{Symbol,Float64}()

    for (sp_j, n_j) in counts
        weight = n_j / N
        pair, _ = _get_pair(params, sp_i, sp_j)
        onsite_j = _onsite_for(pair, sp_i)

        if orbitals === nothing
            orbitals = onsite_j.orbitals
        elseif orbitals != onsite_j.orbitals
            error("Species '$(sp_i)' has inconsistent orbital sets across neighbor pairs (e.g. $(sp_j)); cannot mix onsite energies.")
        end

        if onsite_j.delta3 !== nothing
            delta3_sum += weight * onsite_j.delta3
            any_delta3 = true
        end
        for orb in onsite_j.orbitals
            energies[orb] = get(energies, orb, 0.0) + weight * onsite_j.energies[orb]
        end
    end

    return JancuOnsite(sp_i, orbitals, energies, any_delta3 ? delta3_sum : nothing)
end

function _check_soc_consistency(params::JancuParams, param_file::String, soc::Bool)
    soc || return nothing
    for ((sp1, sp2), pair) in params.pairs
        pair_name = "$(sp1)_$(sp2)"
        pair.onsite1.delta3 === nothing && error("ERROR: parameter `delta3` of atom $(sp1) in pair $(pair_name) needed for SOC is not found in file $(param_file)")
        pair.onsite2.delta3 === nothing && error("ERROR: parameter `delta3` of atom $(sp2) in pair $(pair_name) needed for SOC is not found in file $(param_file)")
    end
end

function _max_cutoff(params::JancuParams, atom_species::Vector{String})
    species_set = Set(atom_species)
    cutoffs = [pair.cutoff for ((sp1, sp2), pair) in params.pairs if sp1 in species_set && sp2 in species_set]
    return maximum(cutoffs)
end

# ------------------------------------------------------------
# Constructor
# ------------------------------------------------------------

"""
    Jancu1998Model(system::System; param_file::String="models/Jancu1998.toml", soc::Bool=true)
"""
function Jancu1998Model(system::System; param_file::String="models/Jancu1998.toml", soc::Bool=true)
    params = _parse_jancu_parameters(param_file)
    _check_soc_consistency(params, param_file, soc)

    atom_species = String[string(atom.species) for atom in system.atoms]

    species_set = Set(atom_species)
    relevant_cutoffs = Float64[]
    for (sp1, sp2) in keys(params.pairs)
        if sp1 in species_set && sp2 in species_set
            push!(relevant_cutoffs, params.pairs[(sp1, sp2)].cutoff)
        end
    end
    isempty(relevant_cutoffs) && error("No parameter pairs found matching the species present in the system: $(join(species_set, ", ")).")
    cutoff = maximum(relevant_cutoffs)

    neighbor_list = [nb for nb in find_neighbors(system, cutoff, canonical=false) if nb.i != nb.j]
    isempty(neighbor_list) && error("No nearest neighbors found within cutoff=$(cutoff); check system geometry.")

    atom_orbitals = Vector{Symbol}[]
    for (i, atom) in enumerate(system.atoms)
        sp_i = atom_species[i]
        neighbor_species = [atom_species[nb.j] for nb in neighbor_list if nb.i == i]
        isempty(neighbor_species) && error("Atom $(i) (species '$(sp_i)') has no nearest neighbors within cutoff=$(cutoff).")
        onsite_i = _mixed_onsite(params, sp_i, neighbor_species)
        push!(atom_orbitals, onsite_i.orbitals)
    end

    return Jancu1998Model(system, params, atom_species, atom_orbitals, length(neighbor_list), soc)
end

function num_bands(model::Jancu1998Model)
    spin_factor = model.soc ? 2 : 1
    return spin_factor * sum(length(orbs) for orbs in model.atom_orbitals)
end

function _p_soc_block(delta3::Float64)
    lambda = delta3
    M = zeros(ComplexF64, 6, 6)
    M[1, 2] = -1im * lambda
    M[2, 1] = 1im * lambda
    M[1, 6] = lambda
    M[6, 1] = lambda
    M[2, 6] = -1im * lambda
    M[6, 2] = 1im * lambda
    M[3, 4] = -lambda
    M[4, 3] = -lambda
    M[3, 5] = 1im * lambda
    M[5, 3] = -1im * lambda
    M[4, 5] = 1im * lambda
    M[5, 4] = -1im * lambda
    return M
end

function _scaled_sk_integrals(pair::JancuPair, forward::Bool, distance::Float64)
    ratio = pair.d0 / distance
    scaled = Dict{Symbol,Float64}()
    for (field, entry) in pair.hopping
        scaled[field] = entry.V0 * ratio^entry.eta
    end
    integ = Dict{Symbol,Float64}()
    for field in keys(scaled)
        lookup = forward ? field : swap_coupling_key(field)
        haskey(scaled, lookup) && (integ[field] = scaled[lookup])
    end
    return integ
end

"""
    build_hamiltonian(model::Jancu1998Model, k::Vector{Float64}) -> Matrix{ComplexF64}
"""
function build_hamiltonian(model::Jancu1998Model, k::Vector{Float64})
    p = model.params
    soc = model.soc
    n_orb = [length(orbs) for orbs in model.atom_orbitals]
    offsets = cumsum([0; n_orb])[1:end-1]
    N_spinless = sum(n_orb)
    spin_offsets = soc ? (0, N_spinless) : (0,)
    N = soc ? 2 * N_spinless : N_spinless

    H = zeros(ComplexF64, N, N)
    cutoff = _max_cutoff(p, model.atom_species)
    neighbor_list = find_neighbors(model.system, cutoff, canonical=false)

    for (i, atom) in enumerate(model.system.atoms)
        sp_i = model.atom_species[i]
        orbitals = model.atom_orbitals[i]
        base = offsets[i]

        neighbor_species = [model.atom_species[nb.j] for nb in neighbor_list if nb.i == i]
        onsite = _mixed_onsite(p, sp_i, neighbor_species)

        for (loc, orb) in enumerate(orbitals)
            e = onsite.energies[orb]
            for so in spin_offsets
                gi = so + base + loc
                H[gi, gi] += e
            end
        end

        if soc
            p_positions = findall(o -> orbital_kind(o) == :p, orbitals)
            d3 = onsite.delta3 === nothing ? 0.0 : onsite.delta3
            if d3 != 0.0 && length(p_positions) == 3
                soc_block = _p_soc_block(d3)
                up_idx = [base + loc for loc in p_positions]
                dn_idx = [N_spinless + base + loc for loc in p_positions]
                gidx = vcat(up_idx, dn_idx)
                for (a, ga) in enumerate(gidx), (b, gb) in enumerate(gidx)
                    H[ga, gb] += soc_block[a, b]
                end
            end
        end
    end

    for nb in neighbor_list
        nb.i == nb.j && continue
        sp1, sp2 = model.atom_species[nb.i], model.atom_species[nb.j]
        orb1, orb2 = model.atom_orbitals[nb.i], model.atom_orbitals[nb.j]

        pair, forward = _get_pair(p, sp1, sp2)
        integ = _scaled_sk_integrals(pair, forward, nb.distance)
        block = sk_block(orb1, orb2, nb.d, integ)

        phase = exp(1im * dot(k, nb.d))
        base_i, base_j = offsets[nb.i], offsets[nb.j]
        n1, n2 = n_orb[nb.i], n_orb[nb.j]

        for so in spin_offsets
            gi = (so+base_i+1):(so+base_i+n1)
            gj = (so+base_j+1):(so+base_j+n2)
            H[gi, gj] .+= block .* phase
            H[gj, gi] .+= block' .* conj(phase)
        end
    end

    return H
end

function show_params(model::Jancu1998Model)
    p = model.params
    println("=== Jancu 1998 Model (soc=$(model.soc)) ===")
    for ((sp1, sp2), pair) in p.pairs
        println("Pair $(sp1)_$(sp2): a=$(pair.a_lattice) Å, d0=$(pair.d0) Å, cutoff=$(pair.cutoff) Å")
        println("  $(sp1): orbitals=$(join(pair.onsite1.orbitals, ",")), delta3=$(pair.onsite1.delta3 === nothing ? "N/A" : pair.onsite1.delta3)")
        println("  $(sp2): orbitals=$(join(pair.onsite2.orbitals, ",")), delta3=$(pair.onsite2.delta3 === nothing ? "N/A" : pair.onsite2.delta3)")
        for (field, entry) in pair.hopping
            println("    $(field): V0=$(entry.V0), eta=$(entry.eta)")
        end
    end
    println("Number of nearest-neighbor bonds: $(model.n_bonds)")
    println("=========================")
end