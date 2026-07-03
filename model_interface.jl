# ============================================================
# model_interface.jl
#
# Framework layer — the ETB model interface.
#
# This is the contract between the framework and any concrete ETB
# model implementation. The framework only ever interacts with a model
# through these functions; it never inspects a model's internal fields.
#
# A concrete model must:
#   1. Define a struct subtyping AbstractETBModel that holds whatever
#      internal state it needs (parsed parameters, precomputed hoppings, ...).
#   2. Implement `build_model(::Type{MyModel}, system::System, param_file::String) -> MyModel`
#      which parses parameters and prepares the model for use.
#   3. Implement `build_hamiltonian(model::MyModel, k::Vector{Float64}) -> Matrix{ComplexF64}`
#      which returns the Bloch Hamiltonian at k-point k.
#   4. Optionally implement `show_params(model::MyModel)` for diagnostics,
#      and `num_bands(model::MyModel)` if it differs from num_orbitals(system).
# ============================================================

"""
    AbstractETBModel

Supertype for all empirical tight-binding models. Every concrete model
must subtype this so the framework's generic orchestration code
(`solve_kpoint`, `band_structure`, ...) can operate on it polymorphically.
"""
abstract type AbstractETBModel end

"""
    build_model(::Type{M}, system::System, param_file::String) where {M<:AbstractETBModel} -> M

Interface function. Given the target model type, the physical system,
and a path to a parameter file, a concrete model implementation parses
the parameters relevant to the species/orbitals present in `system`,
precomputes whatever it needs (e.g. neighbor lists, hopping matrix
elements), and returns a fully initialized model instance.

Must be implemented by each concrete model.
"""
function build_model end

"""
    build_hamiltonian(model::AbstractETBModel, k::Vector{Float64}) -> Matrix{ComplexF64}

Interface function. Returns the Bloch Hamiltonian matrix H(k) for the
given model at k-point `k` (Cartesian reciprocal units, matching the
convention used for atomic positions).

Must be implemented by each concrete model. The returned matrix need
not be manually Hermitized — the framework symmetrizes it before
diagonalization.
"""
function build_hamiltonian end

"""
    num_bands(model::AbstractETBModel) -> Int

Number of bands (Hamiltonian matrix dimension) produced by this model.
Must be implemented by each concrete model.
"""
function num_bands end

"""
    show_params(model::AbstractETBModel)

Optional interface function for displaying a model's parsed parameters.
Concrete models are encouraged to implement this for debugging/inspection.
"""
function show_params(model::AbstractETBModel)
    println("show_params not implemented for $(typeof(model)).")
end
