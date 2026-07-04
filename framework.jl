# ============================================================
# framework.jl
#
# Single entry point for the ETB framework layer. Include this file
# to get access to System building, neighbor search, the generic
# Slater-Koster table, the model interface, the model registry, the
# solver, and plotting utilities.
#
# Concrete models (e.g. models/orthogonal_sk_model.jl) are NOT included
# here — they are registered by name and loaded lazily on demand via
# `load_etb`, so adding a new model never requires touching this file.
# ============================================================

using LinearAlgebra

include("system.jl")
include("crystals.jl")
include("neighbors.jl")
include("sk_table.jl")
include("model_interface.jl")
include("registry.jl")
include("models_registry.jl")
include("solver.jl")
include("plotting.jl")
