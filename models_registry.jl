# ============================================================
# models_registry.jl
# ============================================================

register_model!(:OrthogonalSKModel, joinpath(@__DIR__, "models", "orthogonal_sk.jl"))
register_model!(:Jancu1998Model, joinpath(@__DIR__, "models", "Jancu1998.jl"))