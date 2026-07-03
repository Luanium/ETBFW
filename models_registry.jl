# ============================================================
# models_registry.jl
#
# The single place that lists all available ETB models. Include this
# once at startup (after framework.jl). Adding a new model means
# writing models/your_model.jl and adding one line here — no other
# framework code needs to change.
#
# This only registers lightweight metadata (name, type symbol, file
# path); the model's source file itself is include()'d lazily by
# load_etb() the first time that model is actually requested.
# ============================================================

register_model!("orthogonal_sk", :OrthogonalSKModel, joinpath(@__DIR__, "models", "orthogonal_sk_model.jl"))
register_model!("jancu1998", :Jancu1998Model, joinpath(@__DIR__, "models", "jancu1998_model.jl"))

# To add another model:
# register_model!("your_model_name", :YourModelType, joinpath(@__DIR__, "models", "your_model.jl"))
