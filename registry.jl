# ============================================================
# registry.jl
#
# Framework layer — model registry.
#
# Rather than hardcoding an if/elseif chain of known model names inside
# the framework (which would require editing framework code every time
# a new model is added), models register themselves under a string
# name. The framework then loads/instantiates models purely by name.
# ============================================================

const MODEL_REGISTRY = Dict{String,String}()        # name => source_file
const MODEL_TYPE_BINDINGS = Dict{String,Symbol}()    # name => type symbol, e.g. :OrthogonalSKModel

"""
    register_model!(name::String, type_symbol::Symbol, source_file::String)

Registers a concrete model under `name`, associating it with the
symbol naming its concrete type (e.g. `:OrthogonalSKModel`) and the
Julia source file that defines it. Registration only records this
metadata; the file is `include`d lazily on first use by `load_etb`.
"""
function register_model!(name::String, type_symbol::Symbol, source_file::String)
    MODEL_REGISTRY[name] = source_file
    MODEL_TYPE_BINDINGS[name] = type_symbol
    return nothing
end

"""
    load_etb(system::System, model_name::String, param_file::String) -> AbstractETBModel

Framework entry point: includes the source file registered under
`model_name` (defining its concrete model type and `build_model`
method), then calls `build_model` to parse `param_file` against
`system` and return the initialized model — ready to be handed to
`solve_kpoint` / `band_structure`.
"""
function load_etb(system::System, model_name::String, param_file::String)
    if !haskey(MODEL_REGISTRY, model_name)
        available = isempty(MODEL_REGISTRY) ? "(none registered yet)" : join(keys(MODEL_REGISTRY), ", ")
        error("Model '$(model_name)' is not registered. Available models: $available")
    end

    source_file = MODEL_REGISTRY[model_name]
    include(source_file)

    # invokelatest avoids world-age issues since source_file was just include()'d
    build_fn = getfield(@__MODULE__, :build_model)
    model_type = getfield(@__MODULE__, MODEL_TYPE_BINDINGS[model_name])
    return Base.invokelatest(build_fn, model_type, system, param_file)
end
