# ============================================================
# registry.jl
#
# Framework layer — model registry.
# ============================================================

const MODEL_REGISTRY = Dict{Symbol,String}()   # type symbol => source_file

"""
    register_model!(type_symbol::Symbol, source_file::String)
"""
function register_model!(type_symbol::Symbol, source_file::String)
    MODEL_REGISTRY[type_symbol] = source_file
    return nothing
end

"""
    show_models()
"""
function show_models()
    if isempty(MODEL_REGISTRY)
        println("No models registered.")
        return nothing
    end
    println("Available models:")
    for type_symbol in keys(MODEL_REGISTRY)
        println("  :", type_symbol)
    end
    return nothing
end

"""
    load_model(type_symbol::Symbol)
"""
function load_model(type_symbol::Symbol)
    if !haskey(MODEL_REGISTRY, type_symbol)
        available = isempty(MODEL_REGISTRY) ? "(none registered yet)" : join(sort(collect(keys(MODEL_REGISTRY))), ", ")
        error("Model '$(type_symbol)' is not recognized. Available models: $available")
    end
    include(MODEL_REGISTRY[type_symbol])
    println("Model '$(type_symbol)' loaded successfully.")
    return nothing
end