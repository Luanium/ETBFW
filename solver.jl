# ============================================================
# solver.jl
#
# Framework layer — solver / orchestrator.
#
# Drives the calculation flow: asks a model (any AbstractETBModel) to
# build H(k), diagonalizes it, and assembles results (band structure,
# and in future DOS, eigenvectors, etc.) The framework never builds
# the Hamiltonian itself — that is always delegated to the model.
# ============================================================

using LinearAlgebra

"""
    solve_kpoint(model::AbstractETBModel, k::Vector{Float64}) -> Vector{Float64}

Requests H(k) from the model, Hermitizes it defensively, and returns
sorted eigenvalues.
"""
function solve_kpoint(model::AbstractETBModel, k::Vector{Float64})
    H = build_hamiltonian(model, k)
    H_herm = Hermitian((H + H') / 2)
    return eigvals(H_herm)
end

"""
    band_structure(model::AbstractETBModel, k_path::Vector{Vector{Float64}}, n_points::Int=100)
        -> (k_distances::Vector{Float64}, bands::Matrix{Float64})

Calculates eigenvalues along a piecewise-linear path through
high-symmetry k-points, roughly `n_points` total. Returns the
cumulative k-distance for each sampled point (for plotting) and a
(n_kpoints x n_bands) matrix of eigenvalues.
"""
function band_structure(model::AbstractETBModel, k_path::Vector{Vector{Float64}}, n_points::Int=100)
    num_segments = length(k_path) - 1
    @assert num_segments >= 1 "k_path must contain at least two points."
    points_per_segment = max(1, n_points ÷ num_segments)

    k_points = Vector{Vector{Float64}}()
    k_distances = Float64[]
    current_dist = 0.0

    for s in 1:num_segments
        k_start = k_path[s]
        k_end = k_path[s+1]
        dist = norm(k_end - k_start)

        for i in 0:(points_per_segment-1)
            f = i / points_per_segment
            push!(k_points, k_start + f * (k_end - k_start))
            push!(k_distances, current_dist + f * dist)
        end
        current_dist += dist
    end
    push!(k_points, k_path[end])
    push!(k_distances, current_dist)

    N = num_bands(model)
    bands = zeros(Float64, length(k_points), N)
    for (i, k) in enumerate(k_points)
        bands[i, :] = solve_kpoint(model, k)
    end

    return k_distances, bands
end
