# ============================================================
# plotting.jl
#
# Framework layer — visualization utilities.
# ============================================================

using Plots

"""
    plot_bands(k_distances, bands; xticks=nothing, xticklabels=nothing, title="Band Structure")

Plots a band structure produced by `band_structure`.
"""
function plot_bands(k_distances::Vector{Float64}, bands::Matrix{Float64};
                     xticks=nothing, xticklabels=nothing, title="Band Structure")
    p = plot(title=title, ylabel="Energy (eV)", xlabel="k-path", legend=false, framestyle=:box)

    for b in 1:size(bands, 2)
        plot!(p, k_distances, bands[:, b], color=:blue, linewidth=2)
    end

    if xticks !== nothing && xticklabels !== nothing
        plot!(p, xticks=(xticks, xticklabels))
        for xt in xticks
            vline!(p, [xt], color=:black, linestyle=:dash, linewidth=1)
        end
    end

    return p
end
