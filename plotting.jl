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

"""
    plot_unitcell(system::System; repetitions=(1, 1, 1))

Visualizes the real-space atomistic system in a 3D plot.
Displays atoms with their species names, lattice vectors, and the unit cell box.
`repetitions=(nx, ny, nz)` sets the exact number of unit cells to draw along each direction.
Repeated atoms are shown in faded colors to distinguish them from the main cell.
"""
function plot_unitcell(system::System; repetitions=(1, 1, 1))
    # 1. Automatic 3D backend checking
    has_backend = false
    try
        plotly()
        has_backend = true
    catch
        try
            plotlyjs()
            has_backend = true
        catch
            @warn "Interactive 3D backends (Plotly/PlotlyJS) are not loaded or installed. The plot will be non-interactive. To enable interactive 3D plots, run install packages `Plotly/PlotlyJS` and restart."
            gr()
        end
    end

    p = plot(legend=false, title="Unit Cell", xlabel="X (Å)", ylabel="Y (Å)", zlabel="Z (Å)")

    lat = system.lattice
    a1 = isnothing(lat.a1) ? [0.0, 0.0, 0.0] : lat.a1
    a2 = isnothing(lat.a2) ? [0.0, 0.0, 0.0] : lat.a2
    a3 = isnothing(lat.a3) ? [0.0, 0.0, 0.0] : lat.a3

    # 4. Lattice vectors with manual 3D arrows and hover tooltips
    function draw_vec(v, col, label)
        v_norm = norm(v)
        if v_norm > 1e-6
            hover_str = "$(label): [$(round(v[1], digits=3)), $(round(v[2], digits=3)), $(round(v[3], digits=3))]"
            plot!(p, [0.0, v[1]], [0.0, v[2]], [0.0, v[3]], color=col, linewidth=4, hover=hover_str)
            
            # Manual 3D arrowhead
            u = v ./ v_norm
            temp = abs(u[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
            w1 = [u[2]*temp[3] - u[3]*temp[2], u[3]*temp[1] - u[1]*temp[3], u[1]*temp[2] - u[2]*temp[1]]
            w1 ./= norm(w1)
            w2 = [u[2]*w1[3] - u[3]*w1[2], u[3]*w1[1] - u[1]*w1[3], u[1]*w1[2] - u[2]*w1[1]]
            
            s = min(v_norm * 0.15, 0.5)
            for angle in [0, pi/2, pi, 3*pi/2]
                wing = v .- s .* u .+ (s * 0.3) .* (cos(angle).*w1 .+ sin(angle).*w2)
                plot!(p, [v[1], wing[1]], [v[2], wing[2]], [v[3], wing[3]], color=col, linewidth=4, hover=hover_str)
            end
        end
    end

    draw_vec(a1, :red, "a1")
    draw_vec(a2, :green, "a2")
    draw_vec(a3, :blue, "a3")

    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    edges = [(1, 2), (1, 3), (1, 5), (2, 4), (2, 6), (3, 4), (3, 7), (4, 8), (5, 6), (5, 7), (6, 8), (7, 8)]

    if system.dim > 0
        for (v1_idx, v2_idx) in edges
            c1, c2 = vertices[v1_idx], vertices[v2_idx]

            skip = false
            if isnothing(lat.a1) && (c1[1] != c2[1])
                skip = true
            end
            if isnothing(lat.a2) && (c1[2] != c2[2])
                skip = true
            end
            if isnothing(lat.a3) && (c1[3] != c2[3])
                skip = true
            end

            if !skip
                p1 = c1[1] * a1 + c1[2] * a2 + c1[3] * a3
                p2 = c2[1] * a1 + c2[2] * a2 + c2[3] * a3
                plot!(p, [p1[1], p2[1]], [p1[2], p2[2]], [p1[3], p2[3]], color=:gray, linestyle=:dash, linewidth=2, hover=false)
            end
        end
    end

    species_list = unique([atom.species for atom in system.atoms])
    palette = theme_palette(:auto)
    species_color = Dict{Symbol,Any}()
    for (i, sp) in enumerate(species_list)
        species_color[sp] = palette[mod1(i, length(palette))]
    end

    # Collect coordinates for bounding box equal aspect ratio fix
    all_x, all_y, all_z = Float64[], Float64[], Float64[]

    # 2. Centered repetitions (exact cell counts)
    nx = isnothing(lat.a1) ? 1 : max(1, repetitions[1])
    ny = isnothing(lat.a2) ? 1 : max(1, repetitions[2])
    nz = isnothing(lat.a3) ? 1 : max(1, repetitions[3])

    for n1 in -(nx ÷ 2) : (nx - 1) ÷ 2
        for n2 in -(ny ÷ 2) : (ny - 1) ÷ 2
            for n3 in -(nz ÷ 2) : (nz - 1) ÷ 2
                is_main_cell = (n1 == 0 && n2 == 0 && n3 == 0)
                alpha_val = is_main_cell ? 1.0 : 0.15

                offset = n1 * a1 + n2 * a2 + n3 * a3

                for atom in system.atoms
                    pos = atom.position + offset
                    push!(all_x, pos[1])
                    push!(all_y, pos[2])
                    push!(all_z, pos[3])

                    col = species_color[atom.species]

                    # 3. Smaller atoms, hover info
                    scatter3d!(p, [pos[1]], [pos[2]], [pos[3]],
                        markercolor=col, markeralpha=alpha_val, markersize=4,
                        hover="Species: $(atom.species)<br>Pos: $(round.(pos, digits=3))")
                end
            end
        end
    end

    # 3. Invisible bounding cube to enforce equal aspect ratio
    if !isempty(all_x)
        min_x, max_x = minimum(all_x), maximum(all_x)
        min_y, max_y = minimum(all_y), maximum(all_y)
        min_z, max_z = minimum(all_z), maximum(all_z)

        max_range = maximum([max_x - min_x, max_y - min_y, max_z - min_z])
        max_range = max(max_range, 1.0)

        mid_x, mid_y, mid_z = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2
        half = max_range / 2

        for cx in [mid_x - half, mid_x + half]
            for cy in [mid_y - half, mid_y + half]
                for cz in [mid_z - half, mid_z + half]
                    scatter3d!(p, [cx], [cy], [cz], markeralpha=0.0, markersize=0, hover=false)
                end
            end
        end
    end

    return p
end
