# ============================================================
# run_example.jl
#
# Demonstrates the full workflow:
#   1. Build a System (framework utility)
#   2. Load a model by name via the registry (framework orchestrates,
#      model does the parsing/Hamiltonian construction)
#   3. Solve for the band structure (framework)
#   4. Plot (framework)
#
# Uses a simple cubic lattice with a two-site (A, B) basis as a
# material-agnostic example of the orthogonal_sk model.
# ============================================================

include("framework.jl")
include("models_registry.jl")

# --- 1. Build the atomistic system ---
a = 2.0  # lattice constant (Angstrom)
lattice = Lattice(Matrix{Float64}(I(3)) .* a)
system = System(lattice)

add_atom!(system, :A, [0.0, 0.0, 0.0], ["s"])
add_atom!(system, :B, [a/2, a/2, a/2], ["s"])

# --- 2. Load the model (framework dispatches to the model layer) ---
model = load_etb(system, "orthogonal_sk", "orthogonal_sk_params.toml")
show_params(model)

# --- 3. Define a k-path and compute the band structure ---
Gamma = [0.0, 0.0, 0.0]
X = [pi/a, 0.0, 0.0]
M = [pi/a, pi/a, 0.0]
R = [pi/a, pi/a, pi/a]

k_path = [Gamma, X, M, Gamma, R]
n_points = 300

k_distances, bands = band_structure(model, k_path, n_points)

# tick positions at each high-symmetry point along the path
seg_points = n_points ÷ (length(k_path) - 1)
tick_idxs = [1 + (i-1)*seg_points for i in 1:length(k_path)-1]
push!(tick_idxs, length(k_distances))
xticks = k_distances[tick_idxs]
xticklabels = ["Γ", "X", "M", "Γ", "R"]

# --- 4. Plot ---
p = plot_bands(k_distances, bands; xticks=xticks, xticklabels=xticklabels,
               title="Band Structure — orthogonal_sk example model")
display(p)
savefig(p, "bands_example.png")

println("Done. Plot saved to bands_example.png")
