# ============================================================
# neighbors.jl
#
# Framework layer — geometry utilities.
#
# Periodic neighbor search: given a System and a cutoff radius, finds
# all atom pairs (i, j, R) within range, where R is the lattice
# translation vector connecting the periodic image of j to i.
#
# Models call `find_neighbors` to discover which pairs of orbitals can
# have nonzero hopping, without needing to know anything about how
# periodicity or lattice translations are handled.
# ============================================================

"""
    Neighbor

A neighboring atom-pair relation found within a cutoff radius.

- `i`, `j`   : atom indices in the System (source, target)
- `R`        : integer lattice translation applied to atom j's cell
- `d`        : Cartesian displacement vector from atom i to the periodic image of atom j
- `distance` : norm(d)
"""
struct Neighbor
    i::Int
    j::Int
    R::Vector{Int}
    d::Vector{Float64}
    distance::Float64
end

"""
    is_canonical(i, j, R) -> Bool

Defines a canonical ordering for (i, j, R) vs. its mirror (j, i, -R), so
that `find_neighbors(..., canonical=true)` returns each physical bond
exactly once. Convention: i < j, or i == j with R lexicographically positive.
"""
function is_canonical(i::Int, j::Int, R::Vector{Int})
    if i < j
        return true
    elseif i > j
        return false
    else
        for val in R
            if val > 0
                return true
            elseif val < 0
                return false
            end
        end
        return false
    end
end

"""
    find_neighbors(system::System, cutoff::Float64; canonical::Bool=true) -> Vector{Neighbor}

Finds all atom pairs within `cutoff` Angstrom, scanning the necessary
range of periodic images automatically from the lattice vectors.

If `canonical` is true (default), each physical bond (i,j,R) / (j,i,-R)
pair is returned only once — appropriate for building a Hermitian
Hamiltonian where the conjugate term is added implicitly. Set to false
to get both directions explicitly.
"""
function find_neighbors(system::System, cutoff::Float64; canonical::Bool=true)
    neighbors = Neighbor[]
    lattice = system.lattice
    atoms = system.atoms
    n = length(atoms)

    pseudo_inv, row_norms, valid_indices = get_lattice_params(lattice)

    for i in 1:n
        pos_i = atoms[i].position
        for j in 1:n
            pos_j = atoms[j].position
            r_ij = pos_j - pos_i

            if system.dim == 0
                center = Float64[]
            else
                center = -pseudo_inv * r_ij
            end

            ranges = [0:0, 0:0, 0:0]
            for (k, v_idx) in enumerate(valid_indices)
                lo = floor(Int, center[k] - cutoff * row_norms[k])
                hi = ceil(Int, center[k] + cutoff * row_norms[k])
                ranges[v_idx] = lo:hi
            end

            for n1 in ranges[1], n2 in ranges[2], n3 in ranges[3]
                R = [n1, n2, n3]

                if canonical && !is_canonical(i, j, R)
                    continue
                end

                d = get_dR(lattice, R) + r_ij
                dist = norm(d)

                if dist <= cutoff && (dist > 1e-6 || i != j)
                    push!(neighbors, Neighbor(i, j, R, d, dist))
                end
            end
        end
    end
    return neighbors
end
