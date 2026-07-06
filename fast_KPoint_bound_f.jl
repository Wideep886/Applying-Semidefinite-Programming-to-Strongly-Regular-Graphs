# fast_KPoint_bound_f.jl — k-point SDP bound (feasible-state quotient form)
#
# Based on the Julia scripts of de Laat et al. (KPointBound.txt; arXiv:1812.06045).
# Modifications: PSD variables on feasible state spaces |U_R| (Liao, Thm. 4.5;
# Remark 4.8) instead of the full 3^m monomial grid; three-stage template / fill /
# solve API; faster independent-set orbit enumeration for k >= 6.
#
# Requirements: Julia 1.x; packages Nemo, Combinatorics, IterTools, LinearAlgebra, Printf
# External solver: sdpa_gmp (patched build recommended)
#   http://www.daviddelaat.nl/sdpa-gmp-7.1.3.tar.gz
# Put sdpa_gmp on PATH, or set ENV["SDPAGMP_PATH"], or place it in ./bin/ or ./
#
# Install packages (in Julia):
#   using Pkg; Pkg.add(["Nemo", "Combinatorics", "IterTools"])
#
# Quick start:
#   julia --startup-file=no fast_KPoint_bound_f.jl
#
#   julia --startup-file=no -e '
#     include("fast_KPoint_bound_f.jl"); using .FastKPointBoundF
#     solve_k_point_bound(100, 5, 2, [-1//5, 1//5]; verbose=true)
#   '
#
# API:
#   build_kpoint_template(k, d, D)   # stage A: n-independent (cacheable)
#   fill_kpoint_sdp(n, tpl)          # stage B: fill coefficients
#   solve_kpoint_sdp(prob)           # stage C: compress, SDPA-GMP, Arb verify
#   solve_k_point_bound(n, k, d, D)  # convenience wrapper
#   inspect_kpoint_sdp!(...)         # build + structural check without solving
#
# Arguments: n = ambient sphere dimension S^{n-1}; k = point level; d = Gegenbauer
# truncation (l = 0..d); D = [a, b] inner products of the two-distance set.
#
# SRG non-existence checks (s-eigenspace projection, k=5, d=2; Liao thesis Sec. 5.3):
#   (550,387,260,301): solve_k_point_bound(33, 5, 2, [-1//9, 7//27])
#       → floor(alpha) = 546  (< v=550); certified alpha ≈ 546.12
#   (703,520,372,420): solve_k_point_bound(37, 5, 2, [-5//52, 7//26])
#       → floor(alpha) = 662  (< v=703); certified alpha ≈ 662.35
# In both cases v exceeds the SDP upper bound, so no SRG with those parameters exists.
#
# Default numerics (override via keyword arguments): ARB_PREC=2500, SDPA_PRECISION=1500,
# SDPA_MAXITER=500_000; SDPA tolerances epsilonStar/Dash 1e-16, lambdaStar 1e8,
# betaStar 0.2, betaBar 0.3, gammaStar 0.5.
#
# Please cite de Laat et al. (2022) when using this workflow.
#

module FastKPointBoundF

using LinearAlgebra, Nemo, Combinatorics, IterTools, Printf

export build_kpoint_template, fill_kpoint_sdp, compress_kpoint_sdp, solve_kpoint_sdp, solve_k_point_bound
export SDPAGMPOptions, default_sdpa_gmp_path, run_example!, KPointTemplate
export prepare_kpoint_sdp, inspect_kpoint_sdp!, KPointBuildReport

const arb = Nemo.ArbFieldElem

# Default example (used when running this file directly)
const EXAMPLE_N = 37
const EXAMPLE_K = 5
const EXAMPLE_DEG = 2
const EXAMPLE_INNER_A = -5 // 52
const EXAMPLE_INNER_B = 7 // 26

# Numerical defaults (see file header)
const ARB_PREC = 2500
const SDPA_PRECISION = 1500
const SDPA_MAXITER = 500_000
const SDPA_EPSILONSTAR = "1e-16"
const SDPA_EPSILONDASH = "1e-16"
const SDPA_LAMBDASTAR = "1e8"
const SDPA_BETASTAR = "0.2"
const SDPA_BETABAR = "0.3"
const SDPA_GAMMASTAR = "0.5"

example_D() = [EXAMPLE_INNER_A, EXAMPLE_INNER_B]

# --- Gegenbauer polynomials and two-distance geometry ---

function gegenbauer(k, n, x)
    a = one(x)
    k == 0 && return a
    b = x
    k == 1 && return b
    for l in 2:k
        c = (2l + n - 4) // (l + n - 3) * x * b - (l - 1) // (l + n - 3) * a
        a = b
        b = c
    end
    b
end

gegenbauwer_leading_coeff(k, n) = k == 0 ? 1 : prod((2l + n - 4) // (l + n - 3) for l in 1:k)

function P(n, m, l, u, v, t)
    if isempty(u)
        return l == 0 ? one(t) : gegenbauer(l, n, t)
    end
    s = (1 - dot(u, u)) * (1 - dot(v, v))
    if abs(s) < 1e-12
        gegenbauwer_leading_coeff(l, n - m) * (t - dot(u, v))^l
    else
        s = sqrt(s)
        s^l * gegenbauer(l, n - m, (t - dot(u, v)) / s)
    end
end

function isapprox(x::arb_mat, y::arb_mat)
    all(abs(x[i, j] - y[i, j]) < 1e-12 for i in 1:size(x, 1) for j in 1:size(x, 2))
end

function arbcholesky(x::arb_mat)
    y = zero_matrix(base_ring(x), size(x, 1), size(x, 2))
    status = @ccall Nemo.libflint.arb_mat_cho(
        y::Ref{arb_mat}, x::Ref{arb_mat}, precision(base_ring(x))::Int
    )::Cint
    status, y
end

function arbcholesky(x::Matrix{arb})
    R = parent(x[1, 1])
    M = zero_matrix(R, size(x, 1), size(x, 2))
    for i in 1:size(x, 1), j in 1:size(x, 2)
        M[i, j] = x[i, j]
    end
    s, L = arbcholesky(M)
    s, Matrix(L)
end

function orbitequal(M1::Matrix{T}, M2::Matrix{T}) where T
    size(M1) != size(M2) && return false
    any(
        all(abs(M1[p, p][i, j] - M2[i, j]) < 1e-8 for i in 1:size(M1, 1), j in 1:size(M1, 2))
        for p in permutations(1:size(M1, 1))
    )
end

function orbitequal(Q1::Vector{Vector{T}}, Q2::Vector{Vector{T}}) where T
    length(Q1) != length(Q2) && return false
    length(Q1) == 0 && return true
    orbitequal(
        [dot(Q1[i], Q1[j]) for i in 1:length(Q1), j in 1:length(Q1)],
        [dot(Q2[i], Q2[j]) for i in 1:length(Q2), j in 1:length(Q2)],
    )
end

function pcholesky(A)
    A = deepcopy(A)
    n = size(A, 1)
    p = collect(1:n)
    for k in 1:n
        d = [A[i, i] for i in 1:n]
        val, s = findmax(d[k:n])
        s = s + k - 1
        val < 0 && return false, A, p
        if s != k
            A[:, [k, s]] = A[:, [s, k]]
            A[[k, s], :] = A[[s, k], :]
            p[[k, s]] = p[[s, k]]
        end
        A[k, k] = sqrt(A[k, k])
        k == n && break
        A[k, k + 1:n] = A[k, k + 1:n] ./ A[k, k]
        j = (k + 1):n
        A[j, j] = A[j, j] .- A[k, j] .* transpose(A[k, j])
    end
    for i in 1:n, j in 1:i - 1
        A[i, j] = zero(A[i, j])
    end
    true, A, p
end

"""Upper-triangular lex-min signature for a Gram matrix orbit."""
function gram_orbit_signature(M::Matrix{Float64})
    n = size(M, 1)
    best = Tuple(M[i, j] for i in 1:n for j in i:n)
    n == 1 && return best
    for p in permutations(1:n)
        p == collect(1:n) && continue
        sig = Tuple(M[p, p][i, j] for i in 1:n for j in i:n)
        sig < best && (best = sig)
    end
    best
end

"""
Enumerate two-distance independent-set orbits R (de Laat).
For |R|=card there are 2^C(card,2) inner-product patterns; card=6 gives 2^15 configs.
Uses Float64 orbit signatures instead of Arb Set deduplication for speed at k=6.
"""
function independentsets(n::Int, card::Int, innerproducts::Vector)
    @assert n >= card
    R = parent(innerproducts[1])
    ip = [Float64(x) for x in innerproducts]
    seen = Set{Tuple{Vararg{Float64}}}()
    out = Vector{Vector{arb}}[]
    card == 0 && (push!(out, Vector{arb}[]); return out)

    npairs = binomial(card, 2)
    total = length(ip)^npairs
    report = card >= 5

    for k in 0:total - 1
        report && (k % 4096 == 0 || k == total - 1) && begin
            print(stderr, "\r  independentsets |R|=$card: ", k + 1, "/", total)
            flush(stderr)
        end

        M = Matrix{Float64}(undef, card, card)
        for i in 1:card
            M[i, i] = 1.0
        end
        i = j = 1
        for z in digits(k, base = length(ip), pad = npairs)
            j += 1
            M[i, j] = M[j, i] = ip[z + 1]
            if j == card
                i += 1
                j = i
            end
        end

        try
            cholesky(Symmetric(M))
        catch
            continue
        end
        sig = gram_orbit_signature(M)
        sig in seen && continue
        push!(seen, sig)

        # Rebuild Gram matrices with exact Arb inner products
        Marb = Matrix{arb}(undef, card, card)
        for i in 1:card
            Marb[i, i] = one(R)
        end
        i = j = 1
        for z in digits(k, base = length(innerproducts), pad = npairs)
            j += 1
            Marb[i, j] = Marb[j, i] = innerproducts[z + 1]
            if j == card
                i += 1
                j = i
            end
        end
        ok, F, _ = pcholesky(Marb)
        ok && push!(out, [[F[r, c] for r in 1:size(F, 2)] for c in 1:card])
    end
    report && println(stderr)
    out
end

function permequal(u::Vector{Vector{T}}, v::Vector{Vector{T}}) where T
    length(u) != length(v) && return false
    all(
        all(abs(x - y) < 1e-12 for (x, y) in zip(a, b))
        for (a, b) in zip(sort(u), sort(v))
    )
end

function is_psd_extension(G::Matrix{arb}, u::Vector{arb})
    m = length(u)
    H = Matrix{arb}(undef, m + 1, m + 1)
    for i in 1:m, j in 1:m
        H[i, j] = G[i, j]
    end
    for i in 1:m
        H[i, m + 1] = H[m + 1, i] = u[i]
    end
    H[m + 1, m + 1] = one(parent(u[1]))
    arbcholesky(H)[1] == 1
end

function is_psd_gram(G::Matrix{arb}, u::Vector{arb}, v::Vector{arb}, t)
    m = length(u)
    H = Matrix{arb}(undef, m + 2, m + 2)
    for i in 1:m, j in 1:m
        H[i, j] = G[i, j]
    end
    for i in 1:m
        H[i, m + 1] = H[m + 1, i] = u[i]
        H[i, m + 2] = H[m + 2, i] = v[i]
    end
    H[m + 1, m + 1] = H[m + 2, m + 2] = one(parent(u[1]))
    H[m + 1, m + 2] = H[m + 2, m + 1] = t
    arbcholesky(H)[1] == 1
end

"""Enumerate U_R on {1,a,b}^m (feasible states + reference states G*e_i)."""
function feasible_states(Q::Vector{Vector{arb}}, D::Vector{arb})
    m = length(Q)
    m == 0 && return [arb[]]
    R = parent(D[1])
    G = Matrix{arb}([dot(Q[i], Q[j]) for i in 1:m, j in 1:m])
    grid = (one(R), D[1], D[2])
    refs = [[G[j, i] for j in 1:m] for i in 1:m]
    cand = Vector{Vector{arb}}()
    for u in refs
        is_psd_extension(G, u) && push!(cand, u)
    end
    for tup in Iterators.product(ntuple(_ -> grid, m)...)
        u = collect(tup)
        is_psd_extension(G, u) || continue
        any(all(abs(u[j] - c[j]) <= 1e-12 for j in 1:m) for c in cand) && continue
        push!(cand, u)
    end
    Delta = (one(R), D[1], D[2])
    U = Vector{Vector{arb}}()
    for u in cand
        any(any(is_psd_gram(G, u, v, t) for t in Delta) for v in cand) && push!(U, u)
    end
    for u in refs
        any(all(abs(u[j] - w[j]) <= 1e-12 for j in 1:m) for w in U) || push!(U, u)
    end
    U
end

"""D_ext = D union {1}; flat index for the 3^m state grid."""
function build_D_ext(D::Vector{arb})
    R = parent(D[1])
    ext = [copy(D); one(R)]
    ext
end

function get_grid_index(mx::Vector{arb}, D_ext::Vector{arb}, tol::arb)
    idx = 1
    len_D = length(D_ext)
    for i in 1:length(mx)
        match_k = 0
        for k in 1:len_D
            if abs(mx[i] - D_ext[k]) < tol
                match_k = k
                break
            end
        end
        match_k == 0 && error("get_grid_index: value $(mx[i]) not in D_ext")
        idx += (match_k - 1) * len_D^(i - 1)
    end
    idx
end

"""Map flat 3^m grid index to state index in U_R (1..|U_R|)."""
function build_grid_to_U(U::Vector{Vector{arb}}, D_ext::Vector{arb}, tol::arb)
    m = isempty(U[1]) ? 0 : length(U[1])
    len_D = length(D_ext)
    m == 0 && return Dict(1 => 1)
    out = Dict{Int, Int}()
    for (i, u) in enumerate(U)
        idx = 1
        for j in 1:m
            match_k = 0
            for k in 1:len_D
                if abs(u[j] - D_ext[k]) < tol
                    match_k = k
                    break
                end
            end
            match_k == 0 && error("build_grid_to_U: state coord not in D_ext")
            idx += (match_k - 1) * len_D^(j - 1)
        end
        out[idx] = i
    end
    out
end

function arbtril(L::arb_mat, B::arb_mat)
    X = zero_matrix(base_ring(L), size(B, 1), size(B, 2))
    @ccall Nemo.libflint.arb_mat_solve(
        X::Ref{arb_mat}, L::Ref{arb_mat}, B::Ref{arb_mat}, precision(base_ring(L))::Int
    )::Cint
    X
end

# --- Fcal_U: Theorem 4.5 — accumulate P^{n,m}_l on U_R ---

"""Geometry for subset Q: Cholesky factor L, Gram-preserving permutations, normalization."""
struct FcalGeom
    m::Int
    AR::Matrix{arb}
    L::Matrix{arb}
    perms::Vector{Vector{Int}}
    nf::arb
end

function build_fcal_geom(Qp::Vector{Vector{arb}}, R::arb)
    m = length(Qp)
    m == 0 && return FcalGeom(0, Matrix{arb}(undef, 0, 0), Matrix{arb}(undef, 0, 0), Vector{Int}[], one(R))
    R = parent(Qp[1][1])
    AR = Matrix{arb}(reduce(hcat, Qp))
    M = matrix(R, transpose(AR) * AR)
    _, Lraw = arbcholesky(M)
    L = Lraw isa Matrix{arb} ? Lraw : Matrix(Lraw)
    valid = Vector{Int}[]
    nf = zero(R)
    for p in permutations(1:m)
        if isapprox(M[p, p], M)
            push!(valid, collect(p))
            nf += one(R)
        end
    end
    FcalGeom(m, AR, L, valid, nf)
end

"""True if u is a reference state G*e_i (some coordinate equals 1; Remark 4.8)."""
function is_reference_state(u::Vector{arb}, tol::arb)
    isempty(u) && return false
    R = parent(u[1])
    any(abs(u[j] - one(R)) < tol for j in 1:length(u))
end

# --- Remark 4.8: PSD block dimensions for M_0 vs M_{l>=1} ---

"""
Remark 4.8 block dimensions:
  dim0    = |U_R| (l=0, includes reference states G*e_i)
  dim_pos = count of generic states (l>=1, no coordinate equal to 1)
Bounds: dim0 <= 2^m+m, dim_pos <= 2^m.
"""
struct BlockDims
    dim0::Int
    dim_pos::Int
end

"""Build grid maps and Remark 4.8 block dimensions."""
function build_block_dims(U::Vector{Vector{arb}}, D_ext::Vector{arb}, tol::arb)
    dim0 = length(U)
    gen_u_idx = [i for i in 1:dim0 if !is_reference_state(U[i], tol)]
    dim_pos = length(gen_u_idx)
    grid_full = build_grid_to_U(U, D_ext, tol)
    grid_gen = Dict{Int, Int}()
    for (gix, uidx) in grid_full
        pos = findfirst(==(uidx), gen_u_idx)
        pos !== nothing && (grid_gen[gix] = pos)
    end
    BlockDims(dim0, dim_pos), grid_full, grid_gen
end

block_psd_dim(bd::BlockDims, l::Int) = l == 0 ? bd.dim0 : bd.dim_pos

"""Accumulate P^{n,m}_l coefficients on U_R (l=0) or U_gen (l>=1); skip reference states when l>=1."""
function fcal_U_accumulate!(
    M::Matrix, n::Int, l::Int, geom::FcalGeom, x::Vector{arb}, y::Vector{arb},
    grid_to_U::Dict{Int, Int}, grid_to_Ugen::Dict{Int, Int}, D_ext::Vector{arb}, tol::arb,
)
    R = parent(x[1])
    gmap = l == 0 ? grid_to_U : grid_to_Ugen
    if geom.m == 0
        M[1, 1] += gegenbauer(l, n, sum(a * b for (a, b) in zip(x, y)))
        return
    end
    Lmat = matrix(R, geom.L)
    invnf = inv(geom.nf)
    t = dot(x, y)
    for p in geom.perms
        mx = [dot(geom.AR[:, p[k]], x) for k in 1:geom.m]
        my = [dot(geom.AR[:, p[k]], y) for k in 1:geom.m]
        gix = get_grid_index(mx, D_ext, tol)
        giy = get_grid_index(my, D_ext, tol)
        haskey(gmap, gix) && haskey(gmap, giy) || continue
        ix, iy = gmap[gix], gmap[giy]
        Bx = zero_matrix(R, length(mx), 1)
        By = zero_matrix(R, length(my), 1)
        for i in 1:length(mx)
            Bx[i, 1] = mx[i]
        end
        for i in 1:length(my)
            By[i, 1] = my[i]
        end
        u = arbtril(Lmat, Bx)
        v = arbtril(Lmat, By)
        uvec = [u[k, 1] for k in 1:size(u, 1)]
        vvec = [v[k, 1] for k in 1:size(v, 1)]
        pval = P(n, geom.m, l, uvec, vvec, t)
        iszero(pval) && continue
        M[ix, iy] += pval * invnf
    end
end

# --- Stage A: n-independent template ---

"""One subset Q in a constraint: orbit index, geometry, prefiltered (x,y) pairs."""
struct SubsetRecipe
    s::Int
    orbitindex::Int
    Qp::Vector{Vector{arb}}
    geom::FcalGeom
    pairs::Vector{Tuple{Vector{arb}, Vector{arb}}}
end

struct ConstraintRecipe
    m::Int                              # |S| for this constraint (eq. 34-35)
    subsets::Vector{SubsetRecipe}
end

"""Cached template: fixed k, d, D; reusable for different n."""
struct KPointTemplate
    k::Int
    d::Int
    D::Vector{arb}
    D_ext::Vector{arb}
    tol::arb
    prec::Int
    Rcal::Vector{Vector{Vector{Vector{arb}}}}
    Ucache::Dict{Tuple{Int, Int}, Tuple{Vector{Vector{arb}}, Int}}
    block_dims::Dict{Tuple{Int, Int}, BlockDims}
    grid_to_U::Dict{Tuple{Int, Int}, Dict{Int, Int}}
    grid_to_Ugen::Dict{Tuple{Int, Int}, Dict{Int, Int}}
    recipes::Vector{ConstraintRecipe}
end

"""Orbit of Q in Rcal (de Laat equivalence class)."""
function orbit_perm(Q::Vector{Vector{arb}}, Rcal_s::Vector{Vector{Vector{arb}}})
    s = length(Q)
    s == 0 && return 1, Int[]
    GramQ = matrix(parent(Q[1][1]), [dot(Q[i], Q[j]) for i in 1:s, j in 1:s])
    oi = findfirst(x -> orbitequal(x, Q), Rcal_s)
    gramR = matrix(
        parent(Q[1][1]),
        [dot(Rcal_s[oi][i], Rcal_s[oi][j]) for i in 1:s, j in 1:s],
    )
    for p in permutations(1:s)
        if isapprox(GramQ[p, p], gramR)
            return oi, collect(p)
        end
    end
    error("orbit permutation not found")
end

"""Keep (x,y) only if union(Q,{x,y}) is in the same O(n) orbit as R."""
function prefilter_pairs(R::Vector{Vector{arb}}, Q::Vector{Vector{arb}})
    out = Tuple{Vector{arb}, Vector{arb}}[]
    for x in R, y in R
        permequal(union(Q, [x, y]), R) && push!(out, (x, y))
    end
    out
end

"""Stage A: enumerate U_R, Remark 4.8 dimensions, constraint recipes (independent of n)."""
function build_kpoint_template(k::Int, d::Int, D::Vector; prec::Int = ARB_PREC, verbose::Bool = true)
    Field = ArbField(prec)
    D = [Field(a) for a in D]
    D_ext = build_D_ext(D)
    tol = Field("1e-18")
    n_dummy = max(k, 1)

    verbose && println("Template: Thm 4.5 (U_R) + Remark 4.8, l=0:$d")

    verbose && println("Template: independent sets + block dimensions...")
    Rcal = Vector{Vector{Vector{Vector{arb}}}}(undef, k + 1)
    for m in 0:k
        if verbose
            print("  Rcal m=$m (|R|=$m)... ")
            flush(stdout)
        end
        t0 = time()
        Rcal[1 + m] = independentsets(n_dummy, m, D)
        verbose && println("$(length(Rcal[1 + m])) orbits, $(round(time() - t0; digits = 1)) s")
    end
    Ucache = Dict{Tuple{Int, Int}, Tuple{Vector{Vector{arb}}, Int}}()
    block_dims = Dict{Tuple{Int, Int}, BlockDims}()
    grid_to_U = Dict{Tuple{Int, Int}, Dict{Int, Int}}()
    grid_to_Ugen = Dict{Tuple{Int, Int}, Dict{Int, Int}}()

    for m in 0:(k - 2)
        theory0 = m == 0 ? 1 : 2^m + m
        theory_gen = m == 0 ? 1 : 2^m
        for (oi, Q) in enumerate(Rcal[1 + m])
            U = feasible_states(Q, D)
            bd, gfull, ggen = build_block_dims(U, D_ext, tol)
            Ucache[(m, oi)] = (U, bd.dim0)
            grid_to_U[(m, oi)] = gfull
            grid_to_Ugen[(m, oi)] = ggen
            block_dims[(m, oi)] = bd
            verbose && println(
                "  m=$m orbit $oi: |U_R|=$(bd.dim0) (M_0, ≤$theory0), |U_gen|=$(bd.dim_pos) (M_{ℓ≥1}, ≤$theory_gen)",
            )
        end
    end

    verbose && println("Template: constraint recipes + pair prefilter...")
    recipes = ConstraintRecipe[]
    for m in 1:k
        total = length(Rcal[1 + m])
        for (cont, Rindex) in enumerate(eachindex(Rcal[1 + m]))
            R = Rcal[1 + m][Rindex]
            verbose && println("  $(m)-point recipe $cont/$total...")
            subset_recipes = SubsetRecipe[]
            for s in 0:(k - 2), Q in subsets(R, s)
                oi, p = orbit_perm(Q, Rcal[1 + s])
                Qp = isempty(p) ? Q : Q[p]
                geom = build_fcal_geom(Qp, D[1])
                push!(subset_recipes, SubsetRecipe(s, oi, Qp, geom, prefilter_pairs(R, Q)))
            end
            push!(recipes, ConstraintRecipe(m, subset_recipes))
        end
    end

    KPointTemplate(k, d, D, D_ext, tol, prec, Rcal, Ucache, block_dims, grid_to_U, grid_to_Ugen, recipes)
end

# --- Stage B: fill SDP coefficients for fixed n ---

"""PSD variable block registry; sizes[b] = dimension of block b."""
mutable struct PsdVarRegistry
    sizes::Vector{Int}
end
PsdVarRegistry() = PsdVarRegistry(Int[])

new_psd_block!(reg::PsdVarRegistry, n::Int) = (push!(reg.sizes, n); length(reg.sizes))

struct LinearConstraint
    terms::Dict{Tuple{Int, Int, Int}, arb}  # (block, i, j) -> coeff (upper tri; off-diag factor 2)
    slack_block::Int                         # 1x1 slack PSD block for this inequality
    rhs::arb
end

function add_term!(terms, b, i, j, c)
    iszero(c) && return terms
    i, j = min(i, j), max(i, j)
    terms[(b, i, j)] = get(terms, (b, i, j), zero(c)) + c
end

"""Symmetric inner product <M,X>; write linear terms into the SDP."""
function add_symtraceprod!(terms, block, M)
    n = size(M, 1)
    for i in 1:n, j in 1:i
        add_term!(terms, block, i, j, (i == j ? 1 : 2) * M[i, j])
    end
end

struct KPointSDP
    field
    registry::PsdVarRegistry
    alpha_block::Int                    # 1x1 PSD block for objective min alpha
    constraints::Vector{LinearConstraint}
end

"""
Stage B: substitute n; for each k-point constraint accumulate <M_l, A_{n,l}(x,y)>.
F[m][oi][l] is the PSD block index for orbit (m,oi) at Gegenbauer layer l.
"""
function fill_kpoint_sdp(n::Int, tpl::KPointTemplate; prec::Int = tpl.prec, verbose::Bool = false)
    k, d = tpl.k, tpl.d
    Field = ArbField(prec)
    reg = PsdVarRegistry()

    alpha_block = new_psd_block!(reg, 1)
    F = Vector{Vector{Int}}[]
    for m in 0:(k - 2)
        row = Vector{Int}[]
        for oi in 1:length(tpl.Rcal[1 + m])
            blks = Int[]
            bd = tpl.block_dims[(m, oi)]
            for l in 0:d
                push!(blks, new_psd_block!(reg, block_psd_dim(bd, l)))
            end
            push!(row, blks)
        end
        push!(F, row)
    end

    constraints = LinearConstraint[]
    for (ci, recipe) in enumerate(tpl.recipes)
        verbose && println("  fill constraint $ci/$(length(tpl.recipes))...")
        terms = Dict{Tuple{Int, Int, Int}, arb}()
        rhs = zero(Field)
        for sub in recipe.subsets
            blk = F[1 + sub.s][sub.orbitindex]
            bd = tpl.block_dims[(sub.s, sub.orbitindex)]
            gfull = tpl.grid_to_U[(sub.s, sub.orbitindex)]
            ggen = tpl.grid_to_Ugen[(sub.s, sub.orbitindex)]
            for l in 0:d
                dim = block_psd_dim(bd, l)
                M = [zero(Field) for _ in 1:dim, _ in 1:dim]
                for (x, y) in sub.pairs
                    fcal_U_accumulate!(M, n, l, sub.geom, x, y, gfull, ggen, tpl.D_ext, tpl.tol)
                end
                add_symtraceprod!(terms, blk[1 + l], M)
            end
        end
        recipe.m == 2 && (rhs += Field(2))
        recipe.m == 1 && (add_term!(terms, alpha_block, 1, 1, -one(Field)); rhs += one(Field))
        slack_block = new_psd_block!(reg, 1)
        push!(constraints, LinearConstraint(terms, slack_block, rhs))
    end

    verbose && println("SDP: $(length(constraints)) constraints, $(length(reg.sizes)) blocks, $(sum(reg.sizes .^ 2)) scalar vars")
    KPointSDP(Field, reg, alpha_block, constraints)
end

# --- PSD block compression and inspect (prepare) ---

"""Scan linear terms; build raw-to-compressed index map per PSD block."""
function build_psd_index_maps(prob::KPointSDP)
    nblocks = length(prob.registry.sizes)
    maps = [Dict{Int, Int}() for _ in 1:nblocks]
    next_idx = ones(Int, nblocks)

    function register!(b::Int, idx::Int)
        m = maps[b]
        haskey(m, idx) && return
        m[idx] = next_idx[b]
        next_idx[b] += 1
    end

    register!(prob.alpha_block, 1)
    for c in prob.constraints
        for ((b, i, j), _) in c.terms
            register!(b, i)
            register!(b, j)
        end
        register!(c.slack_block, 1)
    end
    maps
end

"""Compress nominal |U|x|U| PSD blocks to active indices used in constraints."""
function compress_kpoint_sdp(prob::KPointSDP)
    maps = build_psd_index_maps(prob)
    new_sizes = [
        isempty(maps[b]) ? prob.registry.sizes[b] : length(maps[b])
        for b in 1:length(prob.registry.sizes)
    ]
    new_constraints = LinearConstraint[]
    for c in prob.constraints
        new_terms = Dict{Tuple{Int, Int, Int}, arb}()
        for ((b, i, j), v) in c.terms
            ni, nj = maps[b][i], maps[b][j]
            ni, nj = min(ni, nj), max(ni, nj)
            key = (b, ni, nj)
            new_terms[key] = get(new_terms, key, zero(v)) + v
        end
        push!(new_constraints, LinearConstraint(new_terms, c.slack_block, c.rhs))
    end
    KPointSDP(prob.field, PsdVarRegistry(new_sizes), prob.alpha_block, new_constraints)
end

compression_scalar_vars(prob::KPointSDP) = sum(prob.registry.sizes .^ 2)

function count_sdpa_nnz(prob::KPointSDP)
    1 + sum(length(c.terms) + 1 for c in prob.constraints)
end

"""Build-stage summary (does not call sdpa_gmp)."""
struct KPointBuildReport
    k::Int
    n::Int
    d::Int
    template_secs::Float64
    fill_secs::Float64
    n_constraints::Int
    n_blocks_raw::Int
    n_blocks_compressed::Int
    scalar_vars_raw::Int
    scalar_vars_compressed::Int
    nnz_linear_terms::Int
    max_block_dim::Int
    sum_block_dim_sq::Int
end

"""
Stages A+B: build template, fill coefficients, compress PSD blocks.
Does not solve. Returns (tpl, prob, cprob, report) for inspection or later solve_kpoint_sdp(cprob).
"""
function prepare_kpoint_sdp(
    n::Int, k::Int, d::Int, D::Vector;
    tpl::Union{KPointTemplate, Nothing} = nothing,
    prec::Int = ARB_PREC,
    verbose::Bool = true,
)
    if tpl === nothing
        t0 = time()
        tpl = build_kpoint_template(k, d, D; prec = prec, verbose = verbose)
        template_secs = time() - t0
        verbose && println("Template build: $(round(template_secs; digits = 1)) s")
    else
        template_secs = 0.0
    end

    t1 = time()
    prob = fill_kpoint_sdp(n, tpl; prec = prec, verbose = verbose)
    fill_secs = time() - t1
    verbose && println("Fill (n=$n): $(round(fill_secs; digits = 1)) s")

    cprob = compress_kpoint_sdp(prob)
    sizes = cprob.registry.sizes
    report = KPointBuildReport(
        k, n, d, template_secs, fill_secs,
        length(cprob.constraints),
        length(prob.registry.sizes),
        length(sizes),
        compression_scalar_vars(prob),
        compression_scalar_vars(cprob),
        count_sdpa_nnz(cprob),
        isempty(sizes) ? 0 : maximum(sizes),
        sum(sizes .^ 2),
    )
    tpl, prob, cprob, report
end

"""Check constraints and SDPA export; optional trial .dat write (no solve)."""
function inspect_kpoint_sdp!(
    n::Int, k::Int, d::Int, D::Vector;
    tpl::Union{KPointTemplate, Nothing} = nothing,
    prec::Int = ARB_PREC,
    verbose::Bool = true,
    write_dat::Union{Nothing, AbstractString} = nothing,
    dat_digits::Int = SDPA_PRECISION,
)
    tpl, prob, cprob, rep = prepare_kpoint_sdp(n, k, d, D; tpl = tpl, prec = prec, verbose = verbose)

    issues = String[]
    for (i, c) in enumerate(cprob.constraints)
        isempty(c.terms) && push!(issues, "constraint $i: empty linear terms")
        recipe = tpl.recipes[i]
        recipe.m != 1 && continue
        has_alpha = any(k[1] == cprob.alpha_block for k in keys(c.terms))
        !has_alpha && push!(issues, "constraint $i (|S|=1): missing alpha term")
    end
    length(issues) > 0 && error("inspect failed:\n  " * join(issues, "\n  "))

    if write_dat !== nothing
        t0 = time()
        write_sdpa_sparse(write_dat, cprob, dat_digits)
        bytes = filesize(write_dat)
        verbose && println(
            "SDPA .dat trial write: $write_dat  ($(round(bytes / 1e6; digits = 2)) MB, ",
            "$(round(time() - t0; digits = 1)) s)",
        )
    end

    if verbose
        println("\n--- inspect summary (k=$(rep.k), n=$(rep.n), d=$(rep.d)) ---")
        @printf("  constraints       = %d\n", rep.n_constraints)
        @printf("  PSD blocks        = %d -> %d (compressed)\n", rep.n_blocks_raw, rep.n_blocks_compressed)
        @printf("  scalar vars sum n^2 = %d -> %d\n", rep.scalar_vars_raw, rep.scalar_vars_compressed)
        @printf("  max block dim     = %d\n", rep.max_block_dim)
        @printf("  SDPA nnz (approx) = %d\n", rep.nnz_linear_terms)
        @printf("  template %.1fs + fill %.1fs\n", rep.template_secs, rep.fill_secs)
        println("  structure check passed (sdpa_gmp not called)")
    end

    tpl, prob, cprob, rep
end

# --- SDPA I/O, sdpa_gmp solve, high-level API ---

arb_to_bigfloat(x::arb) = BigFloat(x)

function write_sdpa_sparse(path::AbstractString, prob::KPointSDP, digits::Int)
    setprecision(digits) do
        open(path, "w") do io
            m = length(prob.constraints)
            println(io, m)
            println(io, length(prob.registry.sizes))
            println(io, join(prob.registry.sizes, " "))
            for (idx, c) in enumerate(prob.constraints)
                line = string(-arb_to_bigfloat(c.rhs))
                idx < m ? print(io, line, " ") : println(io, line)
            end
            println(io, "0 ", prob.alpha_block, " 1 1 -1")
            for (ci, c) in enumerate(prob.constraints)
                for ((blk, i, j), v) in c.terms
                    coeff = arb_to_bigfloat(v)
                    iszero(coeff) && continue
                    println(io, ci, " ", blk, " ", i, " ", j, " ", i == j ? coeff : coeff / 2)
                end
                println(io, ci, " ", c.slack_block, " 1 1 1")
            end
        end
    end
end

struct SDPAGMPOptions
    executable::String
    precision::Int
    maxiteration::Int
    epsilonstar::String
    epsilondash::String
    lambdastar::String
    betastar::String
    betabar::String
    gammastar::String
    verbose::Bool
end

function default_sdpa_gmp_path()
    for p in (
        get(ENV, "SDPAGMP_PATH", ""),
        joinpath(@__DIR__, "bin", "sdpa_gmp"),
        joinpath(@__DIR__, "sdpa_gmp"),
        "sdpa_gmp",
    )
        !isempty(p) && isfile(p) && return p
    end
    "sdpa_gmp"
end

SDPAGMPOptions(;
    executable = default_sdpa_gmp_path(),
    precision = SDPA_PRECISION,
    maxiteration = SDPA_MAXITER,
    epsilonstar = SDPA_EPSILONSTAR,
    epsilondash = SDPA_EPSILONDASH,
    lambdastar = SDPA_LAMBDASTAR,
    betastar = SDPA_BETASTAR,
    betabar = SDPA_BETABAR,
    gammastar = SDPA_GAMMASTAR,
    verbose = true,
) = SDPAGMPOptions(
    executable, precision, maxiteration,
    epsilonstar, epsilondash, lambdastar,
    betastar, betabar, gammastar, verbose,
)

function write_sdpa_param(path::AbstractString, opt::SDPAGMPOptions)
    open(path, "w") do io
        for line in (
            "$(opt.maxiteration) unsigned int maxIteration;",
            "$(opt.epsilonstar) double 0.0 < epsilonStar;",
            "$(opt.lambdastar) double 0.0 < lambdaStar;",
            "2.0 double 1.0 < omegaStar;",
            "-1e5 double lowerBound;",
            "1e5 double upperBound;",
            "$(opt.betastar) double 0.0 <= betaStar <  1.0;",
            "$(opt.betabar) double 0.0 <= betaBar  <  1.0, betaStar <= betaBar;",
            "$(opt.gammastar) double 0.0 < gammaStar  <  1.0;",
            "$(opt.epsilondash) double 0.0 < epsilonDash;",
            "$(opt.precision) precision;",
        )
            println(io, line)
        end
    end
end

function _commasplit(s)
    out, depth, start = String[], 0, 1
    for (k, c) in enumerate(s)
        c == '{' && (depth += 1)
        c == '}' && (depth -= 1)
        c == ',' && depth == 0 && (push!(out, s[start:k-1]); start = k + 1)
    end
    push!(out, s[start:end])
    out
end

_parselist(x) = x[1] == '{' ? [_parselist(w) for w in _commasplit(x[2:end-1])] : parse(BigFloat, x)

function _split_top_level_blocks(s)
    s = replace(replace(s, " " => ""), "\n" => "")
    startswith(s, "{") && endswith(s, "}") && (s = s[2:end-1])
    blocks = String[]
    depth = start = 0
    for (k, c) in enumerate(s)
        if c == '{'
            depth += 1
            depth == 1 && (start = k)
        elseif c == '}'
            depth == 1 && push!(blocks, s[start:k])
            depth -= 1
        end
    end
    blocks
end

function _parse_one_block(raw::AbstractString, Field)
    obj = _parselist(strip(raw))
    if obj isa Vector && !isempty(obj) && obj[1] isa Vector
        n = length(obj)
        A = Matrix{arb}(undef, n, n)
        for i in 1:n, j in 1:length(obj[i])
            A[i, j] = Field(obj[i][j])
        end
        A
    elseif obj isa Vector
        n = length(obj)
        A = Matrix{arb}(undef, n, n)
        for i in 1:n
            v = obj[i]
            A[i, i] = Field(v isa Vector ? v[1] : v)
        end
        A
    else
        Matrix{arb}(Field[Field(obj)])
    end
end

function _parse_y_matrices(raw, Field)
    [_parse_one_block(b, Field) for b in _split_top_level_blocks(raw)]
end

function read_sdpa_output(out_path::AbstractString, Field)
    status, pobj = "", zero(Field)
    ystr = ""
    mode = false
    for line in eachline(out_path)
        startswith(line, "phase.value = ") && (status = strip(split(line, " = ")[2]))
        startswith(line, "objValPrimal = ") && (pobj = Field(parse(BigFloat, split(line, " = ")[2])))
        startswith(line, "yMat =") && (mode = true; continue)
        startswith(line, "    main loop time") && (mode = false)
        mode && (ystr *= strip(line))
    end
    status, pobj, _parse_y_matrices(ystr, Field)
end

mat_entry(mats, b, i, j) = mats[b][i, j]

function eval_constraint_lhs(c::LinearConstraint, mats)
    s = zero(mats[1][1, 1])
    for ((b, i, j), coeff) in c.terms
        s += coeff * mat_entry(mats, b, i, j)
    end
    s + mat_entry(mats, c.slack_block, 1, 1)
end

function verify_solution!(prob::KPointSDP, mats; atol = 1e-12, verbose = false)
    tol = prob.field(atol)
    for (bi, M) in enumerate(mats)
        arbcholesky(M)[1] == 0 && error("ARB PSD check failed on block $bi")
    end
    verbose && println("ARB OK: all ", length(mats), " PSD blocks pass Cholesky")
    for (i, c) in enumerate(prob.constraints)
        lhs = eval_constraint_lhs(c, mats)
        lhs + c.rhs > tol && error("ARB constraint $i violated: lhs+rhs=$(lhs + c.rhs) > tol=$tol")
    end
    verbose && println("ARB OK: all ", length(prob.constraints), " linear constraints satisfied")
end

function solve_kpoint_sdp(prob::KPointSDP, solver::SDPAGMPOptions = SDPAGMPOptions(); compress::Bool = true)
    # Compress sparse indices, then write SDPA .dat
    cprob = compress ? compress_kpoint_sdp(prob) : prob
    if compress && solver.verbose
        before = compression_scalar_vars(prob)
        after = compression_scalar_vars(cprob)
        println("PSD compress: Σ n² $before → $after  (blocks $(length(prob.registry.sizes)))")
    end
    dat, out, par = tempname() * ".dat", tempname() * ".out", tempname() * ".par"
    try
        write_sdpa_sparse(dat, cprob, solver.precision)
        write_sdpa_param(par, solver)
        isfile(solver.executable) || error("sdpa_gmp not found: $(solver.executable)")
        cmd = `$(solver.executable) -ds $(dat) -o $(out) -p $(par)`
        solver.verbose && println("Running: ", cmd)
        run(cmd)
        status, pobj, mats = setprecision(solver.precision) do
            read_sdpa_output(out, prob.field)
        end
        if solver.verbose
            println("sdpa phase.value = ", status)
            status != "pdOPT" && println("  sdpa objValPrimal = ", pobj)
        end
        isempty(mats) && error("sdpa_gmp produced no yMat (phase = $status)")
        length(mats) == length(cprob.registry.sizes) ||
            error("SDPA returned $(length(mats)) blocks, expected $(length(cprob.registry.sizes))")
        if status == "pdOPT"
            verify_solution!(cprob, mats; verbose = solver.verbose)
        elseif solver.verbose
            println("Skipping ARB verify (sdpa status = $status)")
        end
        status != "pdOPT" && @warn "sdpa status is $status (bound may be invalid)"
        mat_entry(mats, prob.alpha_block, 1, 1), status
    finally
        for f in (dat, out, par)
            isfile(f) && rm(f, force = true)
        end
    end
end

function solve_k_point_bound(
    n::Int, k::Int, d::Int, D::Vector;
    tpl::Union{KPointTemplate, Nothing} = nothing,
    prec::Int = ARB_PREC,
    verbose::Bool = true,
    solve::Bool = true,
    solver::SDPAGMPOptions = SDPAGMPOptions(verbose = verbose),
)
    verbose && println("Thesis k-point (Thm 4.5)  n=$n k=$k d=$d  D=$D")
    verbose && println("  ARB prec=$prec  sdpa=$(solver.precision)")

    if !solve
        tpl_out, _, _, _ = inspect_kpoint_sdp!(n, k, d, D; tpl = tpl, prec = prec, verbose = verbose)
        return nothing, nothing, tpl_out
    end

    if tpl === nothing
        t0 = time()
        tpl = build_kpoint_template(k, d, D; prec = prec, verbose = verbose)
        verbose && println("Template build: $(round(time() - t0; digits = 1)) s")
    end

    t1 = time()
    prob = fill_kpoint_sdp(n, tpl; prec = prec, verbose = verbose)
    verbose && println("Fill (n=$n): $(round(time() - t1; digits = 1)) s")

    t2 = time()
    alpha_arb, status = solve_kpoint_sdp(prob, solver)
    verbose && println("Solve: $(round(time() - t2; digits = 1)) s")

    alpha_val = Float64(alpha_arb)
    bound = Int(floor(alpha_val))
    verbose && println("Computed alpha = $alpha_val, floor = $bound ($status)")
    bound, alpha_val, tpl
end

function run_example!()
    n, k, d = EXAMPLE_N, EXAMPLE_K, EXAMPLE_DEG
    D = example_D()
    println("=== fast_KPoint_bound_f  n=$n k=$k d=$d  (Thm 4.5 + Remark 4.8) ===\n")

    t0 = time()
    tpl = build_kpoint_template(k, d, D; verbose = true)
    println("Template build: $(round(time() - t0; digits = 1)) s\n")

    t1 = time()
    prob = fill_kpoint_sdp(n, tpl; verbose = true)
    println("Fill (n=$n): $(round(time() - t1; digits = 1)) s\n")

    t2 = time()
    alpha_arb, status = solve_kpoint_sdp(prob, SDPAGMPOptions(verbose = true))
    println("Solve: $(round(time() - t2; digits = 1)) s")

    alpha = Float64(alpha_arb)
    bound = Int(floor(alpha))
    println("status = $status")
    println("bound = $bound, alpha = $alpha")
    bound, alpha
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using .FastKPointBoundF
    run_example!()
end
