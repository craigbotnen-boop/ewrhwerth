using DynamicPolynomials
using TSSOS
using JuMP
using Printf
using SHA

# Gate 019R5D: direct 97-variable order-3 Positivstellensatz STRUCTURE scout.
# The 119 polynomial equalities are encoded as paired inequalities h>=0 and
# -h>=0. This defines exactly the same feasible set. A positive rho certificate
# is therefore valid, while failure of this restricted finite-order certificate
# class is inconclusive. Gate 019R5C used TS="MD" and hit a CliqueTrees stack
# overflow during chordal extension, so this scout uses TS="block" (connected
# components of the term graph) to avoid that implementation bottleneck.

const N = 7
const EDGES = [(i,j) for i in 1:N for j in i+1:N]
const TRIS  = [(i,j,k) for i in 1:N for j in i+1:N for k in j+1:N]
const EIDX  = Dict(e => a for (a,e) in enumerate(EDGES))
const ORDER = 3

const TU_NUM = [
    (5,7), (0,-1), (9,-9), (-1,-15), (15,-2), (13,-13), (11,-7),
    (2,-14), (2,-4), (3,8), (13,9), (-6,-6), (-14,11), (-3,12),
    (-8,5), (11,0), (11,3), (5,3), (3,11), (-2,-11), (-3,-14)
]

function stereo_rat(a::Int,b::Int)
    t = BigInt(a)//BigInt(16)
    u = BigInt(b)//BigInt(16)
    d = 1 + t^2 + u^2
    return (2t/d, 2u/d, (1-t^2-u^2)/d)
end

function det3num(a,b,c)
    return a[1]*(b[2]*c[3]-b[3]*c[2]) -
           a[2]*(b[1]*c[3]-b[3]*c[1]) +
           a[3]*(b[1]*c[2]-b[2]*c[1])
end

AXR = [stereo_rat(a,b) for (a,b) in TU_NUM]
PSTAR_R = Rational{BigInt}[]
for (i,j,k) in TRIS
    a = AXR[EIDX[(i,j)]]
    b = AXR[EIDX[(j,k)]]
    c = AXR[EIDX[(i,k)]]
    push!(PSTAR_R, det3num(a,b,c))
end
PSTAR = Float64.(PSTAR_R)

const Q = findfirst(==((2,4,6)), TRIS)
@assert Q == 21

@polyvar nx[1:21] ny[1:21] nz[1:21]
@polyvar lamfree[1:34]
vars = [nx; ny; nz; lamfree]
@assert length(vars) == 97

nvec(e) = [nx[e], ny[e], nz[e]]
function crossp(a,b)
    return [a[2]*b[3]-a[3]*b[2],
            a[3]*b[1]-a[1]*b[3],
            a[1]*b[2]-a[2]*b[1]]
end
function det3(a,b,c)
    return a[1]*(b[2]*c[3]-b[3]*c[2]) -
           a[2]*(b[1]*c[3]-b[3]*c[1]) +
           a[3]*(b[1]*c[2]-b[2]*c[1])
end

tri_edges = [(EIDX[(i,j)], EIDX[(j,k)], EIDX[(i,k)]) for (i,j,k) in TRIS]
F = [1.0*det3(nvec(ea), nvec(eb), nvec(ec)) for (ea,eb,ec) in tri_edges]
lam(t) = t == Q ? 1.0 : lamfree[t < Q ? t : t-1]

function grad_component(e::Int, d::Int)
    s = 0
    for t in 1:35
        ea,eb,ec = tri_edges[t]
        A,B,C = nvec(ea), nvec(eb), nvec(ec)
        lt = lam(t)
        if e == ea
            s = s + lt*crossp(B,C)[d]
        elseif e == eb
            s = s + lt*crossp(C,A)[d]
        elseif e == ec
            s = s + lt*crossp(A,B)[d]
        end
    end
    return 1.0*s
end

g = [[grad_component(e,d) for d in 1:3] for e in 1:21]
stationarity = [crossp(nvec(e), g[e])[d] for e in 1:21 for d in 1:3]

eqraw = vcat(
    [1.0*nx[e]^2 + ny[e]^2 + nz[e]^2 - 1.0 for e in 1:21],
    [F[t] - PSTAR[t] for t in 1:35],
    stationarity
)
@assert length(eqraw) == 119

lambda_bounds = [1.0 - l^2 for l in lamfree]
ineq = vcat(lambda_bounds, eqraw, [-p for p in eqraw])
eq = eltype(eqraw)[]
@assert length(ineq) == 272
@assert isempty(eq)

function run_structure_scout()
    t0 = time()
    model = Model()
    @variable(model, 0 <= rho <= 1)
    nonneg = -rho*(1.0 + nx[1]^2) + rho*nx[1]^2

    println("GATE019R5D_MODEL vars=$(length(vars)) ineq=$(length(ineq)) eq=$(length(eq)) chart=$(TRIS[Q])")
    println("GATE019R5D_ORIGINAL_EQUALITIES=119 paired_inequalities=238")
    println("GATE019R5D_MAX_CONSTRAINT_DEGREE=4 relaxation_order=$(ORDER)")
    println("GATE019R5D_TERM_SPARSITY=block")
    flush(stdout)

    info = add_psatz!(model, nonneg, vars, ineq, eq, ORDER;
        CS="MF", TS="block", eqTS=false, SO=1,
        GroebnerBasis=false, QUIET=false)

    blocksizes = Int[]
    for i in eachindex(info.blocksize), j in eachindex(info.blocksize[i])
        append!(blocksizes, info.blocksize[i][j])
    end
    maxclique = isempty(info.cliquesize) ? 0 : maximum(info.cliquesize)
    maxblock = isempty(blocksizes) ? 0 : maximum(blocksizes)
    nblocks = length(blocksizes)
    psd_scalar_vars = sum(b*(b+1)÷2 for b in blocksizes)
    elapsed = time()-t0

    println("GATE019R5D_MAX_CLIQUE=", maxclique)
    println("GATE019R5D_MAX_BLOCK=", maxblock)
    println("GATE019R5D_PSD_BLOCKS=", nblocks)
    println("GATE019R5D_PSD_SCALAR_VARIABLES=", psd_scalar_vars)
    println("GATE019R5D_AFFINE_IDENTITIES=", length(info.tsupp))
    println("GATE019R5D_JUMP_VARIABLES=", num_variables(model))
    println("GATE019R5D_BUILD_SECONDS=", elapsed)
    flush(stdout)

    open("gate019r4_target.txt","w") do io
        println(io, "Gate 019R5D direct 97D order-3 paired-inequality Positivstellensatz block structure scout")
        println(io, "chart_index_1based=", Q)
        println(io, "chart_triangle=", TRIS[Q])
        println(io, "variables=", length(vars))
        println(io, "inequalities=", length(ineq))
        println(io, "equalities=", length(eq))
        println(io, "original_equalities=119")
        println(io, "paired_inequalities=238")
        println(io, "relaxation_order=", ORDER)
        println(io, "CS=MF")
        println(io, "TS=block")
        println(io, "eqTS=false")
        println(io, "SO=1")
        println(io, "GroebnerBasis=false")
        println(io, "representation=paired_inequalities_exact_same_feasible_set")
        println(io, "finite_order_interpretation=positive_rho_valid_failure_inconclusive")
        for (i,p) in enumerate(PSTAR_R)
            println(io, @sprintf("PSTAR[%02d]=%s", i, string(p)))
        end
    end

    open("gate019r4_result.txt","w") do io
        println(io, "status=COMPLETED")
        println(io, "gate=019R5D")
        println(io, "mode=STRUCTURE_ONLY")
        println(io, "relaxation_order=", ORDER)
        println(io, "term_sparsity=block")
        println(io, "max_clique=", maxclique)
        println(io, "max_block=", maxblock)
        println(io, "psd_blocks=", nblocks)
        println(io, "psd_scalar_variables=", psd_scalar_vars)
        println(io, "affine_identities=", length(info.tsupp))
        println(io, "jump_variables=", num_variables(model))
        println(io, "build_seconds=", elapsed)
    end

    target_bytes = read("gate019r4_target.txt")
    open("gate019r4_sha256.txt","w") do io
        println(io, bytes2hex(sha256(target_bytes)), "  gate019r4_target.txt")
    end
end

try
    run_structure_scout()
catch err
    bt = catch_backtrace()
    open("gate019r4_result.txt","w") do io
        println(io, "status=ERROR")
        println(io, "gate=019R5D")
        println(io, "mode=STRUCTURE_ONLY")
        println(io, "error=", sprint(showerror, err, bt))
    end
    println("GATE019R5D_ERROR")
    showerror(stdout, err, bt)
    println()
end
