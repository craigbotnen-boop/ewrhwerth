using DynamicPolynomials
using TSSOS
using JuMP
using SCS
using LinearAlgebra
using Printf
using SHA

# Gate 019R4: first CS-TSSOS scout for a fixed-curvature K8 fiber.
# Fresh deterministic exact-rational target.  This does NOT reuse an unsaved prior target.

const N = 7
const EDGES = [(i,j) for i in 1:N for j in i+1:N]
const TRIS  = [(i,j,k) for i in 1:N for j in i+1:N for k in j+1:N]
const EIDX  = Dict(e => a for (a,e) in enumerate(EDGES))

# Rational stereographic coordinates (t,u) = integer/16, selected offline
# from a deterministic rational search for a reasonably conditioned rank-35 witness.
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

# The weakest left-singular direction of the 35x42 stereographic Jacobian for
# this fresh target is dominated by triangle (2,4,6), index 21 (1-based).
const Q = findfirst(==((2,4,6)), TRIS)
@assert Q == 21

@polyvar nx[1:21] ny[1:21] nz[1:21]
@polyvar lamfree[1:34]
@polyvar mu[1:21]
@polyvar r[1:35]

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

F = Vector{Any}(undef,35)
tri_edges = Vector{NTuple{3,Int}}(undef,35)
for (tt,(i,j,k)) in enumerate(TRIS)
    ea = EIDX[(i,j)]
    eb = EIDX[(j,k)]
    ec = EIDX[(i,k)]
    tri_edges[tt] = (ea,eb,ec)
    F[tt] = det3(nvec(ea), nvec(eb), nvec(ec))
end

lambda = Vector{Any}(undef,35)
qfree = 1
for t in 1:35
    if t == Q
        lambda[t] = 1.0
    else
        lambda[t] = lamfree[qfree]
        qfree += 1
    end
end

# Criticality on the product of spheres: for each edge, the multiplier-weighted
# gradient must be normal to S^2, i.e. g_e = mu_e n_e.
zpoly = 0*nx[1]
g = [[zpoly,zpoly,zpoly] for _ in 1:21]
for t in 1:35
    ea,eb,ec = tri_edges[t]
    A,B,C = nvec(ea), nvec(eb), nvec(ec)
    ga = crossp(B,C) # d det(A,B,C) / dA
    gb = crossp(C,A) # d / dB
    gc = crossp(A,B) # d / dC
    for d in 1:3
        g[ea][d] += lambda[t]*ga[d]
        g[eb][d] += lambda[t]*gb[d]
        g[ec][d] += lambda[t]*gc[d]
    end
end

ineq = Any[]
# Projective multiplier chart: lambda_Q=1 and all other lambda in [-1,1].
for l in lamfree
    push!(ineq, 1 - l^2)
end
# Explicit redundant bounds to make compactness/scaling transparent.
# Each edge lies in five triangles and each cross product has norm <= 1, so |mu_e| <= 5.
for m in mu
    push!(ineq, 25 - m^2)
end
# Each determinant and target lie in [-1,1], so residuals lie in [-2,2].
for rr in r
    push!(ineq, 4 - rr^2)
end

eq = Any[]
# Unit-sphere constraints.
for e in 1:21
    push!(eq, nx[e]^2 + ny[e]^2 + nz[e]^2 - 1)
end
# Lifted residual equations: r_t = F_t - P*_t.  This keeps the objective quadratic
# so an order-2 relaxation is algebraically admissible despite cubic F_t.
for t in 1:35
    push!(eq, r[t] - F[t] + PSTAR[t])
end
# Lagrange criticality equations.
for e in 1:21, d in 1:3
    push!(eq, g[e][d] - mu[e]*nvec(e)[d])
end

obj = sum(rr^2 for rr in r)
vars = [nx; ny; nz; lamfree; mu; r]
pop = [obj; ineq; eq]

open("gate019r4_target.txt","w") do io
    println(io, "Gate 019R4 fresh deterministic target")
    println(io, "chart_index_1based=", Q)
    println(io, "chart_triangle=", TRIS[Q])
    println(io, "variables=", length(vars))
    println(io, "inequalities=", length(ineq))
    println(io, "equalities=", length(eq))
    println(io, "relaxation_order=2")
    for (i,p) in enumerate(PSTAR_R)
        println(io, @sprintf("PSTAR[%02d]=%s", i, string(p)))
    end
end

println("GATE019R4_MODEL vars=$(length(vars)) ineq=$(length(ineq)) eq=$(length(eq)) chart=$(TRIS[Q])")
println("objective_degree=2 max_constraint_degree=3")
flush(stdout)

# SCS scout only. A positive floating-point bound is NOT the final exact proof;
# it is a candidate for later rational SOS/Gram reconstruction.
model = Model(optimizer_with_attributes(SCS.Optimizer,
    "eps_abs" => 1.0e-5,
    "eps_rel" => 1.0e-5,
    "max_iters" => 200000,
    "verbose" => 1))

result_text = ""
try
    t0 = time()
    opt, sol, data = cs_tssos(pop, vars, 2;
        numeq=length(eq),
        CS="MF",
        TS="MD",
        eqTS="MD",
        MomentOne=true,
        solution=false,
        Gram=false,
        QUIET=false,
        model=model)
    elapsed = time()-t0
    result_text = "status=COMPLETED\nopt_lower_bound=$(repr(opt))\nelapsed_seconds=$(elapsed)\n"
    println("GATE019R4_OPT_LOWER_BOUND=", opt)
    println("GATE019R4_ELAPSED_SECONDS=", elapsed)
catch err
    bt = catch_backtrace()
    result_text = "status=ERROR\nerror=$(sprint(showerror,err,bt))\n"
    println("GATE019R4_ERROR")
    showerror(stdout,err,bt)
    println()
end

open("gate019r4_result.txt","w") do io
    write(io,result_text)
end

# Hash the exact target receipt for reproducibility.
target_bytes = read("gate019r4_target.txt")
open("gate019r4_sha256.txt","w") do io
    println(io, bytes2hex(sha256(target_bytes)), "  gate019r4_target.txt")
end

# Trigger stamp v2: workflow is now registered on the default branch.
