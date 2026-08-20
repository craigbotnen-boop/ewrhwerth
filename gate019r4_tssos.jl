using DynamicPolynomials
using TSSOS
using JuMP
using COSMO
using LinearAlgebra
using Printf
using SHA

# Gate 019R4B: cross-check the same order-2 CS-TSSOS relaxation with COSMO.
# The mathematical model/target is unchanged from run #11; only the SDP backend changes.

const N = 7
const EDGES = [(i,j) for i in 1:N for j in i+1:N]
const TRIS  = [(i,j,k) for i in 1:N for j in i+1:N for k in j+1:N]
const EIDX  = Dict(e => a for (a,e) in enumerate(EDGES))

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

tri_edges = [(EIDX[(i,j)], EIDX[(j,k)], EIDX[(i,k)]) for (i,j,k) in TRIS]
F = [1.0*det3(nvec(ea), nvec(eb), nvec(ec)) for (ea,eb,ec) in tri_edges]

# Projective chart lambda_Q = 1, with all other multipliers in [-1,1].
lam(t) = t == Q ? 1 : lamfree[t < Q ? t : t-1]

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

ineq = vcat(
    [1.0 - l^2 for l in lamfree],
    [25.0 - m^2 for m in mu],
    [4.0 - rr^2 for rr in r]
)

eq = vcat(
    [1.0*nx[e]^2 + ny[e]^2 + nz[e]^2 - 1.0 for e in 1:21],
    [r[t] - F[t] + PSTAR[t] for t in 1:35],
    [g[e][d] - 1.0*mu[e]*nvec(e)[d] for e in 1:21 for d in 1:3]
)

obj = sum(1.0*rr^2 for rr in r)
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
    println(io, "solver=COSMO")
    println(io, "solver_eps_abs=1e-5")
    println(io, "solver_eps_rel=1e-5")
    println(io, "solver_time_limit_seconds=4200")
    for (i,p) in enumerate(PSTAR_R)
        println(io, @sprintf("PSTAR[%02d]=%s", i, string(p)))
    end
end

println("GATE019R4_MODEL vars=$(length(vars)) ineq=$(length(ineq)) eq=$(length(eq)) chart=$(TRIS[Q])")
println("objective_degree=2 max_constraint_degree=3 solver=COSMO")
flush(stdout)

# COSMO has its own chordal PSD decomposition enabled by default. Keep the same
# 1e-5 target tolerances as run #11, but impose a 70-minute solver limit so the
# workflow has time to serialize a receipt before GitHub's 90-minute job timeout.
model = Model(optimizer_with_attributes(COSMO.Optimizer,
    "eps_abs" => 1.0e-5,
    "eps_rel" => 1.0e-5,
    "max_iter" => 100000,
    "time_limit" => 4200.0,
    "verbose" => true,
    "verbose_timing" => true))

function run_scout(pop, vars, eq, model)
    t0 = time()
    try
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
        term = termination_status(model)
        pstat = primal_status(model)
        dstat = dual_status(model)
        result_text = "status=COMPLETED\nsolver=COSMO\ntermination_status=$(term)\nprimal_status=$(pstat)\ndual_status=$(dstat)\nopt_lower_bound=$(repr(opt))\nelapsed_seconds=$(elapsed)\n"
        println("GATE019R4_TERMINATION_STATUS=", term)
        println("GATE019R4_PRIMAL_STATUS=", pstat)
        println("GATE019R4_DUAL_STATUS=", dstat)
        println("GATE019R4_OPT_LOWER_BOUND=", opt)
        println("GATE019R4_ELAPSED_SECONDS=", elapsed)
        return result_text
    catch err
        elapsed = time()-t0
        bt = catch_backtrace()
        result_text = "status=ERROR\nsolver=COSMO\nelapsed_seconds=$(elapsed)\nerror=$(sprint(showerror,err,bt))\n"
        println("GATE019R4_ERROR")
        showerror(stdout,err,bt)
        println()
        return result_text
    end
end

result_text = run_scout(pop, vars, eq, model)

open("gate019r4_result.txt","w") do io
    write(io,result_text)
end

target_bytes = read("gate019r4_target.txt")
open("gate019r4_sha256.txt","w") do io
    println(io, bytes2hex(sha256(target_bytes)), "  gate019r4_target.txt")
end
