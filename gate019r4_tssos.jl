using DynamicPolynomials
using TSSOS
using JuMP
using SCS
using MathOptInterface
using LinearAlgebra
using Printf
using SHA
using Serialization
const MOI = MathOptInterface

# Gate 019R5A: direct 97-variable Positivstellensatz scout for chart (2,4,6).
# Eliminate the 35 residual variables and the 21 radial multipliers mu_e.
# Criticality is imposed directly as n_e x g_e = 0.  Instead of asking a
# feasibility solver for a fixed -1 certificate, maximize rho in [0,1] subject
# to -rho lying in the sparse Putinar quadratic module + equality ideal.
# rho=0 is an explicit feasible point; any robust rho>0 is a direct emptiness
# certificate candidate, and exact rho>0 can be rescaled to a -1 certificate.

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

ineq = [1.0 - l^2 for l in lamfree]
eq = vcat(
    [1.0*nx[e]^2 + ny[e]^2 + nz[e]^2 - 1.0 for e in 1:21],
    [F[t] - PSTAR[t] for t in 1:35],
    stationarity
)
@assert length(ineq) == 34
@assert length(eq) == 119

open("gate019r4_target.txt","w") do io
    println(io, "Gate 019R5A direct 97D Positivstellensatz scout")
    println(io, "chart_index_1based=", Q)
    println(io, "chart_triangle=", TRIS[Q])
    println(io, "variables=", length(vars))
    println(io, "inequalities=", length(ineq))
    println(io, "equalities=", length(eq))
    println(io, "sphere_equalities=21")
    println(io, "target_equalities=35")
    println(io, "cross_stationarity_equalities=63")
    println(io, "relaxation_order=2")
    println(io, "CS=MF")
    println(io, "TS=MD")
    println(io, "eqTS=MD")
    println(io, "SO=1")
    println(io, "GroebnerBasis=false")
    println(io, "solver=SCS")
    println(io, "rho_bounds=[0,1]")
    for (i,p) in enumerate(PSTAR_R)
        println(io, @sprintf("PSTAR[%02d]=%s", i, string(p)))
    end
end

println("GATE019R5A_MODEL vars=$(length(vars)) ineq=$(length(ineq)) eq=$(length(eq)) chart=$(TRIS[Q])")
println("GATE019R5A_MAX_CONSTRAINT_DEGREE=4 relaxation_order=2")
flush(stdout)

model = Model(optimizer_with_attributes(SCS.Optimizer,
    "eps_abs" => 1.0e-5,
    "eps_rel" => 1.0e-5,
    "eps_infeas" => 1.0e-7,
    "max_iters" => 30000,
    "verbose" => 1))

@variable(model, 0 <= rho <= 1)
# Force a DynamicPolynomials polynomial with JuMP affine coefficients that is
# algebraically equal to -rho.
nonneg = -rho*(1.0 + nx[1]^2) + rho*nx[1]^2

result_text = ""
cert_saved = false
try
    t0 = time()
    info = add_psatz!(model, nonneg, vars, ineq, eq, 2;
        CS="MF", TS="MD", eqTS="MD", SO=1,
        GroebnerBasis=false, QUIET=false)

    blocksizes = Int[]
    for i in eachindex(info.blocksize), j in eachindex(info.blocksize[i])
        append!(blocksizes, info.blocksize[i][j])
    end
    maxclique = isempty(info.cliquesize) ? 0 : maximum(info.cliquesize)
    maxblock = isempty(blocksizes) ? 0 : maximum(blocksizes)
    println("GATE019R5A_MAX_CLIQUE=", maxclique)
    println("GATE019R5A_MAX_BLOCK=", maxblock)
    println("GATE019R5A_AFFINE_IDENTITIES=", length(info.tsupp))
    println("GATE019R5A_JUMP_VARIABLES=", num_variables(model))
    flush(stdout)

    @objective(model, Max, rho)
    optimize!(model)
    elapsed = time()-t0
    term = termination_status(model)
    pstat = primal_status(model)
    dstat = dual_status(model)
    rho_val = has_values(model) ? value(rho) : NaN
    candidate = isfinite(rho_val) && rho_val > 1.0e-4 && string(term) in ("OPTIMAL", "ALMOST_OPTIMAL")

    if candidate
        GramMat = [[[value.(info.GramMat[i][j][l]) for l in eachindex(info.GramMat[i][j])]
                    for j in eachindex(info.GramMat[i])] for i in eachindex(info.GramMat)]
        multiplier = [[value.(info.multiplier[i][j]) for j in eachindex(info.multiplier[i])]
                      for i in eachindex(info.multiplier)]
        cert = (
            gate="019R5A", chart=TRIS[Q], rho=rho_val,
            termination_status=string(term), primal_status=string(pstat), dual_status=string(dstat),
            GramMat=GramMat, multiplier=multiplier,
            basis=info.basis, ebasis=info.ebasis, blocks=info.blocks, eblocks=info.eblocks,
            I=info.I, J=info.J, cliques=info.cliques, tsupp=info.tsupp,
            PSTAR_R=PSTAR_R
        )
        serialize("gate019r4_numeric_certificate.jls", cert)
        cert_saved = true
    end

    result_text = "status=COMPLETED\ngate=019R5A\nsolver=SCS\ntermination_status=$(term)\nprimal_status=$(pstat)\ndual_status=$(dstat)\nrho=$(repr(rho_val))\npositive_certificate_candidate=$(candidate)\ncertificate_saved=$(cert_saved)\nmax_clique=$(maxclique)\nmax_block=$(maxblock)\naffine_identities=$(length(info.tsupp))\njump_variables=$(num_variables(model))\nelapsed_seconds=$(elapsed)\n"
    println("GATE019R5A_TERMINATION_STATUS=", term)
    println("GATE019R5A_PRIMAL_STATUS=", pstat)
    println("GATE019R5A_DUAL_STATUS=", dstat)
    println("GATE019R5A_RHO=", rho_val)
    println("GATE019R5A_POSITIVE_CERTIFICATE_CANDIDATE=", candidate)
    println("GATE019R5A_CERTIFICATE_SAVED=", cert_saved)
    println("GATE019R5A_ELAPSED_SECONDS=", elapsed)
catch err
    bt = catch_backtrace()
    result_text = "status=ERROR\ngate=019R5A\nsolver=SCS\nerror=$(sprint(showerror,err,bt))\n"
    println("GATE019R5A_ERROR")
    showerror(stdout,err,bt)
    println()
end

open("gate019r4_result.txt","w") do io
    write(io,result_text)
end

target_bytes = read("gate019r4_target.txt")
open("gate019r4_sha256.txt","w") do io
    println(io, bytes2hex(sha256(target_bytes)), "  gate019r4_target.txt")
end
