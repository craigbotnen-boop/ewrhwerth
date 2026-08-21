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

# Gate 019R4C2: rescaled MomentOne certificate-readiness gate.
# Run #16 proved that lower=0 is explicitly feasible in the sparse SOS template
# and that the MomentOne=false relaxation sits numerically at zero. The only
# earlier hint of positive separation came from the MomentOne=true relaxation,
# so this gate restores MomentOne while keeping the 1e4 objective rescaling.
# If an optimal/almost-optimal SOS solution is returned, serialize the Gram
# matrices, equality multipliers, and moment data needed for exact follow-up.

const N = 7
const EDGES = [(i,j) for i in 1:N for j in i+1:N]
const TRIS  = [(i,j,k) for i in 1:N for j in i+1:N for k in j+1:N]
const EIDX  = Dict(e => a for (a,e) in enumerate(EDGES))
const OBJ_SCALE = 1.0e4

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

base_obj = sum(1.0*rr^2 for rr in r)
obj = OBJ_SCALE * base_obj
vars = [nx; ny; nz; lamfree; mu; r]
pop = [obj; ineq; eq]

# ---------- Exact/combinatorial sanity audit of the sparse SOS template ----------
# TSSOS's primal SDP seeks obj-lower = SOS + inequality SOS multipliers + equality
# multipliers. At lower=0, obj itself is the explicit SOS OBJ_SCALE*sum(r_i^2).
# We verify that every linear monomial r_i is present in a base-SOS block, so the
# chosen sparse template really contains this known feasible point.
function find_base_sos_slot(data, vid::Int)
    target = UInt16[vid]
    for i in eachindex(data.cliques)
        j = findfirst(==(1), data.I[i]) # ineq_cons[1] is the constant polynomial 1
        j === nothing && continue
        for (l, block) in enumerate(data.blocks[i][j])
            for k in block
                if data.basis[i][j][k] == target
                    return (clique=i, local_ineq=j, block=l, basis_index=k)
                end
            end
        end
    end
    return nothing
end

println("GATE019R4C2_STRUCTURE_AUDIT_START")
_, _, structure = cs_tssos(pop, vars, 2;
    numeq=length(eq), CS="MF", TS="MD", eqTS="MD",
    MomentOne=true, solution=false, Gram=false, QUIET=true, solve=false)

r_first = length(vars) - length(r) + 1
slots = [find_base_sos_slot(structure, r_first + k - 1) for k in 1:length(r)]
known_lower0_feasible = all(!isnothing, slots)
missing_r = [k for k in 1:length(r) if isnothing(slots[k])]
println("GATE019R4C2_KNOWN_LOWER0_FEASIBLE=", known_lower0_feasible)
println("GATE019R4C2_MISSING_R_SLOTS=", missing_r)
println("GATE019R4C2_OBJECTIVE_SCALE=", OBJ_SCALE)
flush(stdout)

open("gate019r4_target.txt","w") do io
    println(io, "Gate 019R4C2 rescaled MomentOne certificate-readiness target")
    println(io, "chart_index_1based=", Q)
    println(io, "chart_triangle=", TRIS[Q])
    println(io, "variables=", length(vars))
    println(io, "inequalities=", length(ineq))
    println(io, "equalities=", length(eq))
    println(io, "relaxation_order=2")
    println(io, "objective_scale=", OBJ_SCALE)
    println(io, "solver=SCS")
    println(io, "moment_one=true")
    println(io, "known_lower0_feasible=", known_lower0_feasible)
    println(io, "missing_r_slots=", missing_r)
    for (i,p) in enumerate(PSTAR_R)
        println(io, @sprintf("PSTAR[%02d]=%s", i, string(p)))
    end
end

println("GATE019R4_MODEL vars=$(length(vars)) ineq=$(length(ineq)) eq=$(length(eq)) chart=$(TRIS[Q])")
println("objective_degree=2 max_constraint_degree=3 solver=SCS objective_scale=$(OBJ_SCALE) MomentOne=true")
flush(stdout)

model = Model(optimizer_with_attributes(SCS.Optimizer,
    "eps_abs" => 1.0e-5,
    "eps_rel" => 1.0e-5,
    "max_iters" => 40000,
    "verbose" => 1))

function run_scout(pop, vars, eq, model, known_lower0_feasible)
    t0 = time()
    try
        opt, sol, data = cs_tssos(pop, vars, 2;
            numeq=length(eq),
            CS="MF",
            TS="MD",
            eqTS="MD",
            MomentOne=true,
            solution=false,
            Gram=true,
            QUIET=false,
            model=model)
        elapsed = time()-t0
        term = termination_status(model)
        pstat = primal_status(model)
        dstat = dual_status(model)
        unscaled = isfinite(opt) ? opt / OBJ_SCALE : opt
        solver_false_infeas = known_lower0_feasible && string(term) == "INFEASIBLE"
        cert_saved = false
        if string(term) in ("OPTIMAL", "ALMOST_OPTIMAL") && data.GramMat !== nothing
            cert = (
                chart=TRIS[Q], objective_scale=OBJ_SCALE,
                opt_scaled=opt, opt_unscaled=unscaled,
                termination_status=string(term),
                primal_status=string(pstat), dual_status=string(dstat),
                GramMat=data.GramMat, multiplier=data.multiplier,
                moment=data.moment, ksupp=data.ksupp,
                basis=data.basis, ebasis=data.ebasis,
                blocks=data.blocks, eblocks=data.eblocks,
                I=data.I, J=data.J, cliques=data.cliques,
                PSTAR_R=PSTAR_R
            )
            serialize("gate019r4_numeric_certificate.jls", cert)
            cert_saved = true
        end
        result_text = "status=COMPLETED\nsolver=SCS\ntermination_status=$(term)\nprimal_status=$(pstat)\ndual_status=$(dstat)\nmoment_one=true\nknown_lower0_feasible=$(known_lower0_feasible)\nsolver_false_infeasibility=$(solver_false_infeas)\nopt_scaled=$(repr(opt))\nopt_unscaled=$(repr(unscaled))\ncertificate_saved=$(cert_saved)\nelapsed_seconds=$(elapsed)\n"
        println("GATE019R4_TERMINATION_STATUS=", term)
        println("GATE019R4_PRIMAL_STATUS=", pstat)
        println("GATE019R4_DUAL_STATUS=", dstat)
        println("GATE019R4_SOLVER_FALSE_INFEASIBILITY=", solver_false_infeas)
        println("GATE019R4_OPT_SCALED=", opt)
        println("GATE019R4_OPT_UNSCALED=", unscaled)
        println("GATE019R4_CERTIFICATE_SAVED=", cert_saved)
        println("GATE019R4_ELAPSED_SECONDS=", elapsed)
        return result_text
    catch err
        elapsed = time()-t0
        bt = catch_backtrace()
        result_text = "status=ERROR\nsolver=SCS\nmoment_one=true\nknown_lower0_feasible=$(known_lower0_feasible)\nelapsed_seconds=$(elapsed)\nerror=$(sprint(showerror,err,bt))\n"
        println("GATE019R4_ERROR")
        showerror(stdout,err,bt)
        println()
        return result_text
    end
end

result_text = run_scout(pop, vars, eq, model, known_lower0_feasible)

open("gate019r4_result.txt","w") do io
    write(io,result_text)
end

target_bytes = read("gate019r4_target.txt")
open("gate019r4_sha256.txt","w") do io
    println(io, bytes2hex(sha256(target_bytes)), "  gate019r4_target.txt")
end
