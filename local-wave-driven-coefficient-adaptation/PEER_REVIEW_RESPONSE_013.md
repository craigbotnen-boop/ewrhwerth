# Response to the independent peer and literature review

## Manuscript

**Local Wave-Driven Coefficient Adaptation on Metric Networks: Short-Time Well-Posedness and Finite Propagation**

## Controlling status

**MAJOR REVISION IMPLEMENTED - FOCUSED RE-AUDIT REQUESTED**

The revision addresses every major and minor item in the review. Because the main change is theorem-level, the release is not labeled proof-sealed until the new vertex-trace argument is independently checked.

## Major comment 1: differentiated coefficient-dependent scattering

### Revision

A new lemma derives, in the weak second-derivative sense,

\[
\ell_t=S(c)r_t+DS(c)[c_t]r,
\]

and

\[
\ell_{tt}=S(c)r_{tt}
+2DS(c)[c_t]r_t
+DS(c)[c_{tt}]r
+D^2S(c)[c_t,c_t]r.
\]

The manuscript now anchors the first two time jets of the coefficient trace to the fixed initial wave jet. The lower-order scattering terms split into data-determined values plus an \(O(T)\) remainder. The principal operator at derivative orders zero, one, and two is the same bounded matrix \(S(c)\).

The proof also restricts the local interval below the minimum edge traversal time. Every backward characteristic meets at most one vertex, so the trace dependence is triangular and no repeated-junction amplification is hidden.

### Location

Lemmas **4.2-4.5**, especially equations **(25)-(32)**.

## Major comment 2: corner-characteristic patching

### Revision

A characteristic patching lemma now proves that piecewise \(W^{2,\infty}\) regularity plus matching first-order traces gives global \(C^{1,1}\). The differentiated transport equation and first-order compatibility are used to show that the one-sided first-derivative traces agree along each characteristic emitted from a vertex-time or external corner. Bounded jumps in second derivatives therefore create no singular distributional term.

### Location

Lemma **4.4** and the final paragraph of Lemma **4.5**.

## Major comment 3: literature boundary

### Revision

The introduction and discussion now distinguish the present distributed edge ODE from:

- classical quasilinear waves on networks;
- BV and entropy theories for conservation laws and the p-system at junctions;
- dynamic transmission variables located at vertices;
- junction and lumped-parameter ODE coupling;
- prescribed nonautonomous network parameters;
- continuum wave-material feedback.

Seven references were added: Ali Mehmeti; Bressan-Canic-Garavello-Herty-Piccoli; Garavello; Colombo-Garavello; Kramar Fijavz-Mugnolo-Nicaise on dynamic transmission; Borsche-Kall; and Chitour-Mazanti-Sigalotti.

## Major comment 4: numerical method definition

### Revision

The manuscript now states:

- the characteristic equations for \(r\), \(\ell\), and \(a\);
- the second-order one-sided stencils;
- the centered and endpoint formula for \(c_x\);
- projection of scattering and external traces at every SSP-RK stage;
- the coefficient-box projection;
- CFL number \(0.35\);
- the restricted nested-grid Euclidean error;
- the exact outside-cone mask;
- the \(10^{-10}\) detection threshold.

## Minor comments

- Edge-side coefficient traces may differ and are interpreted as edge impedances/wave speeds.
- The condition \(p=0\) is identified as homogeneous reflecting; nonhomogeneous incoming data would enter the support source set.
- The data vector and external trace interpretation remain explicit in the fixed-point proof.
- The empirical coefficient order remains de-emphasized.
- The immutable public code snapshot is retained; no DOI has been fabricated.

## Requested focused re-audit

Please check only:

1. the anchored coefficient-jet estimate;
2. the differentiated trace formulas and the \(O(T)\) remainder bound;
3. the one-vertex trace estimate through derivative order two;
4. the characteristic patching argument.
