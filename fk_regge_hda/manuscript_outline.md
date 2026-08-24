# Manuscript outline

## Working title
Fourth-Order Restoration of the Canonical Gauge Sector in Refined Four-Dimensional Regge Calculus

## 1. Scope and claim boundary
State the result as a perturbative/on-shell Euclidean Regge theorem and computational realization on a shape-regular Freudenthal-Kuhn refinement family. Do not claim exact finite-spacing diffeomorphism symmetry, fourth-order suppression of the full canonical map, or an unrestricted off-shell Dirac algebra.

Headline block hierarchy on the fully stationary FK48 family:

- middle-slice gauge Hessian: H_gg = O(h^4);
- canonical old/new gauge-to-gauge block: K_gg = O(h^4);
- gauge-to-physical mixed block: K_gp = O(h^2);
- stationary response to boundary deformation directions: L Q_g = O(1);
- unrestricted boundary response may scale as O(h^-2).

## 2. Prior-work mechanism and precise novelty
Separate prior art from the new result.

Prior art:
- Hamber-Kagel: exact nonlinear group-valued Regge Bianchi identity; small-deficit contracted form.
- Gentle-Kheyfets-McDonald-Miller: Kirchhoff-like contracted conservation law and refinement power counting.
- Williams: linearized contracted Bianchi identities related explicitly to sums of 4D Regge equations.
- Bahr-Dittrich: curved Regge pseudo-constraints and quadratic lifting of would-be gauge Hessian eigenvalues with deficit angle.
- Dittrich-Hohn: covariant-to-canonical tent/Pachner formalism and mixed-Hessian bridge.
- Bonzom-Dittrich: exact discrete HDA in 3D and flat/homogeneously curved 4D sectors.

New package claimed here:
1. shape-regular FK refinement realization on fully stationary ordinary 4D Regge solutions;
2. explicit Schur-corrected h^4 gauge lifting on the middle slice;
3. direct old/new canonical gauge-to-gauge block K_gg = O(h^4) while K_gp = O(h^2);
4. bounded stationary response specifically on the boundary deformation subspace;
5. direct FK inverse-metric structure-function identification;
6. fixed-volume weak assembly preserving the h^4 gauge-sector rate.

Frame h^4 as a refinement realization of a mechanism anticipated by prior Regge work, not as discovery of quadratic gauge breaking itself.

## 3. FK graph-to-metric dictionary
Derive the principal tensor

A = [[4,2,2],[2,4,2],[2,2,4]]

and the spatial metric g=A^-1. Fix the index convention g^-1=A. State the signature-dependent deformation vector

beta^a = sigma g^{ab}(N partial_b M - M partial_b N),

with sigma=+1 for the Euclidean canonical theorem. The Lorentzian sigma=-1 normal-commutator check is a kinematical appendix/control, not part of the Lorentzian canonical theorem.

## 4. Two-step tent-move canonical framework
Use the effective two-step Regge action with true stationary middle-slice variables. Distinguish bulk stationarity from Hamilton-principal boundary derivatives, which are momenta and are not set to zero.

Define the flat physical/gauge splitting and the old/new mixed Lagrangian Hessian K. Anchor the canonical bridge to Bahr-Dittrich Eqs. (7.9)-(7.11), H L = -K.

## 5. Contracted Regge Bianchi mechanism
For B_A=Y_A.E, use

Y_B[B_A] = (nabla_{Y_B}Y_A).E + Y_A^T H Y_B.

At a stationary middle slice E=0. Derive the hinge area-gradient identity

nabla_v A_h = (1/2) U_h.(a-b),

and map the vertex contraction to the contracted deficit-bivector Bianchi combination.

Cite Hamber-Kagel Eq. (9.18) for the small-deficit 4D contraction and Eq. (9.23) for the exact nonlinear identity. Emphasize that B_A=O(epsilon^2) is consistent with, and historically anticipated by, prior Regge work.

## 6. Generic h^3 sector and Schur-corrected obstruction
Use the generic expansion

H_h = H_0 + h^2 H_2 + h^3 H_3 + h^4 H_4 + O(h^5).

Bianchi implies

P_L H_2 P_R = 0,
P_L H_3 P_R = 0.

Corrected gauge vectors are

X^(2) = -H_0^+ H_2 X^(0),
X^(3) = -H_0^+ H_3 X^(0),

and the leading gauge obstruction is

G_4^eff = P_L (H_4 - H_2 H_0^+ H_2) P_R.

State physical-gap, shape-regularity, tent-pole nondegeneracy, and fourth-order transversality hypotheses precisely.

## 7. Fully stationary FK48 validation
This becomes the principal computational section.

For h=0.09, 0.065, 0.045:
- solve the 12 physical middle equations to machine precision;
- observe the four unsolved pseudo-constraint residuals scale as h^3.995912 before gauge-coordinate adjustment;
- use the reduced 4x4 gauge Jacobian to continue to fully stationary solutions with all 16 equations at ~1e-15;
- show 12 physical singular values remain O(1), with the physical gap near 3.98;
- show the four lifted gauge singular values scale with exponents approximately 3.99931, 3.99933, 4.00188, 4.00651;
- report fourth-order transversality with sigma_min(G_4^eff)/h^4 approximately 0.0535 in the matched family.

## 8. Direct canonical mixed-block test
Differentiate the stationary equations with respect to upper-boundary data and form K.

Use geometric boundary deformation directions Q_g and a physical complement Q_phys. Report:

||U_g^T K Q_g|| = O(h^4),
||U_g^T K Q_phys|| = O(h^2).

With refined finite differences, the four gauge-to-gauge singular-value exponents are approximately
4.00028, 3.99826, 3.99923, 4.01677.

Report the crucial hostile control:

||L|| for arbitrary boundary perturbations is not bounded and behaves approximately as h^-2,

whereas

||L Q_g|| = O(1).

This block separation is part of the result, not a nuisance to hide.

## 9. Direct FK structure-function identification
Compute the normal-deformation commutator on all six shared FK tetrahedra. With g=A^-1 verify

[X_N,X_M]^i = sigma A^{ij}(N partial_j M - M partial_j N).

Include x/y/z permutation controls. Euclidean sigma=+1 is part of the main theorem package. Lorentzian sigma=-1 is a separate kinematical validation.

## 10. Supporting off-shell/local curvature controls
Move the earlier algebraic-world-function experiments here rather than presenting them as theorem-satisfying solutions.

Include:
- local 24-simplex Schur/HVP h^4 fits;
- neighboring same-boundary ordering defect O(h^6), finite-move normalized O(h^4);
- independent full-rank algebraic-curvature replication;
- curvature-amplitude exponent ~2.

Label these explicitly as local/off-shell scaling controls unless stationarity is verified.

## 11. Periodic weak assembly
Define the discrete weak/cell-integrated error norm explicitly. For bounded C^1 smearings on fixed physical volume, show that local h^4 gauge-sector coefficients remain h^4 after summing O(h^-3) cells with h^3 volume weights.

## 12. Relation to rigorous refinement literature
Use shape-regularity language consistent with modern Regge/finite-element convergence work. Cite Christiansen (IMA J. Numer. Anal. 44, 2024, DOI 10.1093/imanum/drad095) and Gawlik-Neunteufel (Math. Comp. 94, 2025, DOI 10.1090/mcom/4038) as refinement/curvature-convergence context, not as proofs of the canonical theorem.

## 13. Off-shell and Lorentzian boundaries
Use Bojowald-Duque-Shah (Phys. Rev. D 111, 124048, 2025) to motivate the on-shell formulation. State explicitly:

- no unrestricted off-shell Dirac-algebra theorem is claimed;
- no Lorentzian stationary FK48 K_gg=O(h^4) theorem is yet claimed;
- Lorentzian normal-deformation kinematics has been checked with the expected sigma=-1 sign;
- a Lorentzian canonical extension requires a controlled causal sector/branch of the Regge action.

## 14. Reproducibility and audit appendix
Preserve failed/retired observables and hostile controls, including:
- difference-action boundary derivative as wrong observable for structure-function direction;
- 4-valent minimal tent as topologically/physically degenerate control;
- symmetric 5-valent tent-pole degeneracy control;
- false stronger hypothesis that the full boundary stationary-response operator is uniformly bounded.

The audit trail is part of the credibility of the final claim.
