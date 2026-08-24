# Fourth-Order Restoration of the Canonical Gauge Sector in Refined Four-Dimensional Regge Calculus

## Abstract

Discretizations of general relativity generically break diffeomorphism symmetry, replacing exact canonical constraints by pseudo-constraints away from flat or specially curved sectors. Earlier Regge-calculus analyses established approximate contracted Bianchi identities at small deficit angle and observed quadratic lifting of would-be gauge Hessian modes. Here we study how this mechanism propagates through canonical tent-move evolution under shape-regular refinement of a four-dimensional Freudenthal-Kuhn (FK) simplicial family.

For a near-flat Euclidean Regge refinement family with stationary middle-slice data, a contracted-Bianchi argument implies that the corrected vertex-displacement sector of the reduced middle-slice Hessian has no contributions at orders h^2 or h^3. After eliminating physical mixing, the leading gauge obstruction is the fourth-order Schur term

\[
\mathcal G_4^{\mathrm{eff}}
=
P_L\bigl(H_4-H_2H_0^+H_2\bigr)P_R.
\]

Under a uniform physical spectral gap and fourth-order transversality, this gives an O(h^4) gauge-sector pseudo-constraint defect. We test the result on fully stationary ordinary-Regge FK two-step complexes containing 48 four-simplices. The four lifted gauge singular values of the middle-slice Hessian scale as h^4 while the 12-dimensional physical gap remains O(1). Differentiating the stationary equations to form the old/new mixed Lagrangian Hessian K, we find a sharp block hierarchy: the gauge-to-gauge block satisfies K_gg=O(h^4), whereas gauge-to-physical mixing remains O(h^2). The stationary response is uniformly bounded on boundary deformation directions but not for arbitrary physical boundary perturbations.

Independently, direct normal-deformation commutators on neighboring FK tetrahedra recover the expected inverse-metric structure function,

\[
\beta^a=\sigma g^{ab}(N\partial_bM-M\partial_bN),
\]

with g^{-1}=A determined by the FK graph symbol. A fixed-volume periodic weak assembly preserves the fourth-order gauge-sector rate for bounded smearings.

The result is an on-shell, perturbative Euclidean Regge statement. It does not assert exact finite-spacing diffeomorphism symmetry, fourth-order suppression of the full canonical map, or an unrestricted off-shell Dirac algebra. Lorentzian deformation kinematics is checked separately, while a fully stationary Lorentzian canonical extension is left for future work.

---

## 1. Introduction

The hypersurface-deformation algebra is the canonical imprint of spacetime covariance in general relativity. In continuum ADM variables, the bracket of two normal deformations closes into a tangential deformation whose coefficient contains the inverse spatial metric. A central difficulty in discrete gravity is that a generic discretization does not preserve the full diffeomorphism symmetry at finite resolution. In Regge calculus this is particularly transparent: flat configurations possess exact vertex-displacement symmetries, whereas curvature generically lifts those gauge directions and turns canonical constraints into pseudo-constraints [@BahrDittrich2009; @DittrichHoehn2010].

Several ingredients of the expected refinement behavior are already known. Hamber and Kagel derived an exact nonlinear Regge Bianchi identity in terms of products of finite hinge rotations [@HamberKagel2004]. Gentle, Kheyfets, McDonald and Miller formulated a Kirchhoff-like contracted conservation law and explicitly power-counted its refinement behavior [@GentleEtAl2009]. Williams showed directly, in four-dimensional linearized Regge calculus, how contracted Bianchi relations at a vertex are related to sums of the edge equations of motion [@Williams2012]. Bahr and Dittrich further observed that the small Hessian eigenvalues associated with broken vertex-displacement symmetry vanish quadratically with the deficit angle in curved Regge examples [@BahrDittrich2009]. Since shape-regular refinement of a smooth geometry gives deficits of order h^2, these results already suggest an h^4 scale for the gauge lifting.

What is not supplied by those observations alone is the corresponding canonical statement after physical/gauge reduction. In particular, the old/new mixed Lagrangian Hessian governing the discrete Legendre transformation contains physical-gauge mixing, and one cannot infer its gauge-sector refinement rate by looking only at a local covariant Hessian eigenvalue. Nor is it correct to demand that the full canonical map be fourth-order close to a gauge symmetry: physical boundary perturbations can and do produce parametrically larger responses.

The purpose of this work is to isolate the deformation sector that is relevant to the hypersurface-deformation algebra and to follow the small-curvature Bianchi mechanism through the complete canonical reduction. We use a four-dimensional Freudenthal-Kuhn refinement family for three reasons. First, the spatial graph has a simple and exact graph-to-metric dictionary. Second, its vertex star provides a fixed local combinatorial template under refinement. Third, the resulting tent-move complexes are large enough to contain a nontrivial physical sector but still small enough for high-precision stationary Regge calculations.

Our central result is a block statement rather than a blanket norm estimate. On fully stationary Euclidean FK two-step Regge configurations we find

\[
H_{gg}=O(h^4),
\qquad
K_{gg}=O(h^4),
\qquad
K_{gp}=O(h^2),
\]

where H is the stationary middle-slice Hessian, K is the old/new mixed Lagrangian Hessian, g denotes the four vertex-displacement directions, and p denotes the physical complement. The stationary response to boundary deformation directions is O(1), whereas the unrestricted response to arbitrary physical boundary perturbations scales approximately as h^{-2}. Thus the fourth-order restoration is real but sector-specific.

This distinction also keeps the result compatible with the modern off-shell analysis of hypersurface deformations. Geometric hypersurface deformations and canonical gauge transformations are not identical as purely kinematical off-shell operations; their identification requires constraint and evolution information [@BojowaldDuqueShah2025]. Accordingly, the theorem candidate developed below is an on-shell middle-slice statement and its canonical corollary is restricted to the boundary deformation subspace.

### 1.1 Main contributions

The analysis develops the following analytic chain and tests its consequences on a fully stationary FK refinement family.

1. A contracted Regge Bianchi identity implies that the corrected middle-slice gauge-gauge Hessian has no h^2 or h^3 term.
2. Physical/gauge Schur reduction produces the leading obstruction
   \[
   \mathcal G_4^{\mathrm{eff}}
   =P_L(H_4-H_2H_0^+H_2)P_R.
   \]
3. On a fully stationary 48-simplex FK family, the four lifted gauge singular values of H scale as h^4 while the physical gap stays finite.
4. The canonical old/new gauge-to-gauge block K_gg also scales as h^4; the gauge-to-physical block remains h^2.
5. Direct neighboring-tetrahedron normal commutators recover the inverse-metric structure function of the HDA using the same FK graph metric.
6. A fixed-volume weak periodic assembly preserves the h^4 gauge-sector rate.

The h^4 exponent itself should be viewed as a refinement realization of a mechanism anticipated by earlier Regge work, not as the discovery of quadratic gauge breaking. The new element is the complete on-shell canonical block separation and its direct realization on the FK refinement family.

---

## 2. FK graph metric and deformation structure function

Consider the seven positive FK directions

\[
e_1=(1,0,0),\quad e_2=(0,1,0),\quad e_3=(0,0,1),
\]
\[
e_4=(1,1,0),\quad e_5=(1,0,1),\quad e_6=(0,1,1),\quad e_7=(1,1,1).
\]

For unit conductance in each direction, the principal tensor of the graph Laplacian is

\[
A=\sum_{r=1}^{7}e_r e_r^T
=
\begin{pmatrix}
4&2&2\\
2&4&2\\
2&2&4
\end{pmatrix}.
\]

The spatial covariant metric associated with the continuum operator is

\[
g=A^{-1},
\]

so the inverse metric entering the hypersurface-deformation structure function is exactly A.

For deformation lapses N and M, the continuum normal-normal bracket has tangential component

\[
\beta^a=\sigma g^{ab}(N\partial_bM-M\partial_bN),
\]

where \(\sigma=n^\mu n_\mu\) is the normal-sign convention. Bonzom and Dittrich write the corresponding continuum structure function explicitly as the inverse spatial metric in their Eq. (2.11) [@BonzomDittrich2013].

For the adjacent-x FK tent pair used below, the averaged piecewise-linear lapse covector is proportional to

\[
\xi_x=\left(1,-\frac13,-\frac13\right).
\]

Applying the graph inverse metric gives

\[
A\xi_x=\left(\frac83,0,0\right),
\]

with the y and z cases obtained by permutation. Embedding the simplicial spatial metric with \(L^TL=A^{-1}\), recomputing tetrahedral normals after each infinitesimal deformation, and transforming the resulting tangential displacement back to lattice coordinates gives the same axis-pure result. In Euclidean signature the sign is positive; in Minkowski signature the direct normal-commutator calculation flips by \(\sigma=-1\), as expected.

This structure-function calculation is kinematical. The remaining sections address the harder canonical question: whether the corresponding gauge sector of the Regge evolution map is restored under refinement.

---

## 3. Two-step Regge action and canonical mixed matrix

We consider two consecutive tent moves and eliminate the tent poles by their stationarity equations. The resulting effective action may be written as

\[
\widetilde S_h(l^{n-1},l^n,l^{n+1}),
\]

where the middle-slice edge variables \(l^n\) are genuine bulk variables. They therefore obey

\[
E_h:=\frac{\partial\widetilde S_h}{\partial l^n}=0.
\]

The stationary middle-slice Hessian is

\[
H_h:=\frac{\partial^2\widetilde S_h}{\partial l^n\partial l^n}.
\]

Differentiating the middle-slice stationarity equations with respect to the future-boundary data gives the standard tent-move relation derived by Bahr and Dittrich [@BahrDittrich2009, Eqs. 7.9--7.11],

\[
H_h\,\mathscr L_h=-K_h,
\]

where

\[
\mathscr L_h
=
\frac{\partial l^n_*}{\partial l^{n+1}}
\]

is the stationary-response Jacobian and

\[
K_h
=
\frac{\partial^2\widetilde S_h}
{\partial l^n\partial l^{n+1}}
\]

is the old/new mixed Lagrangian Hessian controlling the discrete Legendre transformation.

A key claim boundary is already visible here. Boundary derivatives of the Hamilton principal function are momenta and are not set to zero. The on-shell Bianchi argument must therefore be made on the stationary middle slice and only then transferred to K through the differentiated stationarity equations.

---

## 4. Contracted Bianchi identity and fourth-order gauge obstruction

Let \(Y_A\) be a normalized vertex-displacement field on the middle slice and define

\[
B_A=Y_A\cdot E_h.
\]

Differentiation along a second displacement field gives the exact identity

\[
Y_B[B_A]
=
(\nabla_{Y_B}Y_A)\cdot E_h
+
Y_A^T H_hY_B.
\]

At a stationary middle slice \(E_h=0\), hence

\[
Y_A^T H_hY_B=Y_B[B_A].
\]

For a hinge triangle \(t=(v,a,b)\), define

\[
u=a-v,\qquad w=b-v,\qquad
U_t=\frac{u\wedge w}{|u\wedge w|}.
\]

The hinge-area gradient with respect to the moved vertex is

\[
\nabla_v A_t
=
\frac12 U_t\cdot(a-b).
\]

Using the Schlaefli-reduced Regge equations, the vertex-displacement contraction becomes a deficit-bivector/edge sum of the same form entering the contracted Regge Bianchi relations. Hamber and Kagel give the small-deficit four-dimensional contraction in Eq. (9.18) and the exact arbitrary-deficit identity in Eq. (9.23) [@HamberKagel2004]. The linear term in the small hinge rotations therefore cancels, leaving

\[
B_A=O(\epsilon^2).
\]

Gentle et al. [@GentleEtAl2009] and Williams [@Williams2012] provide complementary contracted/conservation-law formulations of the same small-deficit mechanism, while Bahr and Dittrich explicitly identify quadratic lifting of would-be gauge modes with deficit angle in curved Regge configurations [@BahrDittrich2009].

For a smooth shape-regular refinement,

\[
\epsilon_h=h^2R+h^3G+O(h^4),
\qquad
\delta_Y\epsilon_h=h^2A_Y+h^3B_Y+O(h^4).
\]

Consequently,

\[
Y_B[B_A]=O(h^4),
\]

and therefore

\[
P_LH_hP_R=O(h^4)
\]

on the flat gauge subspaces. Writing

\[
H_h=H_0+h^2H_2+h^3H_3+h^4H_4+O(h^5),
\]

coefficient comparison gives

\[
P_LH_2P_R=0,
\qquad
P_LH_3P_R=0.
\]

The h^2 and h^3 physical-gauge mixing is removed by corrected left/right gauge vectors. To leading orders,

\[
X_R^{(2)}=-H_0^+H_2X_R^{(0)},
\qquad
X_R^{(3)}=-H_0^+H_3X_R^{(0)},
\]

with analogous left formulas. The first nonvanishing gauge obstruction is then

\[
\boxed{
\mathcal G_4^{\mathrm{eff}}
=
P_L\left(H_4-H_2H_0^+H_2\right)P_R
}.
\]

The h^3 curvature-gradient sector affects the corrected gauge vectors and higher-order remainder but does not modify this fourth-order Schur term.

### 4.1 Hypotheses

For the theorem-level statement we assume:

1. a C^8 smooth Euclidean background on a convex normal neighborhood;
2. a shape-regular FK refinement family with uniform rescaled Gram bounds;
3. nondegenerate tent-pole stationary elimination;
4. the deficit and Hessian expansions above with uniform coefficient bounds;
5. a uniform flat physical gap \(\sigma_{\min}(A_0)\ge\gamma>0\);
6. fourth-order gauge transversality \(\sigma_{\min}(\mathcal G_4^{\mathrm{eff}})\ge\kappa>0\) on the generic curved branch.

The refinement language is consistent with rigorous Regge/finite-element curvature-convergence frameworks [@Christiansen2024; @GawlikNeunteufel2025], although those works do not prove the canonical statement developed here.

Under these hypotheses, the physical pseudoinverse remains uniformly bounded and the corrected middle-slice gauge residual is O(h^4).

---

## 5. Fully stationary FK48 test

The principal numerical test uses an ordinary Euclidean Regge two-step FK complex with 48 four-simplices and 16 internal middle variables after the chosen reduction. The flat middle Hessian has a four-dimensional vertex-displacement kernel and a 12-dimensional physical complement.

For the refinement values

\[
h\in\{0.09,0.065,0.045\},
\]

we first solve the 12 physical equations while fixing the four gauge coordinates. The physical residual is reduced to approximately 3--4 x 10^{-15}, while the remaining four pseudo-constraint components have norms

\[
5.30026\times10^{-7},\quad
1.44565\times10^{-7},\quad
3.32222\times10^{-8},
\]

with

\[
\frac{\|E_g\|}{h^4}
=
0.00807844,\quad0.00809858,\quad0.00810173.
\]

A power-law fit gives

\[
\|E_g\|\propto h^{3.9959124},
\qquad R^2=0.9999997755.
\]

The reduced 4x4 gauge Jacobian is transverse. Its singular values divided by h^4 approach finite nonzero constants, with the smallest approximately 0.0535 and a condition number near 3.16. Newton continuation in the four gauge coordinates then produces a fully stationary branch on which all 16 equations are solved to approximately 10^{-15}.

On that fully stationary branch the 12 physical singular values of H_h remain O(1), with the smallest physical singular value approaching approximately 3.98. The four lifted gauge singular values have fitted refinement exponents

\[
3.99931,\quad3.99933,\quad4.00188,\quad4.00651,
\]

all with R^2 greater than 0.9999997. This is the direct on-shell realization of the fourth-order middle-slice gauge lifting.

---

## 6. Canonical block separation

The canonical test differentiates the fully stationary equations with respect to normalized upper-boundary spoke lengths and forms the mixed Hessian K_h. Let Q_g denote the four geometric deformation directions on the future boundary and Q_p an orthogonal physical complement. Let U_g denote the corrected middle-slice gauge subspace.

The gauge-to-gauge block satisfies

\[
C_h:=U_g^T K_hQ_g=O(h^4).
\]

Using a central finite-difference step of 10^{-6}, the four singular-value fits give exponents

\[
4.00028,\quad3.99826,\quad3.99923,\quad4.01677,
\]

with stable values of \(\sigma(C_h)/h^4\). The operator-norm fit gives

\[
\|C_h\|\propto h^{3.99584}
\]

with R^2 essentially unity.

The larger mixed block behaves differently:

\[
\|U_g^TK_hQ_p\|
\propto h^{1.99764}.
\]

Thus the numerical hierarchy is

\[
\boxed{K_{gg}=O(h^4)},
\qquad
\boxed{K_{gp}=O(h^2)}.
\]

The stationary-response matrix exhibits the corresponding source dependence. For arbitrary boundary perturbations,

\[
\|\mathscr L_h\|
\sim h^{-2},
\]

whereas on the boundary deformation subspace,

\[
\|\mathscr L_hQ_g\|
=
0.33317,\quad0.33351,\quad0.33418,
\]

and is therefore O(1). This is the response required by the canonical HDA sector. The false stronger conjecture that the full stationary-response operator is uniformly bounded is explicitly rejected.

A Lyapunov-Schmidt reduction explains the distinction. After solving the physical middle equations, the gauge equation takes the schematic form

\[
\Psi_h(g,b_g,b_p)
=
h^4\bigl(K_g g+C_g b_g\bigr)
+h^2D_pb_p+\cdots.
\]

If K_g is transverse, boundary gauge forcing gives \(\partial g/\partial b_g=O(1)\), whereas generic physical boundary forcing may give \(\partial g/\partial b_p=O(h^{-2})\). The observed block hierarchy is therefore the expected canonical consequence of fourth-order gauge lifting rather than a numerical inconsistency.

---

## 7. Supporting controls

Several calculations that preceded the fully stationary FK48 test are retained as controls rather than promoted to theorem examples.

1. A generic tilted 24-simplex local Schur calculation gave a corrected gauge defect exponent 3.9997515.
2. A direct neighboring same-boundary ordering calculation gave an action difference scaling as h^6.0002555; after division by two O(h) finite tent displacements the inferred canonical rate was h^4.0002555.
3. An independent full-rank algebraic-curvature replication gave h^6.0030366 for the action-ordering difference and h^4.0030366 after finite-move normalization.
4. Curvature-amplitude scaling was quadratic, with exponent approximately 2.00025.

Because the algebraic world-function controls are not automatically stationary solutions of the vacuum Regge equations, they are treated as local/off-shell scaling controls. The fully stationary FK48 family is the primary theorem-matched validation.

---

## 8. Fixed-volume weak assembly

Let \(e_v[N,M]\) denote the local gauge-sector canonical coefficient at a spatial cell and define the cell-integrated weak defect

\[
\mathcal E_h[N,M]
=h^3\sum_{v\in\Lambda_h}e_v[N,M].
\]

For bounded C^1 smearing interpolation on a fixed physical three-volume V, shape regularity gives O(h^{-3}) cells. If the local gauge-sector coefficient is O(h^4), then

\[
|\mathcal E_h[N,M]|
\le
C V\|N\|_{C^1}\|M\|_{C^1}h^4+O(h^5).
\]

The periodic FK computations reproduce this scaling for constant and smoothly varying local coefficients and for unrelated smooth smearings.

---

## 9. Relation to prior work and claim boundary

The present result sits downstream of several well-established Regge mechanisms rather than replacing them. The exact nonlinear Bianchi identity is due to Hamber and Kagel [@HamberKagel2004]. Approximate contracted conservation laws and refinement power counting were developed by Gentle et al. [@GentleEtAl2009], and Williams [@Williams2012] related the linearized contracted identities directly to the 4D Regge equations. Bahr and Dittrich [@BahrDittrich2009] established curved pseudo-constraints and quadratic lifting of would-be gauge Hessian modes. Dittrich and Hohn [@DittrichHoehn2010] developed the covariant-to-canonical simplicial framework, while Bonzom and Dittrich [@BonzomDittrich2013] constructed exact discrete HDA representations in 3D and in flat or homogeneously curved 4D sectors.

Our contribution is narrower: we trace the small-deficit Bianchi mechanism through physical/gauge Schur reduction and canonical old/new evolution on a shape-regular FK refinement family, and we verify on fully stationary curved 4D Regge configurations that the deformation block K_gg is fourth order even though K_gp remains second order.

The result should therefore not be summarized as "the full discrete Dirac algebra closes at O(h^4)." It is an on-shell statement about the canonical deformation block. This distinction is consistent with the off-shell analysis of Bojowald, Duque and Shah [@BojowaldDuqueShah2025].

The present canonical calculation is Euclidean. A separate Minkowski normal-deformation test reproduces the expected signature factor \(\sigma=-1\) in the structure function, but no fully stationary Lorentzian FK48 K_gg refinement theorem is claimed here.

---

## 10. Conclusion

We have isolated a fourth-order restoration mechanism in the canonical gauge sector of refined four-dimensional Regge calculus. The result is both more limited and more informative than a full-operator convergence claim. Curvature induces O(h^2) physical-gauge mixing, but the Bianchi identities remove the h^2 and h^3 contributions from the corrected gauge-gauge sector. After Schur reduction the first gauge obstruction is O(h^4). On fully stationary FK48 Regge configurations this rate is seen directly in both the middle-slice Hessian and the old/new gauge-to-gauge mixed Lagrangian Hessian, while the gauge-to-physical block remains O(h^2).

Together with the direct FK inverse-metric structure-function calculation and the fixed-volume weak assembly, these results provide a concrete refinement realization of how broken finite-spacing Regge gauge symmetry approaches the canonical hypersurface-deformation structure. The remaining steps are an independent proof audit, publication-quality reproducibility packaging, and a separate Lorentzian canonical analysis in a controlled causal sector.
