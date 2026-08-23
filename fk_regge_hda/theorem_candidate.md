# Perturbative fourth-order restoration theorem candidate

## Scope and normalization

The proof is organized in two stages.

1. A **covariant/intermediate-slice theorem**: the Regge Bianchi identity forces the corrected gauge row of the Hessian at an interior stationary vertex to be O(h^4).
2. A **canonical bridge**: the standard tent-move identity of Bahr--Dittrich transfers this estimate to the old/new mixed Lagrangian Hessian, under a bounded stationary-response hypothesis.

This distinction is essential because derivatives of a Hamilton principal function with respect to boundary data are boundary momenta and are not set to zero by the bulk equations of motion.

Work with normalized edge variables \(\hat l=l/h\) and dimensionless actions \(\widehat S=h^{-2}S\). This removes trivial length dimensions from the Hessians.

## Geometric hypotheses

Let \(T_h\) be a family of four-dimensional FK/Regge tent complexes approximating a smooth metric \(g\) in a convex normal neighborhood. Assume:

1. **Smooth background:** \(g\in C^8\), with uniform derivative bounds sufficient for the required world-function expansion. This is deliberately conservative.
2. **Shape regularity / fatness:** after rescaling each simplex by \(h^{-1}\), all edge-Gram eigenvalues lie in a fixed interval \([\lambda_*,\lambda^*]\subset(0,\infty)\).
3. **Tent-pole nondegeneracy:** internal tent variables are locally eliminated by stationarity and the corresponding internal Hessian blocks have singular values bounded below by \(\tau>0\).
4. **Curvature scaling:** hinge holonomy/deficit variables obey
   \[
   \epsilon_h=h^2R+h^3G+O(h^4),
   \qquad
   \delta_Y\epsilon_h=h^2A_Y+h^3B_Y+O(h^4),
   \]
   uniformly on each local star.
5. **Uniform reduced-Hessian expansion:** after tent-pole elimination, the middle-slice Hessian has
   \[
   H_h=H_0+h^2H_2+h^3H_3+h^4H_4+R_5(h),
   \qquad \|R_5(h)\|\le C_5h^5.
   \]
6. **Flat gauge splitting:** \(H_0\) has the four expected vertex-displacement gauge directions. In a left/right physical-gauge adapted basis, its physical block \(A_0\) obeys
   \[
   \sigma_{\min}(A_0)\ge\gamma>0.
   \]
7. **Uniform coefficient bounds:** \(\|H_j\|\le C_j\) for \(j=2,3,4\).

A separate bounded-response hypothesis is introduced only when passing to the canonical mixed matrix.

# I. Covariant/intermediate-slice theorem

Consider two consecutive tent moves. After eliminating the tent poles, let

\[
\widetilde S_h(l^{n-1},l^n,l^{n+1})
\]

be the effective two-step Regge action. The middle-slice variables \(l^n\) are bulk variables and satisfy the stationary Regge equations

\[
E_h:=\frac{\partial\widetilde S_h}{\partial l^n}=0.
\]

Let

\[
H_h:=\frac{\partial^2\widetilde S_h}{\partial l^n\partial l^n}
\]

be the middle-slice Hessian.

## Lemma 1: contracted Bianchi identity gives an O(h^4) gauge-gauge block

For a normalized vertex-displacement field \(Y_A\), define

\[
B_A=Y_A\cdot E_h.
\]

The exact differential identity is

\[
Y_B[B_A]=(\nabla_{Y_B}Y_A)\cdot E_h+Y_A^TH_hY_B.
\]

At a stationary middle slice, \(E_h=0\), hence

\[
Y_A^TH_hY_B=Y_B[B_A].
\]

For a hinge triangle \(h=(v,a,b)\), write

\[
u=a-v,\qquad w=b-v,\qquad d_h=a-b,\qquad U_h=\frac{u\wedge w}{|u\wedge w|}.
\]

The area-gradient identity is

\[
\nabla_vA_h=\frac12 U_h\cdot d_h.
\]

Using the Schlaefli-reduced Regge equations,

\[
Y_\xi\cdot E_h
=
\frac12\xi_\mu\sum_{h\supset v}\epsilon_hU_h^{\mu\nu}(a-b)_\nu.
\]

The linearized contracted Regge Bianchi identity annihilates the term linear in the deficits; the exact nonlinear identity contains quadratic and higher curvature terms. Hence

\[
B_A=O(\epsilon^2).
\]

Because \(\epsilon_h=O(h^2)\) and \(\delta_Y\epsilon_h=O(h^2)\),

\[
Y_B[B_A]=O(h^4),
\]

and therefore

\[
\boxed{Y_A^TH_hY_B=O(h^4)}.
\]

In particular, if \(P_L,P_R\) project onto the flat left/right gauge spaces,

\[
P_LH_hP_R=O(h^4),
\]

so coefficient comparison gives

\[
\boxed{P_LH_2P_R=0},
\qquad
\boxed{P_LH_3P_R=0}.
\]

## Lemma 2: corrected gauge rows and columns

In a physical/gauge adapted block basis,

\[
H_0=\begin{pmatrix}A_0&0\\0&0\end{pmatrix}.
\]

The O(h^2) and O(h^3) gauge-to-physical mixing can be removed perturbatively. Corrected right gauge vectors satisfy

\[
X_R(h)=X_R^{(0)}+h^2X_R^{(2)}+h^3X_R^{(3)}+O(h^4),
\]

with

\[
X_R^{(2)}=-H_0^+H_2X_R^{(0)},
\qquad
X_R^{(3)}=-H_0^+H_3X_R^{(0)}.
\]

Analogously, corrected left gauge vectors satisfy

\[
X_L(h)^T=X_L^{(0)T}+h^2X_L^{(2)T}+h^3X_L^{(3)T}+O(h^4),
\]

with the corresponding left Schur corrections.

The full corrected row/column residuals are O(h^4), not merely their gauge-gauge matrix elements. Their leading gauge obstruction is the Schur term

\[
\boxed{
\mathcal G_4^{\rm eff}
=
P_L\left(H_4-H_2H_0^+H_2\right)P_R.
}
\]

The h^3 curvature-gradient sector changes the corrected gauge vectors and the h^5 remainder, but does not enter \(\mathcal G_4^{\rm eff}\).

A symbolic 2-physical + 1-gauge calculation verifies zero corrected-row coefficients through h^3 and exactly the Schur coefficient at h^4.

## Lemma 3: internal tent-pole elimination preserves the quadratic estimate

For a full Hessian partitioned into boundary/intermediate variables B and eliminated internal variables I,

\[
H_{\rm full}=\begin{pmatrix}H_{BB}&H_{BI}\\H_{IB}&H_{II}\end{pmatrix},
\]

the stationary reduced Hessian is

\[
H_{\rm red}=H_{BB}-H_{BI}H_{II}^{-1}H_{IB}.
\]

For the stationary lift

\[
Y_I=-H_{II}^{-1}H_{IB}Y_B,
\]

one has exactly

\[
Y_{\rm full}^TH_{\rm full}Y_{\rm full}=Y_B^TH_{\rm red}Y_B.
\]

Therefore the Bianchi/Hessian order survives exact tent-pole elimination.

## Uniform local bound

Weyl's inequality gives, if \(\|A_h-A_0\|\le C_Ah^2\),

\[
\sigma_{\min}(A_h)\ge\gamma-C_Ah^2.
\]

For \(h^2\le\gamma/(2C_A)\),

\[
\|A_h^+\|\le\frac2\gamma.
\]

At flat order,

\[
\|X^{(2)}\|\le\frac{C_2}{\gamma},
\qquad
\|X^{(3)}\|\le\frac{C_3}{\gamma},
\]

and

\[
\boxed{
\|\mathcal G_4^{\rm eff}\|
\le C_4+\frac{C_2^2}{\gamma}.
}
\]

A convenient corrected-row estimate is

\[
\|X_L(h)^TH_h\|
\le
h^4\left(C_4+\frac{C_2^2}{\gamma}\right)
+h^5\left(C_5+\frac{2C_2C_3}{\gamma}\right)
+O(h^6).
\]

# II. Canonical mixed-matrix bridge

Let the middle-slice stationary solution be

\[
l^n=L_h(l^{n-1},l^{n+1}).
\]

Differentiating the middle-slice equations with respect to the upper boundary gives the standard tent-move identity

\[
\boxed{H_h\,\mathscr L_h=-K_h},
\]

where

\[
\mathscr L_h:=\frac{\partial L_h}{\partial l^{n+1}},
\qquad
K_h:=\frac{\partial^2\widetilde S_{(n,n+1)}}{\partial l^n\partial l^{n+1}}.
\]

The matrix \(K_h\) is the discrete Lagrangian two-form / mixed canonical matrix controlling the Legendre transform. This is the precise covariant-to-canonical bridge derived in the tent-move literature.

## Additional canonical-response hypothesis

Assume there is a chosen smooth stationary branch (or gauge-fixed continuation from the flat family) for which

\[
\boxed{\|\mathscr L_h\|\le C_L}
\]

uniformly for sufficiently small h.

Then the corrected left gauge row obeys

\[
X_L(h)^TK_h
=-X_L(h)^TH_h\mathscr L_h,
\]

so

\[
\boxed{
\|X_L(h)^TK_h\|
\le
C_L\left(C_4+\frac{C_2^2}{\gamma}\right)h^4+O(h^5).
}
\]

The time-reversed/lower-boundary relation gives the analogous corrected right-gauge estimate.

Thus, **under the bounded stationary-response hypothesis**, the canonical mixed matrix inherits the fourth-order restoration rate.

This bounded-response condition is now the principal analytic hypothesis still requiring either a geometric proof for the FK refinement family or an explicit theorem assumption plus numerical validation.

# III. Fixed-volume weak assembly

Define the global defect functional using the chosen cell quadrature,

\[
\mathcal E_h[N,M]=h^3\sum_{v\in\Lambda_h} e_v[N,M],
\]

where the local coefficient obeys the O(h^4) canonical estimate and the smearing interpolation is stable for bounded C^1 smearings. Shape regularity gives O(h^{-3}) cells in a fixed physical three-volume V. Therefore

\[
\boxed{
|\mathcal E_h[N,M]|
\le
C\,V\,\|N\|_{C^1}\|M\|_{C^1}\,h^4+O(h^5).
}
\]

This corollary refers to this explicitly defined weak/cell-integrated norm; it is not a claim about every possible discrete norm.

# IV. FK structure-function corollary

For the unweighted FK spatial symbol

\[
A=\begin{pmatrix}4&2&2\\2&4&2\\2&2&4\end{pmatrix},
\qquad g=A^{-1},
\]

the geometric normal-deformation commutator has the HDA tensor type

\[
\beta^i=g^{ij}(N\partial_jM-M\partial_jN)=A^{ij}(N\partial_jM-M\partial_jN).
\]

The matched adjacent x/y/z FK controls verify this structure-function direction directly. This kinematical structure-function identification is distinct from the covariant Bianchi theorem and from the canonical bounded-response bridge.

# Claim boundary

The proved analytic core is:

- exact Regge Bianchi + smooth small-curvature scaling => corrected middle-slice gauge Hessian residual O(h^4);
- the leading local obstruction is the Schur term \(\mathcal G_4^{\rm eff}\);
- exact tent-pole stationary reduction preserves this estimate.

The canonical mixed-matrix conclusion additionally assumes a uniformly bounded stationary-response Jacobian \(\mathscr L_h\). Until that condition is proved or explicitly retained as a theorem hypothesis, do not label the full canonical O(h^4) statement unconditional.

This is not a universal exact finite-h or unrestricted off-shell Dirac-algebra theorem.

# Remaining work

1. Prove or explicitly validate \(\|\mathscr L_h\|\le C_L\) for the matched FK refinement family.
2. Cite the exact tent-move equations giving \(H\mathscr L=-K\) and the quadratic-curvature symmetry-breaking result.
3. State the final weak norm and smearing space in the manuscript body.
4. Perform an independent adversarial proof audit before closing SPE-11.
