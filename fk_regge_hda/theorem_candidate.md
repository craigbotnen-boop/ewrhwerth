# Perturbative fourth-order restoration theorem candidate

## Scope and normalization

The proof has two stages.

1. **Covariant/intermediate-slice theorem:** the Regge Bianchi identity forces the corrected gauge row of the Hessian at an interior stationary vertex to be O(h^4).
2. **Canonical bridge:** the standard tent-move identity transfers this estimate to the old/new mixed Lagrangian Hessian. A generic fourth-order transversality condition supplies the required bounded stationary response.

This distinction is essential because derivatives of a Hamilton principal function with respect to boundary data are boundary momenta and are not set to zero by the bulk equations of motion.

Work with normalized edge variables \(\hat l=l/h\) and dimensionless actions \(\widehat S=h^{-2}S\).

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
8. **Generic fourth-order transversality (for the canonical corollary):** after physical Lyapunov--Schmidt reduction, the leading four-dimensional gauge Jacobian \(\mathcal G_4^{\rm eff}\) satisfies
   \[
   \sigma_{\min}(\mathcal G_4^{\rm eff})\ge\kappa>0.
   \]
   Exact/perfect symmetry sectors with \(\kappa=0\) are treated separately by gauge fixing or exact constraints.

# I. Covariant/intermediate-slice theorem

Consider two consecutive tent moves. After eliminating the tent poles, let

\[
\widetilde S_h(l^{n-1},l^n,l^{n+1})
\]

be the effective two-step Regge action. The middle-slice variables \(l^n\) are bulk variables and satisfy

\[
E_h:=\frac{\partial\widetilde S_h}{\partial l^n}=0.
\]

Let

\[
H_h:=\frac{\partial^2\widetilde S_h}{\partial l^n\partial l^n}.
\]

## Lemma 1: contracted Bianchi identity gives O(h^4) gauge-gauge breaking

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
u=a-v,\quad w=b-v,\quad d_h=a-b,\quad U_h=\frac{u\wedge w}{|u\wedge w|}.
\]

The area-gradient identity is

\[
\nabla_vA_h=\frac12U_h\cdot d_h.
\]

Using the Schlaefli-reduced Regge equations,

\[
Y_\xi\cdot E_h
=
\frac12\xi_\mu\sum_{h\supset v}\epsilon_hU_h^{\mu\nu}(a-b)_\nu.
\]

The linearized contracted Regge Bianchi identity annihilates the term linear in the deficits, while the exact identity contains quadratic and higher curvature terms. Thus

\[
B_A=O(\epsilon^2).
\]

Since \(\epsilon_h=O(h^2)\) and \(\delta_Y\epsilon_h=O(h^2)\),

\[
Y_B[B_A]=O(h^4),
\]

so

\[
\boxed{Y_A^TH_hY_B=O(h^4)}.
\]

Therefore

\[
P_LH_hP_R=O(h^4),
\]

and

\[
\boxed{P_LH_2P_R=0},
\qquad
\boxed{P_LH_3P_R=0}.
\]

## Lemma 2: corrected gauge rows/columns and Schur obstruction

In a physical/gauge adapted basis,

\[
H_0=\begin{pmatrix}A_0&0\\0&0\end{pmatrix}.
\]

Corrected right gauge vectors have

\[
X_R(h)=X_R^{(0)}+h^2X_R^{(2)}+h^3X_R^{(3)}+O(h^4),
\]

with

\[
X_R^{(2)}=-H_0^+H_2X_R^{(0)},
\qquad
X_R^{(3)}=-H_0^+H_3X_R^{(0)}.
\]

Corrected left gauge vectors satisfy the analogous left formulas. The **full** corrected gauge row and column, including their physical components, vanish through h^3.

The leading gauge obstruction is

\[
\boxed{
\mathcal G_4^{\rm eff}
=
P_L\left(H_4-H_2H_0^+H_2\right)P_R.
}
\]

The h^3 curvature-gradient sector changes the corrected vectors and the h^5 remainder, but not \(\mathcal G_4^{\rm eff}\).

A symbolic 2-physical + 1-gauge calculation verifies zero corrected-row coefficients through h^3 and exactly this Schur coefficient at h^4.

## Lemma 3: internal tent-pole elimination preserves the estimate

For a full Hessian

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

Hence the Bianchi/Hessian order survives tent-pole elimination.

## Uniform local bound

If \(\|A_h-A_0\|\le C_Ah^2\), Weyl's inequality gives

\[
\sigma_{\min}(A_h)\ge\gamma-C_Ah^2.
\]

For \(h^2\le\gamma/(2C_A)\),

\[
\|A_h^+\|\le\frac2\gamma.
\]

Thus

\[
\|X^{(2)}\|\le\frac{C_2}{\gamma},
\qquad
\|X^{(3)}\|\le\frac{C_3}{\gamma},
\]

and

\[
\boxed{\|\mathcal G_4^{\rm eff}\|\le C_4+\frac{C_2^2}{\gamma}}.
\]

A useful corrected-row estimate is

\[
\|X_L(h)^TH_h\|
\le
h^4\left(C_4+\frac{C_2^2}{\gamma}\right)
+h^5\left(C_5+\frac{2C_2C_3}{\gamma}\right)
+O(h^6).
\]

# II. Fourth-order transversality and bounded stationary response

First solve the physical middle-slice equations by the uniform physical gap. Let \(g\in\mathbb R^4\) denote the remaining gauge coordinates and \(b\) the normalized boundary data. The reduced gauge/pseudo-constraint equation has the form

\[
\Psi_h(g,b)=h^4\Psi_4(g,b)+O(h^5).
\]

Its gauge Jacobian at the background is precisely the fourth-order Schur obstruction:

\[
D_g\Psi_4=\mathcal G_4^{\rm eff}.
\]

If

\[
\sigma_{\min}(\mathcal G_4^{\rm eff})\ge\kappa>0,
\]

then dividing the gauge equation by h^4 gives

\[
\overline\Psi_h:=h^{-4}\Psi_h=\Psi_4+O(h).
\]

The ordinary implicit-function theorem applied to \(\overline\Psi_h\) yields a unique local stationary gauge branch

\[
g=g_h(b)
\]

with uniformly bounded derivative. In particular,

\[
\left\|\frac{\partial g_h}{\partial b}\right\|\le C_{gb}
\]

for sufficiently small h. Combining this with the already bounded physical response gives

\[
\boxed{\|\mathscr L_h\|\le C_L},
\]

where

\[
\mathscr L_h:=\frac{\partial L_h}{\partial l^{n+1}}
\]

is the full middle-slice stationary-response Jacobian.

In the scalar prototype

\[
\Psi_h=h^4(kg+bx)+h^5rg,
\]

one obtains

\[
\frac{\partial g}{\partial x}=-\frac{b}{k+hr}=-\frac bk+O(h),
\]

showing explicitly why the h^4 prefactor does not create an h^{-4} response when the leading pseudo-constraint Jacobian is transverse.

# III. Canonical mixed-matrix bridge

Differentiating the middle-slice stationarity equations with respect to the upper boundary gives the standard tent-move identity

\[
\boxed{H_h\mathscr L_h=-K_h},
\]

where

\[
K_h:=\frac{\partial^2\widetilde S_{(n,n+1)}}{\partial l^n\partial l^{n+1}}
\]

is the old/new mixed Lagrangian Hessian controlling the discrete Legendre transform.

For the corrected left gauge row,

\[
X_L(h)^TK_h=-X_L(h)^TH_h\mathscr L_h.
\]

The local Hessian estimate and transversality bound therefore give

\[
\boxed{
\|X_L(h)^TK_h\|
\le
C_L\left(C_4+\frac{C_2^2}{\gamma}\right)h^4+O(h^5).
}
\]

The time-reversed/lower-boundary relation gives the corresponding corrected right-gauge estimate.

Hence, in the generic curved sector satisfying fourth-order transversality, the canonical mixed matrix inherits the O(h^4) restoration rate.

If \(\mathcal G_4^{\rm eff}\) is singular, the conclusion must be reformulated: an exact/perfect residual gauge symmetry may remain, or an additional gauge fixing / higher-order Lyapunov--Schmidt analysis is required.

# IV. Fixed-volume weak assembly

Define

\[
\mathcal E_h[N,M]=h^3\sum_{v\in\Lambda_h}e_v[N,M],
\]

with bounded C^1 smearing interpolation and local O(h^4) canonical coefficient. Shape regularity gives O(h^{-3}) cells in a fixed physical three-volume V, hence

\[
\boxed{
|\mathcal E_h[N,M]|
\le
C\,V\,\|N\|_{C^1}\|M\|_{C^1}\,h^4+O(h^5).
}
\]

This is a statement in this explicitly defined weak/cell-integrated norm.

# V. FK structure-function corollary

For the unweighted FK spatial symbol

\[
A=\begin{pmatrix}4&2&2\\2&4&2\\2&2&4\end{pmatrix},
\qquad g=A^{-1},
\]

the normal-deformation commutator has

\[
\beta^i=g^{ij}(N\partial_jM-M\partial_jN)=A^{ij}(N\partial_jM-M\partial_jN).
\]

The matched adjacent x/y/z FK controls verify this structure-function direction directly. This kinematical statement is distinct from the Bianchi theorem and from the canonical transversality bridge.

# Claim boundary

Under the stated smoothness, shape-regularity, tent nondegeneracy, physical-gap, and **generic fourth-order transversality** hypotheses:

- the covariant/intermediate-slice corrected gauge Hessian residual is O(h^4);
- the stationary response is uniformly bounded;
- the mixed canonical Lagrangian Hessian inherits the O(h^4) gauge residual;
- fixed-volume weak assembly preserves O(h^4).

This is a perturbative/on-shell near-flat Regge theorem. It is not a universal exact finite-h or unrestricted off-shell Dirac-algebra theorem.

# Remaining referee work

1. Give the exact citations/equation numbers for the contracted Regge Bianchi identity and the tent-move relation \(H\mathscr L=-K\).
2. State the final discrete weak norm and smearing interpolation operator in the paper body.
3. Separate analytic assumptions from FK properties verified numerically.
4. Run an independent adversarial proof audit before closing SPE-11.
