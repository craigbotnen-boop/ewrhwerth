# Perturbative fourth-order restoration theorem candidate

## Dimensionless setup

Work with normalized edge variables \(\hat l=l/h\) and the dimensionless reduced Hamilton principal function

\[
\widehat S_h(\hat l_B)=h^{-2}S_h(h\hat l_B,z_h(h\hat l_B)),
\]

where internal tent variables \(z_h\) are eliminated by their exact stationarity equations. All Hessians below are taken with respect to the normalized boundary variables \(\hat l_B\).

## Hypotheses

Let \(T_h\) be a family of four-dimensional FK/Regge tent complexes approximating a smooth metric \(g\) on a convex normal neighborhood. Assume:

1. **Smooth background.** \(g\in C^8\) on the neighborhood, with uniform bounds on the derivatives needed for the world-function expansion.
2. **Shape regularity / fatness.** After rescaling each simplex by \(h^{-1}\), all edge-Gram eigenvalues lie in a fixed interval \([\lambda_*,\lambda^*]\subset(0,\infty)\). Equivalently, simplex aspect ratios and normalized volumes are uniformly controlled.
3. **Nondegenerate tent reduction.** The internal stationary point is locally unique and the internal Hessian block obeys \(\sigma_{\min}(H_{II})\ge \tau>0\) uniformly.
4. **Curvature scaling.** Hinge holonomy/deficit variables satisfy, uniformly on each local star,
   \[
   \epsilon_h=h^2R+h^3G+O(h^4),
   \qquad
   \delta_Y\epsilon_h=h^2A_Y+h^3B_Y+O(h^4).
   \]
5. **Reduced principal-map expansion.** The dimensionless reduced Hessian has a uniform expansion
   \[
   M_h=M_0+h^2M_2+h^3M_3+h^4M_4+R_5(h),
   \qquad \|R_5(h)\|\le C_5h^5.
   \]
6. **Flat gauge splitting.** The flat operator \(M_0\) has a fixed four-dimensional left/right gauge kernel, with corresponding projections \(P_L,P_R\), and the complementary physical block \(A_0\) satisfies
   \[
   \sigma_{\min}(A_0)\ge \gamma>0.
   \]
7. **Bounded coefficient norms.** \(\|M_j\|\le C_j\) for \(j=2,3,4\), uniformly for sufficiently small \(h\).
8. **Periodic/global assembly.** Smearings \(N,M\) are uniformly bounded in \(C^1\), interpolation is stable on the shape-regular family, and the physical spatial volume is fixed.

The \(C^8\) assumption is deliberately conservative; lowering the regularity threshold is not needed for the first theorem.

## Theorem: local gauge-block restoration

Let \(E=d\widehat S_h/d\hat l\), \(H=d^2\widehat S_h/d\hat l^2\), and let \(Y_A\) be a normalized vertex-displacement field. Define the contracted Regge Bianchi quantity

\[
B_A=Y_A\cdot E.
\]

The exact differential identity is

\[
Y_B[B_A]=(\nabla_{Y_B}Y_A)\cdot E+Y_A^THY_B.
\]

On shell, \(E=0\), hence

\[
Y_A^THY_B=Y_B[B_A].
\]

For a hinge triangle \(h=(v,a,b)\), writing \(u=a-v\), \(w=b-v\), \(d_h=a-b\), and \(U_h=(u\wedge w)/|u\wedge w|\), one has

\[
\nabla_vA_h=\frac12U_h\cdot d_h.
\]

Using the Schlaefli-reduced Regge equations,

\[
Y_\xi\cdot E
=
\frac12\xi_\mu\sum_{h\supset v}\epsilon_hU_h^{\mu\nu}(a-b)_\nu.
\]

The linearized contracted Regge Bianchi identity annihilates the term linear in the deficits, while the exact nonlinear identity contains quadratic and higher curvature terms. Therefore

\[
B_A=O(\epsilon^2).
\]

Together with \(\epsilon_h=O(h^2)\) and \(\delta_Y\epsilon_h=O(h^2)\), this gives

\[
Y_A^THY_B=O(h^4).
\]

Consequently

\[
P_LM_hP_R=O(h^4),
\]

and coefficient comparison yields

\[
\boxed{P_LM_2P_R=0},
\qquad
\boxed{P_LM_3P_R=0}.
\]

## Corrected gauge vector

Choose a corrected right gauge vector

\[
X(h)=X^{(0)}+h^2X^{(2)}+h^3X^{(3)}+O(h^4),
\qquad X^{(0)}\in\operatorname{Ran}P_R.
\]

Solving the physical equations order by order gives

\[
\boxed{X^{(2)}=-M_0^+M_2X^{(0)}},
\]

\[
\boxed{X^{(3)}=-M_0^+M_3X^{(0)}}.
\]

The first nonzero gauge residual occurs at fourth order:

\[
P_LM_hX(h)
=
h^4K_4^{\mathrm{eff}}X^{(0)}+O(h^5),
\]

with

\[
\boxed{
K_4^{\mathrm{eff}}
=
P_L\left(M_4-M_2M_0^+M_2\right)P_R.
}
\]

The \(h^3\) curvature-gradient mixing affects \(X^{(3)}\) and the \(h^5\) remainder, but not the leading \(K_4^{\mathrm{eff}}\) obstruction.

## Stationary reduction lemma

If the unreduced Hessian is block-partitioned into boundary/internal variables,

\[
H_{\mathrm{full}}=
\begin{pmatrix}
H_{BB}&H_{BI}\\
H_{IB}&H_{II}
\end{pmatrix},
\]

then the reduced Hessian is

\[
H_{\mathrm{red}}=H_{BB}-H_{BI}H_{II}^{-1}H_{IB}.
\]

For a boundary displacement \(Y_B\), the stationary lift satisfies

\[
Y_I=-H_{II}^{-1}H_{IB}Y_B,
\]

and exactly

\[
Y_{\mathrm{full}}^TH_{\mathrm{full}}Y_{\mathrm{full}}
=Y_B^TH_{\mathrm{red}}Y_B.
\]

Hence the Bianchi/Hessian estimate descends unchanged through exact tent-variable elimination.

## Uniform norm bound

If \(\|A_h-A_0\|\le C_Ah^2\), Weyl's inequality gives

\[
\sigma_{\min}(A_h)\ge\gamma-C_Ah^2.
\]

For \(h^2\le\gamma/(2C_A)\),

\[
\|A_h^+\|\le\frac{2}{\gamma}.
\]

At the flat-order level,

\[
\|X^{(2)}\|\le\frac{C_2}{\gamma},
\qquad
\|X^{(3)}\|\le\frac{C_3}{\gamma},
\]

and

\[
\boxed{
\|K_4^{\mathrm{eff}}\|
\le C_4+\frac{C_2^2}{\gamma}.
}
\]

A convenient explicit local residual estimate is

\[
\|P_LM_hX(h)\|
\le
h^4\left(C_4+\frac{C_2^2}{\gamma}\right)
+h^5\left(C_5+\frac{2C_2C_3}{\gamma}\right)
+O(h^6).
\]

## Fixed-volume corollary

If the local defect is interpreted as a cell density and integrated with cell volume \(O(h^3)\), then each cell contributes \(O(h^7)\). Shape regularity implies \(O(h^{-3})\) cells in a fixed physical three-volume. Therefore, for bounded \(C^1\) smearings,

\[
\boxed{
|\mathcal E_h[N,M]|
\le
C\,V\,\|N\|_{C^1}\|M\|_{C^1}\,h^4+O(h^5).
}
\]

## HDA structure-function corollary for the FK family

For the unweighted FK spatial symbol

\[
A=
\begin{pmatrix}
4&2&2\\
2&4&2\\
2&2&4
\end{pmatrix},
\qquad g=A^{-1},
\]

the geometric normal-deformation commutator has the continuum tensor type

\[
\beta^i=g^{ij}(N\partial_jM-M\partial_jN)=A^{ij}(N\partial_jM-M\partial_jN).
\]

The computational FK controls verify this structure function directly for adjacent x/y/z tents. The theorem above controls the pseudo-constraint obstruction; it does not by itself constitute an unrestricted off-shell proof of the full Dirac algebra.

## Claim boundary

This is a perturbative/on-shell Regge pseudo-constraint restoration theorem under explicit smoothness, shape-regularity, stationary-reduction, and physical-gap hypotheses. It is not a universal exact finite-h or unrestricted off-shell Dirac-algebra theorem.

## Remaining referee polish

- cite the precise contracted Regge Bianchi identity and fat-triangulation convergence results;
- state the chosen discrete global norm in the main text and alternatives in an appendix;
- separate hypotheses that are assumed analytically from properties verified numerically for the FK control family;
- perform an independent adversarial proof audit before marking SPE-11 complete.
