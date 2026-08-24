# Fourth-order restoration theorem — adversarially hardened v2

## Scope

This statement concerns **length Regge calculus** on a near-flat, shape-regular four-dimensional FK refinement family. It has three pieces:

1. a covariant stationary middle-slice estimate;
2. a canonical gauge-to-gauge mixed-Hessian corollary;
3. an FK structure-function/weak-assembly corollary.

It is perturbative and on shell, or fourth-order approximately on shell. It is **not** a theorem for area Regge calculus, an exact finite-spacing diffeomorphism theorem, or an unrestricted off-shell Dirac-algebra theorem.

Use normalized lengths `lhat=l/h` and the dimensionless Regge action `Shat=h^{-2}S`. The physical mixed Hessian has no hidden power of `h` under this normalization:

`partial^2 S/(partial l partial l') = partial^2 Shat/(partial lhat partial lhat')`.

## Hypotheses

Let `T_h` be a family of four-dimensional FK length-Regge tent complexes approximating a smooth metric in a convex normal neighborhood. Assume:

1. **Smoothness.** Conservatively `g in C^8`, with uniform derivative bounds sufficient for the world-function expansion.
2. **Shape regularity/fatness.** Rescaled simplex Gram spectra remain in a fixed compact positive interval.
3. **Nondegenerate local stationary reductions.** Any tent-pole/internal block eliminated by a Schur complement has a uniform inverse bound.
4. **Small-curvature scaling.** Hinge deficits/holonomies and their normalized vertex-displacement variations satisfy
   `epsilon_h=h^2R+h^3G+O(h^4)`,
   `delta_Y epsilon_h=h^2A_Y+h^3B_Y+O(h^4)`.
5. **Middle-Hessian expansion.**
   `H_h=H_0+h^2H_2+h^3H_3+h^4H_4+O(h^5)`.
6. **Flat gauge splitting and physical gap.** `H_0` has four vertex-displacement gauge directions and complementary physical block `A_0` with `sigma_min(A_0)>=gamma>0` uniformly.
7. **Uniform coefficient bounds.** `||H_j||<=C_j`, `j=2,3,4`.
8. **Generic fourth-order transversality.** The reduced fourth-order gauge Jacobian
   `G_4^eff=P_L(H_4-H_2H_0^+H_2)P_R`
   is nonsingular, with `sigma_min(G_4^eff)>=kappa>0`.
9. **Boundary deformation source.** The canonical corollary is restricted to the four-dimensional upper-boundary geometric deformation subspace `Ran Q_g(h)`.

Exact/perfect residual-symmetry sectors with `kappa=0` require gauge fixing or higher-order Lyapunov-Schmidt analysis and are not covered by the generic transversality conclusion.

# Theorem A — stationary middle-slice gauge lifting

Let `S~_h(l^{n-1},l^n,l^{n+1})` be the two-step effective length-Regge action after legitimate local stationary reductions, and let the middle-slice variables satisfy

`E_h=partial S~_h/partial l^n=0`.

For normalized vertex-displacement fields `Y_A`, define `B_A=Y_A·E_h`. Then

`Y_B[B_A]=(nabla_{Y_B}Y_A)·E_h+Y_A^TH_hY_B`.

At exact middle-slice stationarity,

`Y_A^TH_hY_B=Y_B[B_A]`.

For a hinge triangle `(v,a,b)`, the exact area-gradient identity is

`grad_v A_h=(1/2)U_h·(a-b)`,

so the Schlaefli-reduced vertex contraction is a deficit-bivector/edge contraction of the form appearing in the contracted Regge Bianchi identities. The linear term in the deficits cancels; the breaking starts quadratically:

`B_A=O(epsilon^2)`.

Because `epsilon_h=O(h^2)` and `delta_Y epsilon_h=O(h^2)`,

`Y_A^TH_hY_B=O(h^4)`.

Hence

`P_LH_2P_R=0`,

`P_LH_3P_R=0`.

The corrected gauge vectors satisfy

`X_R^(2)=-H_0^+H_2X_R^(0)`,

`X_R^(3)=-H_0^+H_3X_R^(0)`,

and analogously on the left. Their first nonzero gauge obstruction is

`G_4^eff=P_L(H_4-H_2H_0^+H_2)P_R`.

With the uniform physical gap,

`||G_4^eff|| <= C_4 + C_2^2/gamma`,

and the corrected gauge row is `O(h^4)`.

### Approximate-stationarity extension

If the middle-slice residual is only `||E_h||=O(h^q)` and `nabla Y` is uniformly bounded, the gauge-row rate is in general

`O(h^{min(4,q)})`.

Thus fourth-order stationarity consistency (`q>=4`) is sufficient and, without additional cancellation, necessary to preserve the fourth-order rate.

# Theorem B — canonical gauge-to-gauge mixed block

Let `mathscrL_h` denote the stationary middle-slice response to upper-boundary data. Differentiating the middle-slice equations gives the standard two-step tent-move identity

`H_h mathscrL_h = -K_h`,

where `K_h` is the old/new mixed Lagrangian Hessian.

After solving the physical middle-slice equations, let `g` be the four reduced gauge coordinates. For a boundary deformation source `b_g` and a physical boundary source `b_p`, the generic reduced equation has the hierarchy

`Psi_h(g,b_g,b_p)=h^4 Psi_4(g,b_g)+h^2S_2 b_p+O(h^5)`.

Fourth-order transversality means `D_gPsi_4=G_4^eff` is invertible. Therefore:

- for deformation/gauge sources, `D_{b_g}Psi_h=O(h^4)` and `partial g/partial b_g=O(1)`;
- for physical boundary sources entering at `O(h^2)`, `partial g/partial b_p=O(h^{-2})` is allowed and expected.

Consequently only the restricted response is required to be bounded:

`||mathscrL_h Q_g||=O(1)`.

Then

`X_L(h)^T K_h Q_g(h) = -X_L(h)^TH_h mathscrL_h Q_g(h)=O(h^4)`.

Thus the canonical HDA/deformation block obeys

**`K_gg=O(h^4)`**.

No analogous fourth-order claim is made for gauge-to-physical mixing; generically one may have

**`K_gp=O(h^2)`**.

# Failure map — why the hypotheses matter

The adversarial audit gives three sharp degradation rules.

1. **Insufficient stationarity.** If `||E_h||=O(h^q)` with `q<4`, the middle gauge estimate can degrade to `O(h^q)`.
2. **Closing physical gap.** If the physical gap behaves as `gamma_h~h^alpha`, then `H_0^+` may scale as `h^{-alpha}` and the Schur backreaction can degrade from `h^4` to `h^{4-alpha}`.
3. **Lower-order boundary source.** If a source enters the reduced gauge equation at `O(h^r)`, while the gauge Jacobian is `O(h^4)`, then the gauge response scales as `O(h^{r-4})`. Hence `r=4` gives a bounded deformation response, whereas `r=2` gives the observed `O(h^{-2})` physical response.

Shape degeneration or a singular fourth-order gauge Jacobian can likewise invalidate the stated generic conclusion.

# FK48 matched validation

On the fully stationary ordinary length-Regge FK48 family at `h=0.09,0.065,0.045`:

- all 16 middle/internal Regge equations are solved to about `1e-15`;
- 12 physical Hessian singular values remain `O(1)`, with minimum gap approaching about `3.982`;
- the four lifted gauge Hessian modes have fitted exponents approximately `3.99931, 3.99933, 4.00188, 4.00651`;
- the reduced fourth-order gauge Jacobian is full rank, with `sigma/h^4` approaching about `0.1696, 0.1389, 0.1068, 0.05368`;
- the boundary-deformation restricted response is approximately constant near `0.333`, while the full response scales near `h^{-2}`;
- the refined canonical gauge-to-gauge mixed singular-value exponents are approximately `4.00028, 3.99826, 3.99923, 4.01677`;
- the gauge-to-physical mixed norm has exponent about `1.99764`.

These are matched numerical validations of the theorem hypotheses and conclusions, not replacements for the analytic Bianchi argument.

# FK structure-function corollary

For the unweighted FK spatial symbol

`A=[[4,2,2],[2,4,2],[2,2,4]]`,

`g=A^{-1}` and `g^{-1}=A`.

The neighboring normal-deformation commutator has

`beta^i=sigma g^{ij}(N partial_jM-M partial_jN)`.

The six-shared-tetrahedron calculations verify this directly in Euclidean signature (`sigma=+1`). A separate Minkowski-normal calculation verifies the expected Lorentzian kinematical sign (`sigma=-1`) with spatial commutator error `O(epsilon^2)` and normal contamination `O(epsilon)`.

The **fully stationary Lorentzian Regge canonical `K_gg=O(h^4)` calculation remains open** and is not part of this theorem.

# Fixed-volume weak assembly

For bounded `C^1` smearings and a stable interpolation, define

`E_h[N,M]=h^3 sum_{v in Lambda_h} e_v[N,M]`.

If the local deformation-sector coefficient is `O(h^4)`, shape regularity gives `O(h^{-3})` cells in fixed physical volume, hence

`|E_h[N,M]| <= C V ||N||_{C^1}||M||_{C^1} h^4 + O(h^5)`.

This conclusion is only in the stated weak/cell-integrated norm.

# Additional claim boundaries

- This theorem is for **length Regge calculus**. Area-Regge models contain non-metric modes and a different broken-diffeomorphism structure and are not covered.
- The theorem uses stationary discrete solutions rather than judging convergence by raw residual errors obtained by merely sampling a continuum solution on a lattice; this avoids the residual-error ambiguity emphasized in the Regge consistency literature.
- Quadratic lifting in deficit angle is prior art. The new package is the FK refinement realization through the stationary Schur reduction, direct canonical `K_gg` block, block separation `K_gg~h^4` versus `K_gp~h^2`, direct structure function, and weak assembly.
