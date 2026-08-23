# Perturbative fourth-order restoration theorem candidate

## Statement

Let T_h be a shape-regular family of nondegenerate four-dimensional FK/Regge tent complexes approximating a smooth metric in a convex normal neighborhood. Assume:

1. normalized edge-shape variables differ from the flat reference family by O(h^2);
2. hinge holonomy/deficit variables satisfy epsilon_h = h^2 R + O(h^3), uniformly on each local star;
3. vertex-displacement variations satisfy delta_Y epsilon_h = h^2 A_Y + O(h^3);
4. the Regge Hamilton principal map is sufficiently smooth for the Hessian expansion M_h=M_0+h^2M_2+h^4M_4+O(h^5);
5. the flat physical block has a uniform gap sigma_min(A_0) >= gamma > 0;
6. periodic assembly uses bounded C^1 smearings on a fixed physical volume.

Then the gauge-projected canonical pseudo-constraint obstruction is O(h^4). More precisely, for a corrected flat gauge vector X(h)=X^(0)+h^2 X^(2)+...,

P_L M_h X(h) = h^4 K_4^eff X^(0) + O(h^5),

where

X^(2) = -M_0^+ M_2 X^(0)

and

K_4^eff = P_L (M_4 - M_2 M_0^+ M_2) P_R.

If ||M_2||<=C_2 and ||M_4||<=C_4, then

||K_4^eff|| <= C_4 + C_2^2/gamma.

For a fixed physical spatial volume, summing cell-integrated local defects preserves the rate:

||E_h[N,M]|| <= C h^4 + O(h^5).

## Proof skeleton

### 1. Contracted Regge Bianchi identity

Let E_e=dS/dq^e, H_ef=d^2S/(dq^e dq^f), and B_A=Y_A^e E_e. The exact differential identity is

Y_B[B_A] = (nabla_{Y_B}Y_A)^e E_e + Y_A^e H_ef Y_B^f.

On shell, E=0, hence

Y_A H Y_B = Y_B[B_A].

### 2. No linear gauge-gauge curvature term

The exact nonlinear Regge Bianchi identity implies that B_A has no term linear in the small hinge holonomies/deficits:

B_A = Q_A(epsilon,epsilon) + O(epsilon^3).

With epsilon_h=O(h^2) and delta_Y epsilon_h=O(h^2),

Y_B[B_A]=O(h^4).

Therefore

Y_A H Y_B=O(h^4).

Writing M_h=M_0+h^2M_2+h^4M_4+... and projecting onto the flat left/right gauge kernels gives

P_L M_h P_R = h^2 P_L M_2 P_R + O(h^4)=O(h^4),

hence

P_L M_2 P_R=0.

### 3. Corrected gauge direction

The h^2 gauge-to-physical mixing is therefore solvable. In a physical/gauge adapted basis,

M_0 = [[A_0,0],[0,0]],
M_2 = [[B_2,C_2],[D_2,0]].

The physical h^2 equation gives

A_0 X^(2) + C_2 X^(0)=0,

so

X^(2)=-M_0^+M_2X^(0).

### 4. Fourth-order Schur obstruction

Substitution gives the leading gauge residual

K_4^eff=P_L(M_4-M_2M_0^+M_2)P_R.

A symbolic 2-physical + 1-gauge Wolfram check gives exactly the scalar block coefficient G_4-D_2 A_0^{-1} C_2 and zero h^0/h^2 gauge coefficients.

### 5. Uniform gap

If sigma_min(A_0)>=gamma and ||A_h-A_0||<=C_A h^2, Weyl's inequality yields

sigma_min(A_h)>=gamma-C_A h^2.

For h^2<=gamma/(2C_A),

||A_h^+||<=2/gamma.

Thus the Schur correction cannot introduce inverse powers of h.

### 6. Fixed-volume periodic assembly

A cell-integrated defect is O(h^3)*O(h^4)=O(h^7). A fixed spatial volume contains O(h^-3) cells, so the global defect remains O(h^4).

## Claim boundary

This is a perturbative/on-shell Regge pseudo-constraint restoration theorem under explicit smoothness, shape-regularity and physical-gap hypotheses. It is not a universal exact finite-h or unrestricted off-shell Dirac-algebra theorem.

## Remaining proof polish

- specify the normalized-edge coordinate chart and shape-regularity constants;
- state the minimum smoothness class needed for the world-function and principal-function remainders;
- give a precise local-to-global norm and smearing space;
- separate assumptions from numerically verified properties of the FK control family.
