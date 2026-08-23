# Adversarial audit of the FK/Regge O(h^4) theorem

Status: ACTIVE. This file records problems found during proof hardening rather than hiding them.

## 1. Boundary on-shell mistake — FIXED

A Hamilton principal function differentiated with respect to boundary data gives boundary momenta, not zero. Therefore the identity

Y_B[Y_A·E] = (∇_{Y_B}Y_A)·E + Y_A^T H Y_B

cannot be simplified by setting E=0 when E denotes boundary derivatives.

Repair: formulate the Bianchi theorem on the stationary middle-slice bulk variables of a two-step tent complex. There E=0 is a genuine Regge equation. Only after obtaining the corrected middle-slice Hessian estimate is the result transferred to the canonical mixed matrix.

## 2. Canonical bridge — FIXED UNDER STANDARD TENT-MOVE RELATION

For two consecutive tent moves, after tent-pole elimination and middle-slice stationarity,

H_h L_h = -K_h,

where H_h is the middle-slice Hessian, L_h is the stationary middle-slice response to upper-boundary data, and K_h is the old/new mixed Lagrangian Hessian controlling the discrete Legendre map.

The corrected left gauge vector satisfies

X_L(h)^T H_h = O(h^4).

Hence

X_L(h)^T K_h = -X_L(h)^T H_h L_h.

The canonical O(h^4) conclusion therefore requires L_h=O(1).

## 3. Bounded stationary response — REDUCED TO FOURTH-ORDER TRANSVERSALITY

Solve the physical middle-slice equations first using the uniform physical gap. The remaining gauge equations have

Psi_h(g,b)=h^4 Psi_4(g,b)+O(h^5).

The leading gauge Jacobian is the fourth-order Schur obstruction

D_g Psi_4 = G_4^eff.

If sigma_min(G_4^eff)>=kappa>0, divide by h^4 and apply the ordinary implicit-function theorem. Then dg/db=O(1) and the full stationary response L_h is uniformly bounded.

The generic tilted FK control has singular values

0.398730682, 0.119013892, 0.0937999823, 0.00135617207,

so kappa=0.00135617207 and the leading gauge block is full rank. Its condition number is about 294.0. This is numerical validation of the transversality hypothesis, not part of the analytic proof.

If G_4^eff is singular, an exact/perfect residual symmetry or a higher-order Lyapunov--Schmidt analysis is required.

## 4. Generic h^3 curvature-gradient sector — FIXED

The correct smooth expansion is

H_h=H_0+h^2 H_2+h^3 H_3+h^4 H_4+...

not the parity-restricted h^2+h^4 form. Bianchi forces both

P_L H_2 P_R=0,
P_L H_3 P_R=0.

Corrected gauge vectors acquire

X^(2)=-H_0^+ H_2 X^(0),
X^(3)=-H_0^+ H_3 X^(0),

while the leading obstruction remains

G_4^eff=P_L(H_4-H_2 H_0^+ H_2)P_R.

The h^3 sector first affects the h^5 remainder.

## 5. Tent-pole stationary reduction — FIXED

For the stationary lift through an internal block H_II,

H_red=H_BB-H_BI H_II^{-1} H_IB,
Y_I=-H_II^{-1}H_IB Y_B,

and exactly

Y_full^T H_full Y_full = Y_B^T H_red Y_B.

Thus the Bianchi/Hessian order survives exact tent-pole elimination provided H_II remains uniformly nondegenerate.

## 6. Bianchi-to-Regge-equation contraction — FIXED

For a hinge triangle h=(v,a,b), with U_h=(u∧w)/|u∧w| and d_h=a-b,

∇_v A_h = (1/2) U_h·d_h.

Therefore the Schlaefli-reduced vertex contraction is

Y_xi·E = (1/2) xi_mu sum_h epsilon_h U_h^{mu nu}(a-b)_nu.

The linearized contracted Regge Bianchi identity annihilates this term to first order in the deficits, and the exact identity contains nonlinear corrections. Hence Y_xi·E=O(epsilon^2).

## 7. Exact versus approximate on shell

Exact stationarity gives the cleanest theorem. It can be weakened as follows. If

||E_h|| <= C_E h^4

and ||∇Y|| is uniformly bounded, then

Y_A^T H Y_B = Y_B[B_A] - (∇_{Y_B}Y_A)·E_h = O(h^4).

Thus a fourth-order bulk Regge residual is sufficient. A residual of only O(h^2) would in general spoil the claimed rate.

Important numerical-label correction: the arbitrary algebraic-curvature/world-function controls are local geometric/off-shell scaling controls unless their bulk Regge residual is independently shown to satisfy the required stationarity order.

## 8. Signature boundary

The direct normal-commutator FK calculations performed so far are Euclidean-signature controls. In general

beta^a = sigma g^{ab}(N partial_b M - M partial_b N),

where n·n=sigma. The measured controls correspond to sigma=+1. A Lorentzian statement requires the sigma=-1 convention / Lorentzian simplex calculation to be checked separately.

## 9. Novelty boundary

Bahr--Dittrich already report that broken Regge gauge symmetry grows quadratically with the deficit angles. Since smooth refinement has epsilon~h^2, the heuristic h^4 consequence is not by itself novel.

The narrower contribution to defend is the combined package:

- explicit FK refinement and graph-induced structure function;
- corrected Schur coefficient and generic h^3 treatment;
- direct neighboring-tent same-boundary ordering control;
- direct normal-deformation structure-function match;
- arbitrary-smearing fixed-volume periodic assembly;
- independent generic-curvature replication;
- explicit canonical bridge and transversality formulation.

## 10. Highest-risk remaining validation

The cleanest next computational validation is an ordinary flat-simplex 4D Regge two-step tent configuration that is actually stationary in the middle-slice bulk variables (or has verified O(h^4) residual), followed by direct measurement of:

1. corrected middle-slice Hessian gauge block;
2. G_4^eff transversality;
3. stationary response L_h;
4. mixed canonical K_h residual;
5. refinement exponents.

This would test the theorem hypotheses and the canonical bridge in one matched on-shell calculation rather than combining off-shell curvature controls with separate exact/perfect benchmarks.
