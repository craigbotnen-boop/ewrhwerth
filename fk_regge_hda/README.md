# FK Regge HDA — O(h^4) restoration

This directory freezes the computational/theoretical package for a focused manuscript on fourth-order restoration of the hypersurface-deformation algebra in refined 4D FK/Regge calculus.

## Frozen claim

Under smooth-curvature, shape-regular/nondegenerate-tent, bounded-smearing assumptions, the generic-curvature Regge pseudo-constraint/HDA anomaly is computationally observed and perturbatively explained to scale as O(h^4). Exact finite-spacing diffeomorphism symmetry at generic curvature is **not** claimed.

## Core mechanism

- Exact Regge Bianchi identity cancels the contribution linear in small hinge holonomies/deficits.
- With deficits O(h^2), the leading gauge-breaking contribution is quadratic, O(h^4).
- In a 10 physical + 4 gauge decomposition, the corrected gauge vector is X_A^(2) = -M_0^+ M_2 X_A^(0), with solvability supplied by the vanishing linear gauge-gauge block.
- The fourth-order obstruction is K_4 = P_L (M_4 - M_2 M_0^+ M_2) P_R.

## Headline computational checks

- Schur-corrected local pseudo-constraint block: exponent ~3.99975.
- Direct neighboring-tent same-boundary ordering defect: ~h^6; after dividing by the two O(h) tent displacements, inferred canonical rate ~h^4.
- Independent full-rank algebraic-curvature replication: action exponent 6.0030366, bracket exponent 4.0030366.
- Direct normal-deformation commutator on the six shared FK tetrahedra reproduces the graph-induced structure function with g=A^-1 and g^-1=A.
- Periodic fixed-volume arbitrary-smearing assembly preserves O(h^4).

## Claim boundary

Do not upgrade this package to a universal nonlinear-GR theorem or to an exact generic-curvature finite-spacing Dirac algebra. A literal one-shot off-shell Poisson evaluation for the full adjacent 48-simplex Hamilton-principal functions remains outside the frozen claim.

See `evidence.csv` and `manuscript_outline.md` for the current audit table and writing plan.
