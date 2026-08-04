# Numerical method - Release 013

The code evolves

\[
r=p+s/c,\qquad \ell=p-s/c,\qquad c=\sqrt a,
\]

through

\[
r_t-cr_x=(r-\ell)\left(\frac{F}{4a}+\frac{c_x}{2}\right),
\]

\[
\ell_t+c\ell_x=(r-\ell)\left(\frac{c_x}{2}-\frac{F}{4a}\right),
\qquad a_t=F.
\]

The left-moving variable uses a forward second-order upwind stencil and the right-moving variable uses a backward second-order upwind stencil. Time stepping is SSP-RK3. At every stage, the central vertex trace is projected by the coefficient-weighted scattering law and the external incoming trace is set to zero. The nominal CFL number is 0.35.

Successive-grid errors are relative discrete Euclidean differences after restricting the fine grid to coarse nodes. Outside-cone leakage is the maximum characteristic amplitude on nodes satisfying

\[
d_\Gamma(x_i,K)>\sqrt{a_+}\,t_n+10^{-12}.
\]

First-arrival diagnostics use the fixed amplitude threshold \(10^{-10}\).
