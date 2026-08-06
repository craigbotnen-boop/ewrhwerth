# Numerical method - Release 017

The illustration uses second-order one-sided characteristic differences and three-stage SSP Runge-Kutta time stepping on five nested grids:

`h = 0.04, 0.02, 0.01, 0.005, 0.0025`.

The nominal CFL number is `0.35`. At every Runge-Kutta stage:

1. the central vertex trace is projected with the coefficient-weighted scattering matrix;
2. the external reflecting condition is imposed as `r = -ell`;
3. the coefficient is clipped to `[a_min,a_max]` as a fail-safe.

A projection audit records `max(abs(clip(a)-a))` before every clipping operation. The maximum correction across all stages and grids was `0.0`, so the fail-safe did not alter the computed evolution.

The outside-cone wave diagnostic is the maximum of `abs(r)` and `abs(ell)` over nodes satisfying

`d_Gamma(x_i,K) > sqrt(a_max) t_n + 1e-12`.

The structural diagnostic uses `abs(a-a0)` on the same mask. First-arrival times use the fixed threshold `1e-10`.
