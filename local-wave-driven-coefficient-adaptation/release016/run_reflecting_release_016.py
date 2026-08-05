"""Run Release 016 with the theorem's reflecting external boundary condition.

This wrapper patches the archived Release 015 solver so that p=0 is imposed as
r=-ell at each external endpoint, including the differentiated stage relation.
"""
import run_higher_order_solver as base


def project_boundaries(r, ell, a):
    r = r.copy()
    ell = ell.copy()
    c = base.np.sqrt(a)
    ell[:, 0] = base.scattering(c[:, 0], r[:, 0])
    r[:, -1] = -ell[:, -1]
    return r, ell, base.np.clip(a, base.A_MIN, base.A_MAX)


_original_rhs = base.rhs


def reflecting_rhs(r, ell, a, h):
    dr, dell, da = _original_rhs(r, ell, a, h)
    dr[:, -1] = -dell[:, -1]
    return dr, dell, da


base.project_boundaries = project_boundaries
base.rhs = reflecting_rhs

if __name__ == "__main__":
    base.main()
