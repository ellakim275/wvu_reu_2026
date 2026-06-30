"""
The 3D traveling-wave vector field, transcribed from the Desmos setup.

State variables: rho (density), u, v.
A(v) = 1 / (1 - v)**p_exp, defined only for v < 1.

The left- and right-anchored vector fields share the same formula and the
same shock speed s -- only the anchor state (rho, u, v) plugged in changes.
This mirrors the 2D MATLAB code, where `dpdtdudt` was reused with the same
`eta_val` but different (w1, w2) for the left vs. right state.
"""

import numpy as np
import config as cfg


def A(v):
    """Closure function A(v) = 1/(1-v)^p_exp, valid for v < 1 (NaN otherwise)."""
    v = np.asarray(v, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = (1.0 - v) ** (-cfg.p_exp)
    return np.where(v < 1, val, np.nan)


def shock_speed():
    """
    s_SDW(rho_L, u_L, v_L, rho_R, u_R, v_R): the wave speed shared by both
    the left- and right-anchored vector fields (the 3D analogue of
    `eta_val` in the 2D code).
    """
    rho1, u1, v1 = cfg.rho_L, cfg.u_L, cfg.v_L
    rho2, u2, v2 = cfg.rho_R, cfg.u_R, cfg.v_R
    a = cfg.a

    diff_flux = rho2 * u2 - rho1 * u1
    disc = diff_flux**2 - (rho2 - rho1) * (
        rho2 * u2**2 - rho1 * u1**2 - A(v2) / rho2**a + A(v1) / rho1**a
    )

    if disc < 0:
        raise ValueError(f"shock_speed: negative discriminant ({disc}); no real solution.")

    return float((diff_flux + np.sqrt(disc)) / (rho2 - rho1))


def vector_field(x, y, z, rho_anchor, u_anchor, v_anchor, s):
    """
    (P, Q, R) = (drho/dxi, du/dxi, dv/dxi), anchored at
    (rho_anchor, u_anchor, v_anchor). Pass the left state's values for the
    left-anchored field, or the right state's for the right-anchored field.
    x, y, z may be scalars or numpy arrays (e.g. from meshgrid).
    """
    P = x * y - s * x - rho_anchor * u_anchor + s * rho_anchor
    Q = (
        rho_anchor * (u_anchor - s) * (y - u_anchor) / x
        + A(v_anchor) / (rho_anchor**cfg.a * x)
        - A(z) / x ** (cfg.a + 1)
    )
    R = rho_anchor * (u_anchor - s) * (z - v_anchor) / x
    return P, Q, R