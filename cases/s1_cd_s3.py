import numpy as np
from scipy.optimize import brentq


def compute_intermediates(cfg) -> dict:
    A_L = cfg.compute_A(np.array([cfg.v_L]), np.array([cfg.p]))[0]
    A_R = cfg.compute_A(np.array([cfg.v_R]), np.array([cfg.p]))[0]
    a   = cfg.alpha

    ratio = (A_R / A_L)**(1/a)  # rho_M2 = ratio * rho_M1

    def matching(rho_M1):
        rho_M2 = ratio * rho_M1
        u_S1 = cfg.u_L - np.sqrt(A_L
               * (cfg.rho_L - rho_M1)/(cfg.rho_L * rho_M1)
               * (1/rho_M1**a - 1/cfg.rho_L**a))
        u_S3 = cfg.u_R + np.sqrt(A_R
               * (cfg.rho_R - rho_M2)/(cfg.rho_R * rho_M2)
               * (1/rho_M2**a - 1/cfg.rho_R**a))
        return u_S1 - u_S3

    rho_M1 = brentq(matching, cfg.rho_L + 1e-10, cfg.rho_L * 50, xtol=1e-12)
    rho_M2 = ratio * rho_M1
    u_M1   = cfg.u_L - np.sqrt(A_L
             * (cfg.rho_L - rho_M1)/(cfg.rho_L * rho_M1)
             * (1/rho_M1**a - 1/cfg.rho_L**a))
    u_M2   = u_M1
    
    return dict(
        rho_M1=rho_M1, u_M1=u_M1, v_M1=cfg.v_L,
        rho_M2=rho_M2, u_M2=u_M2, v_M2=cfg.v_R
    )
    
def wave_curves(cfg, intermediates: dict, rho_range: np.ndarray) -> list:
    """
    Returns list of dicts, each describing one curve to plot.
    Each dict has: rho, u, v, color, name
    """
    curves = []
    
    a     = cfg.alpha
    rho_L = cfg.rho_L
    u_L   = cfg.u_L
    rho_M2 = intermediates['rho_M2']

    A_L = float(cfg.compute_A(
        np.array([cfg.v_L]),
        np.array([cfg.p])
    )[0])
    
    A_R = float(cfg.compute_A(
        np.array([cfg.v_R]),
        np.array([cfg.p])
    )[0])

    coeff = np.sqrt(a * A_L)
    exp   = (a + 1) / 2.0

    # S1 
    rho_S1 = rho_range[rho_range > rho_L]

    inner_S1 = (A_L
                * (rho_S1 - rho_L) / (rho_L * rho_S1)
                * (1/rho_L**a - 1/rho_S1**a))

    u_S1 = u_L - np.sqrt(np.maximum(inner_S1, 0))

    curves.append(dict(
        rho=rho_S1, u=u_S1,
        v=np.full_like(rho_S1, cfg.v_L),
        color='orange', name='S1 curve'
    ))
    

    # S3
    rho_s3 = rho_range[
    (rho_range >= cfg.rho_R) & (rho_range <= rho_M2)
    ]

    inner_s3 = (A_R
                * (cfg.rho_R - rho_s3) / (cfg.rho_R * rho_s3)
                * (1/rho_s3**a - 1/cfg.rho_R**a))

    u_s3 = cfg.u_R + np.sqrt(np.maximum(inner_s3, 0))

    curves.append(dict(
        rho=rho_s3, u=u_s3,
        v=np.full_like(rho_s3, cfg.v_R),
        color='green', name='S3 curve (M2 → R)'
    ))
    
    return curves