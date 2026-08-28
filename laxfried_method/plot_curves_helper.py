import numpy as np

# plots all four candidate wave curve branches (R1, S1, R3, S3) anchored at the LEFT state only

def wave_curves(cfg, rho_range):
    """
    Returns all four wave-curve branches (R1, S1, R3, S3), all anchored
    at the left state (rho_L, u_L, v_L).

    Each dict has: rho, u, v, color, name
    """
    curves = []

    a = cfg.alpha
    rho_L = cfg.rho_L
    u_L   = cfg.u_L
    v_L   = cfg.v_L

    A_L = float(cfg.compute_A(np.array([v_L]), np.array([cfg.p]))[0])

    exp = (a + 1) / 2.0

    # R1: rarefaction branch through L, valid for rho < rho_L 
    rho_r1 = rho_range[(rho_range > 0) & (rho_range < rho_L)]
    u_r1 = (u_L
            + np.sqrt(a * A_L) * (2.0 / (a + 1))
            * (1.0 / rho_r1**exp - 1.0 / rho_L**exp))

    curves.append(dict(
        rho=rho_r1, u=u_r1,
        v=np.full_like(rho_r1, v_L),
        color='blue', name='R1 curve (through L)'
    ))

    # S1: shock branch through L, valid for rho > rho_L 
    rho_s1 = rho_range[rho_range > rho_L]
    inner_s1 = (A_L
                * (rho_s1 - rho_L) / (rho_L * rho_s1)
                * (1.0 / rho_L**a - 1.0 / rho_s1**a))
    u_s1 = u_L - np.sqrt(np.maximum(inner_s1, 0))

    curves.append(dict(
        rho=rho_s1, u=u_s1,
        v=np.full_like(rho_s1, v_L),
        color='orange', name='S1 curve (through L)'
    ))

    # R3: rarefaction branch, anchored at L (rho_L, u_L), valid for rho > rho_L
    rho_r3 = rho_range[rho_range > rho_L]
    u_r3 = (u_L
            + np.sqrt(a * A_L) * (2.0 / (a + 1))
            * (1.0 / rho_L**exp - 1.0 / rho_r3**exp))

    curves.append(dict(
        rho=rho_r3, u=u_r3,
        v=np.full_like(rho_r3, v_L),
        color='red', name='R3 curve (through L)'
    ))

    # S3: shock branch, anchored at L (rho_L, u_L), valid for rho < rho_L 
    rho_s3 = rho_range[(rho_range > 0) & (rho_range < rho_L)]
    inner_s3 = (A_L
                * (rho_L - rho_s3) / (rho_L * rho_s3)
                * (1.0 / rho_s3**a - 1.0 / rho_L**a))
    u_s3 = u_L - np.sqrt(np.maximum(inner_s3, 0))

    curves.append(dict(
        rho=rho_s3, u=u_s3,
        v=np.full_like(rho_s3, v_L),
        color='green', name='S3 curve (through L)'
    ))

    return curves