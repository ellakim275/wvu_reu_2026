# cases/r1_cd_r3.py

from math import exp

import numpy as np
from scipy.optimize import brentq

def compute_intermediates(cfg) -> dict:
    """
    Computes M1 and M2 for R1 + CD + R3.
    Returns dict with rho_M1, u_M1, v_M1, rho_M2, u_M2, v_M2.
    """
    A_L = cfg.compute_A(np.array([cfg.v_L]), np.array([cfg.p]))[0]
    A_R = cfg.compute_A(np.array([cfg.v_R]), np.array([cfg.p]))[0]
    exp = (cfg.alpha + 1) / 2.0   # shorthand for (a+1)/2

    rho_M1_exp = np.sqrt(cfg.alpha * A_L) * ((1+ (A_L/A_R))**(1/(2*cfg.alpha))) / (exp * (cfg.u_R - cfg.u_L) + (np.sqrt(cfg.alpha * A_R)/cfg.rho_R**exp)+(np.sqrt(cfg.alpha * A_L)/cfg.rho_L**exp))
    rho_M1 = rho_M1_exp ** (1.0 / exp)   
    
    # --- u_M1 from R1 curve, page 79 ---
    u_M1 = (np.sqrt(cfg.alpha * A_L) 
        * (2.0 / (cfg.alpha + 1)) 
        * (1.0 / rho_M1**exp - 1.0 / cfg.rho_L**exp)
        + cfg.u_L)  

    # --- rho_M2 from CD condition, page 79 ---
    rho_M2 = (A_R * rho_M1**cfg.alpha / A_L) ** (1.0 / cfg.alpha)

    # --- u_M2 = u_M1, continuous across CD ---
    u_M2 = u_M1
    
    return dict(
        rho_M1=rho_M1, u_M1=u_M1, v_M1=cfg.v_L,
        rho_M2=rho_M2, u_M2=u_M2, v_M2=cfg.v_R
    )


    
def wave_curves(cfg, intermediates: dict, rho_range: np.ndarray) -> list:
    """
    Returns list of dicts, each describing one curve to plot.
    Each dict has: rho, u, v, color, name
    """
    #MEOW ALOT OF THIS LABELING IS WRONG RIGHT NOW I HAVE TO FIGURE IT OUT 
    curves = []
    
    a     = cfg.alpha
    rho_L = cfg.rho_L
    u_L   = cfg.u_L
    rho_M2 = intermediates['rho_M2']
    u_M2   = intermediates['u_M2']

    A_L = float(cfg.compute_A(
        np.array([cfg.v_L]),
        np.array([cfg.p])
    )[0])

    coeff = np.sqrt(a * A_L)
    exp   = (a + 1) / 2.0

    # R1 rarefaction — rho < rho_L, u increases as rho decreases
    rho_r1   = rho_range[rho_range < rho_L]
    u_r1  = u_L + coeff * (2.0/(a+1)) * (1.0/rho_L**exp - 1.0/rho_r1**exp)

    curves.append(dict(
        rho=rho_r1, u=u_r1,
        v=np.full_like(rho_r1, cfg.v_L),
        color='blue', name='R1 curve'
    ))
    
    A_R = float(cfg.compute_A(
        np.array([cfg.v_R]),
        np.array([cfg.p])
    )[0])

    # only plot between M2 and R
    rho_r3 = rho_range[
        (rho_range >= rho_M2) & (rho_range <= cfg.rho_R)
    ]

    u_r3 = (cfg.u_R
             + np.sqrt(a * A_R)
             * (2.0 / (a + 1))
             * (1.0 / cfg.rho_R**exp - 1.0 / rho_r3**exp))
    
    curves.append(dict(
        rho=rho_r3, u=u_r3,
        v=np.full_like(rho_r3, cfg.v_R),
        color='green', name='R3 curve through M2'
    ))
    
    return curves