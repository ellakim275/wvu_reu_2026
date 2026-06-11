import numpy as np 
from config import get_primitives

# flux function for the system, used in the lax-friedrichs update step
def flux(U, cfg): 
    rho, u, v, = get_primitives(U)
    A = cfg.compute_A(v, rho)
    f1 = rho * u
    f2 = rho * (u**2) - (A / (rho ** cfg.alpha))
    f3 = rho * u * v
    return np.array([f1, f2, f3])