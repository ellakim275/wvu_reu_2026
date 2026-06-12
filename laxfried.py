from config import SolverConfig, SolverState, get_primitives
from systemeigen import systemeigen
from flux import flux 
from renorm import renorm
import numpy as np 

# lax-friedrichs method (finite volume method) to update the solution at each time step 

def lax_friedrichs(state: SolverState, cfg: SolverConfig) -> SolverState:
    if not state.started:
        state.started = True
   
    rho, u, v = get_primitives(state.U)
    lambda_max = systemeigen(u, rho, cfg.alpha, cfg.compute_A(v, rho))
    dt = cfg.CFL / lambda_max
        
    state.t += dt 
    state.iters += 1
    
    U_left = np.concatenate([state.U[:, :1], state.U[:, :-1]], axis=1)
    F = flux(state.U, cfg)
    F_left = np.concatenate([F[:, :1], F[:, :-1]], axis=1)
    
    ##maybe change this update method but this is what was in the matlab code 
    state.U = 0.5 * (U_left + state.U) + (0.5 * dt * (F_left - F)) #check the minus sign here 
    
    # extend U by repeating rightmost column
    state.U = np.concatenate([state.U, state.U[:, -1:]], axis=1)

    # shift x left and append new rightmost point
    state.x = state.x - cfg.dx
    new_point = state.x[-1] + 2 * cfg.dx
    state.x = np.append(state.x, new_point)

    state.lx += 1
    state.norm_counter += 1
    if state.norm_counter == cfg.renorm_interval:
        state = renorm(state)
        state.norm_counter = 0

    return state