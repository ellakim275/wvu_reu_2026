import numpy as np
from config import SolverState, get_primitives

def renorm(state: SolverState) -> SolverState:
    err = 1.0e-7
    lx = state.lx
    rho, u, v = get_primitives(state.U)
 
    def find_left(arr):
        N = 0
        while N < lx - 1 and abs(arr[N] - arr[0]) < err:
            N += 1
        return N
 
    def find_right(arr):
        N = lx - 1
        while N > 0 and abs(arr[N] - arr[-1]) < err:
            N -= 1
        return N
 
    lmin = min(find_left(rho), find_left(u), find_left(v)) - 2
    lmin = max(lmin, 0)
 
    lmax = max(find_right(rho), find_right(u), find_right(v)) + 2
    lmax = min(lmax, lx - 1)
 
    state.U   = state.U[:, lmin:lmax + 1]
    state.x   = state.x[lmin:lmax + 1]
    state.lx  = lmax - lmin + 1
    return state
 