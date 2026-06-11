import numpy as np


def systemeigen(u, rho, alpha, A): 
    lambda1 = u - ((alpha * A)/ (rho ** (alpha +1))) ** (1/2)
    lambda2 = u
    lambda3 = u + ((alpha * A)/ (rho ** (alpha +1))) ** (1/2)
    
    #maximum lambda is used for CFL condition
    return np.max([np.max(np.abs(lambda1)), 
               np.max(np.abs(lambda2)), 
               np.max(np.abs(lambda3))])