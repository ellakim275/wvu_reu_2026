import numpy as np

# computes the eigenvalues of the current system and returns the maximum because we use it in the CFL condition 
def systemeigen(u, rho, alpha, A): 
    lambda1 = u - ((alpha * A)/ (rho ** (alpha +1))) ** (1/2)
    lambda2 = u
    lambda3 = u + ((alpha * A)/ (rho ** (alpha +1))) ** (1/2)
    
    return np.max([np.max(np.abs(lambda1)), 
               np.max(np.abs(lambda2)), 
               np.max(np.abs(lambda3))])