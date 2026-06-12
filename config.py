import numpy as np
from dataclasses import dataclass, field

#configuration and state classes 
@dataclass
class SolverConfig:

    # grid parameters
    dx: float = 1.0          # spatial step size
    lx0: int = 4             # initial number of cells
    CFL: float = 0.1        # Courant number, must be < 1 for stability
    renorm_interval: int = 100   # trim flat regions every this many steps
    total_steps: int = 1000  # total number of time steps per laxfried call

    alpha: float = 0.5   
    p: float = 0.36


    # INITIAL DATA LEFT AND RIGHT STATES 
    # Left state (x < 0)
    rho_L: float = 3.0
    u_L:   float = 3.0
    v_L:   float = 0

    # Right state (x > 0)
    rho_R: float = 10.0
    u_R:   float = 10.0
    v_R:   float = 0.5

    # plotting 
    t_graph: float = 1.0     # reference time used in locus/phase plane plots
    line_width_start: float = 0.25   # initial plot line width
    line_width_increment: float = 0.05  # increase per iteration (for overlay plots)
    A_const: float = 1.0  # constant A for compute_A, can be modified for different cases


    def compute_A(self, v, p: np.ndarray) -> np.ndarray:
        
        return 1/ ((1 - v)**(p))
        #return self.A_const * np.ones_like(rho)  # A is constant for all rho, can be modified as needed


    def case_number(self) -> int:
        """
        placeholder for now until we label all the cases
        """
        if self.alpha > 0:
            if self.u_L >= 0:
                return 1   # expand as needed
            else:
                return 2
        elif self.alpha == 0:
            return 3
        else:
            return 4



# solver state, carries all evolving arrays through the time-stepping loop

@dataclass
class SolverState:
    """
    Holds the current numerical solution and grid metadata.
    All arrays have shape (lx,) for 1D scalar fields.
    The full conserved state is U with shape (3, lx):
        U[0] = ρ
        U[1] = ρu
        U[2] = ρv
    """

    # Conserved variable array, shape (3, lx)
    U: np.ndarray = field(default_factory=lambda: np.zeros((3, 4)))

    # Spatial grid, shape (lx,)
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Current number of active cells
    lx: int = 4

    # Current simulation time
    t: float = 0.0

    # Total number of time steps taken
    iters: int = 0

    # Counter for renormalization trigger
    norm_counter: int = 0

    # Whether the solver has been initialized
    started: bool = False

    # Current line width for plotting (increases each iteration)
    line_width: float = 0.25


# Initialization function

def initialize(cfg: SolverConfig) -> SolverState:
    """
    Builds the initial SolverState from a SolverConfig.
    Equivalent to the initvars + setup block in autogen.m.

    Conserved variables initialized as:
        U[0] = ρ  — left state everywhere, right state from cell 3 onward
        U[1] = ρu — same pattern
        U[2] = ρv — same pattern

    This mirrors the MATLAB initialization:
        p = pL * ones(1, lx0)
        for i = 3:lx0
            p(i) = p(i) + (pR - pL)
        end
    """
    lx = cfg.lx0

    # Build primitive variable arrays
    rho = cfg.rho_L * np.ones(lx)
    u   = cfg.u_L   * np.ones(lx)
    v   = cfg.v_L   * np.ones(lx)

    # Apply right state from index 2 onward (0-indexed)
    rho[2:] += (cfg.rho_R - cfg.rho_L)
    u[2:]   += (cfg.u_R   - cfg.u_L)
    v[2:]   += (cfg.v_R   - cfg.v_L)

    # Build conserved variable array
    U = np.zeros((3, lx))
    U[0] = rho
    U[1] = rho * u
    U[2] = rho * v

    # Build spatial grid
    indices = np.arange(1, lx + 1)
    centerx = (lx + 1) / 2.0
    x = 2 * cfg.dx * (indices - centerx)

    return SolverState(
        U=U,
        x=x,
        lx=lx,
        t=0.0,
        iters=0,
        norm_counter=1,
        started=False,
        line_width=cfg.line_width_start,
    )
    
def __post_init__(self):
        """
        Runs automatically after __init__.
        Validates that all v values are strictly less than 1.
        """
        if self.v_L >= 1.0:
            raise ValueError(
                "v_L must be strictly less than 1.0"
            )
        if self.v_R >= 1.0:
            raise ValueError(
                "v_R must be strictly less than 1.0"
            )


def get_primitives(U: np.ndarray):
    """
    Recover primitive variables from conserved variables.
    Returns (rho, u, v) as 1D arrays of length lx.

    Equivalent to:
        u = q ./ p   (MATLAB)
    but now for all three primitives.
    """
    rho = U[0]
    u   = U[1] / U[0]   # rho * u / rho
    v   = U[2] / U[0]   # rho * v / rho
    return rho, u, v
