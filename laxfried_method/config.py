import numpy as np
from dataclasses import dataclass, field

#configuration and state classes 
@dataclass
class SolverConfig:

    # grid parameters
    dx: float = 0.5          # spatial step size
    lx0: int = 4             # initial number of cells
    CFL: float = 0.8      # Courant number, must be < 1 for stability
    renorm_interval: int = 100   # trim flat regions every this many steps
    total_steps: int = 20000  # total number of time steps per laxfried call

    alpha: float = 0.4
    p: float = 0.146


    # INITIAL DATA LEFT AND RIGHT STATES 
    case_num:  int = 1  # case number for labeling
    # Left state
    rho_L: float = 4
    u_L:   float = 0.7
    v_L:   float = -1.9
    # Right state
    rho_R = 3
    u_R = 2
    v_R = 0.99999

    
    # plotting 
    t_graph: float = 1.0     # reference time used in locus/phase plane plots
    line_width_start: float = 0.25   # initial plot line width
    line_width_increment: float = 0.01  # increase per iteration (for overlay plots)

    def compute_A(self, v, p: np.ndarray):
        
        return 1/ ((1 - v)**(p))

# solver state, carries all evolving arrays through the time-stepping loop
@dataclass
class SolverState:
    """
    Holds the current numerical solution and grid metadata.
    All arrays have shape (lx,) for 1D scalar fields.
    The full conserved state is U with shape (3, lx):
        U[0] = rho
        U[1] = rho*u
        U[2] = rho*v
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

    Conserved variables initialized as:
        U[0] = rho 
        U[1] = rho*u 
        U[2] = rho*v
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
    
    """
    rho = U[0]
    u   = U[1] / U[0]   # rho * u / rho
    v   = U[2] / U[0]   # rho * v / rho
    return rho, u, v
