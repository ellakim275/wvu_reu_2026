"""
Configuration for the 3D traveling-wave vector field.

Mirrors the role of the original 2D MATLAB `initvars.m`: this is the one
file you should need to edit to change states, parameters, or plot ranges.

Plot ranges: set RHO_MIN/RHO_MAX, U_MIN/U_MAX, V_MIN/V_MAX directly. These
three ranges are used everywhere -- the per-state 2D plots, the combined
2D plots (both states overlaid), and the 3D Plotly cone plots all draw
from the same explicit domain, so changing a range here changes it
everywhere consistently.
"""

# --- model parameters ---------------------------------------------------
a = 0.66        # exponent on rho in the flux / closure terms
p_exp = 0.146   # exponent inside A(v) = 1 / (1 - v)^p_exp

rho_L = 4.0
u_L = 0.7
v_L = -1.9

rho_R = 4.2
u_R = 0.15
v_R = 0.99999999999

RHO_MIN, RHO_MAX = 0.01, 6.0
U_MIN, U_MAX = -4.0, 4.0
V_MIN, V_MAX = -10.0, 0.99

RANGES = {
    "rho": (RHO_MIN, RHO_MAX),
    "u": (U_MIN, U_MAX),
    "v": (V_MIN, V_MAX),
}

# --- 2D quiver plot settings ---------------------------------------------
GRID_N = 25                 # grid points per axis
ARROW_LEN_FRACTION = 0.8    # arrow length, as a fraction of local grid spacing

# --- 3D Plotly field settings ----------------------------------------------
GRID_N_3D = 8                # grid points per axis in the 3D cone plot
CONE_SIZE = 0.7               # cone size factor (sizemode="scaled": relative
                               # to grid spacing, not raw vector magnitude)


def state_dir(state_name):
    """Output folder name for a given state, e.g. 'Left' -> 'left_state'."""
    return f"{state_name.lower()}_state"


if RHO_MIN <= 0:
    print(f"warning: RHO_MIN={RHO_MIN} <= 0; the field is singular at rho=0")
if V_MAX >= 1:
    print(f"warning: V_MAX={V_MAX} >= 1; A(v) is singular at v=1")