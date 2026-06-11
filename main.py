import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from config import SolverConfig, initialize, get_primitives
from laxfried import lax_friedrichs

# params
NUM_INNER = 200# how many times lax_friedrichs is called (overlay curves)
PLOT_EVERY = 10       # plot a curve every this many inner iterations
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg   = SolverConfig()
state = initialize(cfg)

# Figure with 3 subplots one per primitive variable
fig, (ax_rho, ax_u, ax_v) = plt.subplots(3, 1, figsize=(8, 9))
fig.subplots_adjust(hspace=0.45)

# Main loop 
for i in range(1, NUM_INNER + 1):
    state = lax_friedrichs(state, cfg)

    if i % PLOT_EVERY == 0 or i == NUM_INNER:
        X = state.x / state.t
        rho, u, v = get_primitives(state.U)
        lw = state.line_width

        ax_rho.plot(X, rho, 'k', linewidth=lw)
        ax_u.plot(X, u,   'k', linewidth=lw)
        ax_v.plot(X, v,   'k', linewidth=lw)
        state.line_width += cfg.line_width_increment  # always increment, not just when plotting


# Labels and title — added once after the loop
ax_rho.set_ylabel(r'$\rho$', fontsize=14)
ax_rho.set_xlabel(r'$x/t$',  fontsize=14)
ax_rho.set_title(
    f"Case {cfg.case_number()}:  "
    r"$\alpha$ = " + f"{cfg.alpha},  "
    f"Steps = {state.iters},  "
    f"t = {state.t:.2f}",
    fontsize=12
)

ax_u.set_ylabel(r'$u$',   fontsize=14)
ax_u.set_xlabel(r'$x/t$', fontsize=14)

ax_v.set_ylabel(r'$v$',   fontsize=14)
ax_v.set_xlabel(
    r"$x/t$;  $L$ = "
    f"({cfg.rho_L}, {cfg.u_L}, {cfg.v_L})  "
    r"$R$ = "
    f"({cfg.rho_R}, {cfg.u_R}, {cfg.v_R})",
    fontsize=9
)

# Save
fname = (
    f"Case{cfg.case_number()}"
    f"_alpha{cfg.alpha}"
    f"_L({cfg.rho_L},{cfg.u_L},{cfg.v_L})"
    f"_R({cfg.rho_R},{cfg.u_R},{cfg.v_R})"
    ".png"
)
out_path = os.path.join(OUTPUT_DIR, fname)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")