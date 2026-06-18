import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from config import SolverConfig, initialize, get_primitives
from laxfried import lax_friedrichs
from phase_plot_3d import phase_plot_3d


def build_filename(cfg: SolverConfig) -> str:
    """
    Builds the descriptive base filename (no extension) used for both
    the x/t plot and the 3d phase plot, so a single run's outputs are
    identifiable by their right state.
    """
    return (
        f"Case{cfg.case_num}"
        f"_alpha{cfg.alpha}"
        f"_L({cfg.rho_L},{cfg.u_L},{cfg.v_L})"
        f"_R({cfg.rho_R},{cfg.u_R},{cfg.v_R})"
    )


def run_simulation(cfg: SolverConfig, xt_dir: str, phase_dir: str,
                    plot_every: int = 1000, curve_mode: str = 'surface') -> dict:
    """
    Runs the Lax-Friedrichs solver for cfg.total_steps steps, saving:
      - an x/t overlay plot (rho, u, v) into xt_dir
      - a 3D phase portrait html into phase_dir

    curve_mode controls how the R1/S1/R3/S3 wave curves are drawn in the
    3D phase portrait: 'surface' (translucent extruded ribbons, default),
    'line' (flat dashed lines through L), 'both', or 'none'.

    Returns a dict with the paths written and the final state, in case
    a caller wants to inspect/aggregate results.
    """
    os.makedirs(xt_dir, exist_ok=True)
    os.makedirs(phase_dir, exist_ok=True)

    state = initialize(cfg)

    NUM_INNER = cfg.total_steps  # how many times lax_friedrichs is called (overlay curves)
    PLOT_EVERY = plot_every       # plot a curve every this many inner iterations

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

    # Labels and title, added once after the loop
    ax_rho.set_ylabel(r'$\rho$', fontsize=14)
    ax_rho.set_xlabel(r'$x/t$',  fontsize=14)
    ax_rho.set_title(
        f"Case {cfg.case_num}:  "
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

    # Save x/t plot
    base_name = build_filename(cfg)
    xt_fname = base_name + ".png"
    xt_path = os.path.join(xt_dir, xt_fname)
    fig.savefig(xt_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {xt_path}")

    # Save 3d phase plot
    phase_fname = base_name + "_3d.html"
    phase_path = os.path.join(phase_dir, phase_fname)
    phase_plot_3d(
        states=[state],
        cfg=cfg,
        save_html=phase_path,
        curve_mode=curve_mode,
    )

    return dict(xt_path=xt_path, phase_path=phase_path, state=state)


if __name__ == "__main__":
    cfg = SolverConfig()
    OUTPUT_DIR = "output"
    run_simulation(cfg, xt_dir=OUTPUT_DIR, phase_dir=OUTPUT_DIR, plot_every=1000)