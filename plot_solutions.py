"""

Plots rho, u, and v against the self-similar variable x/t.

Two modes:
  - overlay mode: all iterations plotted on the same axes with increasing
                  line width, so convergence to the self-similar solution
                  is visible 
  - final mode:   only the last iteration plotted cleanly

"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from config import SolverConfig, SolverState, get_primitives


_fig = None
_axes = None   # tuple of (ax_rho, ax_u, ax_v)


def _setup_figure(cfg: SolverConfig):
    """
    Create the figure and three subplots on first call.
    Subsequent calls reuse the same figure, matching MATLAB's figure(1) behavior.
    """
    global _fig, _axes

    if _fig is None:
        _fig, axs = plt.subplots(3, 1, figsize=(8, 9))
        _fig.subplots_adjust(hspace=0.45)
        _axes = (axs[0], axs[1], axs[2])

    return _fig, _axes


def solution_plot(state: SolverState, cfg: SolverConfig, mode: str = "overlay"):
    """
    Plot rho, u, v against x/t.

    Parameters
    ----------
    state : SolverState
        Current solver state. Must have t > 0 (can't plot x/t at t=0).
    cfg : SolverConfig
        Configuration, used for labels and initial conditions.
    mode : str
        "overlay"  — add this iteration to the existing axes (hold on equivalent)
        "final"    — clear axes and plot only this iteration cleanly

    The line width increases each call via state.line_width, matching:
        lU = lU + 0.05  (MATLAB)
    """
    if state.t == 0:
        return   # can't form x/t at t=0

    fig, (ax_rho, ax_u, ax_v) = _setup_figure(cfg)

    # self-similar variable
    X = state.x / state.t

    # recover primitive variables
    rho, u, v = get_primitives(state.U)

    lw = state.line_width

    if mode == "final":
        ax_rho.cla()
        ax_u.cla()
        ax_v.cla()

    # --- top subplot: rho ---
    ax_rho.plot(X, rho, 'k', linewidth=lw)
    ax_rho.set_ylabel(r'$\rho$', fontsize=14)
    ax_rho.set_xlabel(r'$x/t$', fontsize=14)
    ax_rho.set_title(
        f"Case ?: "
        r"$\alpha$ = " + f"{cfg.alpha}, "
        f"Steps = {state.iters}, "
        f"t = {state.t:.2f}",
        fontsize=13
    )

    # --- middle subplot: u ---
    ax_u.plot(X, u, 'k', linewidth=lw)
    ax_u.set_ylabel(r'$u$', fontsize=14)
    ax_u.set_xlabel(r'$x/t$', fontsize=14)

    # --- bottom subplot: v ---
    ax_v.plot(X, v, 'k', linewidth=lw)
    ax_v.set_ylabel(r'$v$', fontsize=14)

    # bottom xlabel carries the left/right state data, matching MATLAB:
    # xlabel(['\itx/t; \rm Data \itL \rm= (...), \itR \rm= (...)'])
    ax_v.set_xlabel(
        r"$x/t$;  Data $L$ = "
        f"({cfg.rho_L}, {cfg.u_L}, {cfg.v_L}),  "
        r"$R$ = "
        f"({cfg.rho_R}, {cfg.u_R}, {cfg.v_R})",
        fontsize=9
    )

    # increment line width for next iteration
    state.line_width += cfg.line_width_increment

    fig.canvas.draw_idle()
    plt.pause(0.001)   # allows figure to update mid-run if interactive

    return fig


