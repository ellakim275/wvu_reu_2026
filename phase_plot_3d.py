"""
phase_plot_3d.py
================
3D phase portrait of the Riemann solution in (rho, u, v) space,
with 1-wave and 3-wave rarefaction curves through the left state.

Only one public function: phase_plot_3d()
Call it from main.py after the solver loop.
"""

import plotly.graph_objects as go
import numpy as np
from config import SolverConfig, SolverState, get_primitives


def _wave_curves(cfg: SolverConfig, rho_range: np.ndarray):
    """
    Computes the 1-wave (minus) and 3-wave (plus) rarefaction curves
    through the left state (rho_L, u_L) in the rho-u plane.

    Equation 14 (1-wave, minus sign):
        u = sqrt(a * A(v_L)) * ( -2/(a+1) * (x^((a+1)/2) - rho_L^((a+1)/2)) )
            + u_L / sqrt(a * A(v_L))

    Equation 15 (3-wave, plus sign):
        u = sqrt(a * A(v_L)) * ( +2/(a+1) * (x^((a+1)/2) - rho_L^((a+1)/2)) )
            + u_L / sqrt(a * A(v_L))

    where x is rho (the variable along the curve), a = alpha, A = A(v_L).
    """
    a    = cfg.alpha
    rho_L = cfg.rho_L
    u_L   = cfg.u_L

    # A evaluated at v_L — scalar since v_L is a scalar
    # compute_A expects arrays so wrap and unwrap
    A_L = float(cfg.compute_A(
        np.array([cfg.v_L]),
        np.array([cfg.rho_L])
    )[0])

    coeff = np.sqrt(a * A_L)                          # sqrt(a * A(v_L))
    u_shift = u_L / coeff                              # u_L / sqrt(a * A(v_L))
    exp = (a + 1) / 2.0

    rho_term = rho_range ** exp - rho_L ** exp         # x^((a+1)/2) - rho_L^((a+1)/2)
    fan_term = (2.0 / (a + 1)) * rho_term              # 2/(a+1) * (...)

    u_wave1 = coeff * (-fan_term + u_shift)            # eq 14: minus sign
    u_wave3 = coeff * ( fan_term + u_shift)            # eq 15: plus sign

    return u_wave1, u_wave3



def phase_plot_3d(
    states: list,
    cfg: SolverConfig,
    save_html: str = None
):
    """
    3D phase portrait in (rho, u, v) space.
    Shows:
      - Solution path from numerical solver (black curve)
      - Earlier iterations (gray, fading)
      - Left state L (blue dot)
      - Right state R (red dot)
      - 1-wave rarefaction curve through L (blue curve, in rho-u plane)
      - 3-wave rarefaction curve through L (red curve, in rho-u plane)

    The wave curves live in the v = v_L plane since they are derived
    from the left state.
    """
    fig = go.Figure()

    # --- wave curves in (rho, u, v_L) plane ---
    # use a range of rho values spanning both states
    rho_min = min(cfg.rho_L, cfg.rho_R) * 0.5
    rho_max = max(cfg.rho_L, cfg.rho_R) * 1.5
    rho_range = np.linspace(max(rho_min, 0.01), rho_max, 400)

    try:
        u_wave1, u_wave3 = _wave_curves(cfg, rho_range)

        # 1-wave curve (blue) — lives at v = v_L
        fig.add_trace(go.Scatter3d(
            x=rho_range,
            y=u_wave1,
            z=np.full_like(rho_range, cfg.v_L),
            mode='lines',
            line=dict(color='blue', width=3, dash='dash'),
            name='1-wave curve (eq 14)'
        ))

        # 3-wave curve (red) — lives at v = v_L
        fig.add_trace(go.Scatter3d(
            x=rho_range,
            y=u_wave3,
            z=np.full_like(rho_range, cfg.v_L),
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name='3-wave curve (eq 15)'
        ))

    except Exception as e:
        print(f"Warning: could not compute wave curves: {e}")

    # --- earlier iterations in gray ---
    for i, state in enumerate(states[:-1]):
        rho_i, u_i, v_i = get_primitives(state.U)
        opacity = 0.15 + 0.5 * (i / max(len(states) - 2, 1))
        fig.add_trace(go.Scatter3d(
            x=rho_i, y=u_i, z=v_i,
            mode='lines',
            line=dict(color=f'rgba(150,150,150,{opacity:.2f})', width=1),
            showlegend=False
        ))

    # --- final converged solution path (black) ---
    final = states[-1]
    rho, u, v = get_primitives(final.U)
    fig.add_trace(go.Scatter3d(
        x=rho, y=u, z=v,
        mode='lines',
        line=dict(color='black', width=4),
        name='Solution path'
    ))

    # --- left state L (blue dot) ---
    fig.add_trace(go.Scatter3d(
        x=[cfg.rho_L], y=[cfg.u_L], z=[cfg.v_L],
        mode='markers+text',
        marker=dict(size=8, color='blue'),
        text=['L'], textposition='top center',
        name='Left state L'
    ))

    # --- right state R (red dot) ---
    fig.add_trace(go.Scatter3d(
        x=[cfg.rho_R], y=[cfg.u_R], z=[cfg.v_R],
        mode='markers+text',
        marker=dict(size=8, color='red'),
        text=['R'], textposition='top center',
        name='Right state R'
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Phase Portrait (ρ, u, v)  —  "
                f"Case {cfg.case_number()}, α = {cfg.alpha}, "
                f"Steps = {states[-1].iters}, t = {states[-1].t:.2f}"
            ),
            font=dict(size=14)
        ),
        scene=dict(
            xaxis_title='ρ',
            yaxis_title='u',
            zaxis_title='v',
            xaxis=dict(backgroundcolor='white', gridcolor='lightgray'),
            yaxis=dict(backgroundcolor='white', gridcolor='lightgray'),
            zaxis=dict(backgroundcolor='white', gridcolor='lightgray'),
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=50, b=0),
        width=850,
        height=700,
    )

    if save_html:
        fig.write_html(save_html)
        print(f"Saved phase portrait: {save_html}")

    return fig
