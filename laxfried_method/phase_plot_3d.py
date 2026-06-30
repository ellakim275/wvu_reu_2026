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
from cases import testing as curves_only


def phase_plot_3d(states, cfg, save_html=None, curve_mode='surface'):
    """
    curve_mode controls how the R1/S1/R3/S3 wave curves (all anchored at L)
    are rendered:
        'surface' - translucent extruded ribbons spanning v_L to ~v_R (default)
        'line'    - flat dashed lines at v = v_L (the original style)
        'both'    - both surfaces and lines
        'none'    - skip the wave curves entirely
    """
    fig = go.Figure()

    rho_min   = 0.01                              # always start near zero
    rho_max   = max(cfg.rho_L, cfg.rho_R, 2.0) * 1.5  # ensure curves/surfaces extend past rho=2
    rho_range = np.linspace(rho_min, rho_max, 400)

    # get all four wave curves (R1, S1, R3, S3) anchored at L
    intermediates = curves_only.compute_intermediates(cfg)
    curves = curves_only.wave_curves(cfg, intermediates, rho_range)

    if curve_mode not in ('surface', 'line', 'both', 'none'):
        raise ValueError(f"curve_mode must be one of 'surface', 'line', 'both', 'none' (got {curve_mode!r})")

    if curve_mode in ('surface', 'both'):
        # extrude each (rho, u) curve along v from v_L to a bit past v_R as a
        # semi-transparent surface
        v_span = cfg.v_R - cfg.v_L
        if abs(v_span) < 1e-9:
            # v_L == v_R (or nearly so): there's no real span to extend past,
            # so fall back to a small fixed thickness instead of a degenerate
            # zero-width ribbon (which Plotly renders as nothing at all).
            v_end = cfg.v_L + 0.05 * (abs(cfg.v_L) + 1.0)
        else:
            v_end = cfg.v_R + 0.10 * v_span   # extend 10% further than v_R, away from v_L
        v_sweep = np.linspace(cfg.v_L, v_end, 2)   # 2 points is enough for a flat ruled surface

        for c in curves:
            n_rho = len(c['rho'])
            if n_rho == 0:
                continue
            # build (n_rho, 2) grids: rho/u repeat across the v sweep, v varies along columns
            rho_grid = np.tile(c['rho'].reshape(-1, 1), (1, len(v_sweep)))
            u_grid   = np.tile(c['u'].reshape(-1, 1), (1, len(v_sweep)))
            v_grid   = np.tile(v_sweep.reshape(1, -1), (n_rho, 1))

            fig.add_trace(go.Surface(
                x=rho_grid, y=u_grid, z=v_grid,
                surfacecolor=np.zeros_like(rho_grid),
                colorscale=[[0, c['color']], [1, c['color']]],
                showscale=False,
                opacity=0.4,
                name=c['name'],
                showlegend=True,
            ))

    if curve_mode in ('line', 'both'):
        # flat dashed line at v = v_L for each curve (the original style)
        for c in curves:
            if len(c['rho']) == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=c['rho'], y=c['u'], z=c['v'],
                mode='lines',
                line=dict(color=c['color'], width=3, dash='dash'),
                name=c['name']
            ))

    # plot M1 and M2 generically
    """for label, key_rho, key_u, key_v, color in [
        ('M1', 'rho_M1', 'u_M1', 'v_M1', 'orange'),
        ('M2', 'rho_M2', 'u_M2', 'v_M2', 'green'),
    ]:
        if key_rho in intermediates:
            fig.add_trace(go.Scatter3d(
                x=[intermediates[key_rho]],
                y=[intermediates[key_u]],
                z=[intermediates[key_v]],
                mode='markers+text',
                marker=dict(size=8, color=color),
                text=[label], textposition='top center',
                name=f'Intermediate state {label}'
            ))"""

    

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
        marker=dict(size=4, color='blue'),
        text=['L'], textposition='top center',
        name='Left state L'
    ))

    # --- right state R (red dot) ---
    fig.add_trace(go.Scatter3d(
        x=[cfg.rho_R], y=[cfg.u_R], z=[cfg.v_R],
        mode='markers+text',
        marker=dict(size=4, color='red'),
        text=['R'], textposition='top center',
        name='Right state R'
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Phase Portrait (ρ, u, v)  —  "
                f"Case {cfg.case_num}, α = {cfg.alpha}, "
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