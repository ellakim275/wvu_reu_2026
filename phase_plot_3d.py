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
from cases.case_solver import get_intermediates, get_wave_curves

#meow old function but want to save for now for reference 
"""def _wave_curves(cfg: SolverConfig, rho_range: np.ndarray):
    a     = cfg.alpha
    rho_L = cfg.rho_L
    u_L   = cfg.u_L

    A_L = float(cfg.compute_A(
        np.array([cfg.v_L]),
        np.array([cfg.p])
    )[0])

    coeff = np.sqrt(a * A_L)
    exp   = (a + 1) / 2.0

    # R1 rarefaction — rho < rho_L, u increases as rho decreases
    rho_r1   = rho_range[rho_range < rho_L]
    u_wave1  = u_L + coeff * (2.0/(a+1)) * (1.0/rho_L**exp - 1.0/rho_r1**exp)

    # R3 rarefaction — rho > rho_L, u increases as rho increases  
    # from page 79 inverse R3 anchored at L:
    # u = u_L - coeff * (2/(a+1)) * (1/rho_L^exp - 1/rho^exp)
    rho_r3   = rho_range[rho_range > rho_L]
    u_wave3  = u_L - coeff * (2.0/(a+1)) * (1.0/rho_L**exp - 1.0/rho_r3**exp)

    return rho_r1, u_wave1, rho_r3, u_wave3"""

def phase_plot_3d(states, cfg, save_html=None):
    fig = go.Figure()

    rho_min   = 0.01                              # always start near zero
    rho_max   = max(cfg.rho_L, cfg.rho_R) * 1.5  # this part is fine
    rho_range = np.linspace(rho_min, rho_max, 400)

    # get intermediates and curves for whatever case this is
    intermediates = get_intermediates(cfg)
    curves        = get_wave_curves(cfg, intermediates, rho_range)

    # plot all curves generically
    for c in curves:
        fig.add_trace(go.Scatter3d(
            x=c['rho'], y=c['u'], z=c['v'],
            mode='lines',
            line=dict(color=c['color'], width=3, dash='dash'),
            name=c['name']
        ))

    # plot M1 and M2 generically
    for label, key_rho, key_u, key_v, color in [
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
