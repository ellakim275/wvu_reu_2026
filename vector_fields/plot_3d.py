"""
Interactive 3D vector field (Plotly Cone plot) for the left and right
states -- one standalone HTML figure per state. Both figures are built
over the same explicit (rho, u, v) domain from config.RANGES.
"""

import os
import numpy as np
import plotly.graph_objects as go
import config as cfg
from model import vector_field, shock_speed


def make_cone_figure(state_name, anchor, s):
    grid_rho = np.linspace(*cfg.RANGES["rho"], cfg.GRID_N_3D)
    grid_u = np.linspace(*cfg.RANGES["u"], cfg.GRID_N_3D)
    grid_v = np.linspace(*cfg.RANGES["v"], cfg.GRID_N_3D)
    X, Y, Z = np.meshgrid(grid_rho, grid_u, grid_v)

    P, Q, R = vector_field(X, Y, Z, anchor["rho"], anchor["u"], anchor["v"], s)

    fig = go.Figure(
        data=go.Cone(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            u=P.flatten(),
            v=Q.flatten(),
            w=R.flatten(),
            colorscale="Viridis",
            sizemode="scaled",
            sizeref=cfg.CONE_SIZE,
            anchor="tail",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[anchor["rho"]],
            y=[anchor["u"]],
            z=[anchor["v"]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name=f"{state_name} state",
        )
    )

    fig.update_layout(
        title=(
            f"{state_name} state vector field "
            f"(rho={anchor['rho']}, u={anchor['u']}, v={anchor['v']})"
        ),
        scene=dict(
            xaxis_title="rho", yaxis_title="u", zaxis_title="v",
            xaxis=dict(range=list(cfg.RANGES["rho"])),
            yaxis=dict(range=list(cfg.RANGES["u"])),
            zaxis=dict(range=list(cfg.RANGES["v"])),
        ),
    )
    return fig


def main():
    s = shock_speed()
    left = {"rho": cfg.rho_L, "u": cfg.u_L, "v": cfg.v_L}
    right = {"rho": cfg.rho_R, "u": cfg.u_R, "v": cfg.v_R}

    for state_name, anchor in [("Left", left), ("Right", right)]:
        fig = make_cone_figure(state_name, anchor, s)
        out_dir = cfg.state_dir(state_name)
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, "vector_field_3d.html")
        fig.write_html(fname)
        print(f"saved {fname}")


if __name__ == "__main__":
    main()