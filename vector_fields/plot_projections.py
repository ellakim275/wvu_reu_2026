"""
2D quiver plots of the vector field, projected onto each coordinate plane
(rho-u, rho-v, u-v). All plots draw from the same user-set ranges in
config.py (RHO_MIN/MAX, U_MIN/MAX, V_MIN/MAX) -- nothing here computes its
own window.

Produces two kinds of figures:
  - Per-state plots: one field at a time (left_state/*.png, right_state/*.png)
  - Combined plots: both fields overlaid on the same axes (combined_*.png)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from model import vector_field, shock_speed

NAMES = ["rho", "u", "v"]
LABELS = {"rho": r"$\rho$", "u": "$u$", "v": "$v$"}


def plot_field(ax, var_pair, states, s, title=None):
    """
    Draw the normalized vector field(s) in the (var_pair[0], var_pair[1])
    plane, using the ranges from config.RANGES for both axes. The third
    coordinate is held fixed at each state's own anchor value (a slice
    through that state).

    var_pair: tuple of two of {'rho','u','v'} to put on the (x,y) axes.
    states: list of dicts, each with keys:
        label (str), anchor (dict), color (str),
        show_field (bool, default True) -- if False, only the state's
        point is marked (no quiver arrows), useful for showing a state
        for reference alongside another state's field.
    """
    i, j = NAMES.index(var_pair[0]), NAMES.index(var_pair[1])
    k = [idx for idx in range(3) if idx not in (i, j)][0]
    fixed_name = NAMES[k]

    xmin, xmax = cfg.RANGES[var_pair[0]]
    ymin, ymax = cfg.RANGES[var_pair[1]]
    grid_i = np.linspace(xmin, xmax, cfg.GRID_N)
    grid_j = np.linspace(ymin, ymax, cfg.GRID_N)
    Gi, Gj = np.meshgrid(grid_i, grid_j)

    spacing = min(xmax - xmin, ymax - ymin) / cfg.GRID_N
    arrow_len = spacing * cfg.ARROW_LEN_FRACTION

    for entry in states:
        label, anchor, color = entry["label"], entry["anchor"], entry["color"]
        show_field = entry.get("show_field", True)

        if show_field:
            fixed_val = anchor[fixed_name]
            coords = {var_pair[0]: Gi, var_pair[1]: Gj, fixed_name: np.full_like(Gi, fixed_val)}
            x, y, z = coords["rho"], coords["u"], coords["v"]

            if np.any(x == 0):
                print(f"warning: rho=0 in range for {label} {var_pair} plot; field is undefined there")

            P, Q, R = vector_field(x, y, z, anchor["rho"], anchor["u"], anchor["v"], s)
            comps = {"rho": P, "u": Q, "v": R}
            Vi, Vj = comps[var_pair[0]], comps[var_pair[1]]

            mag = np.sqrt(Vi**2 + Vj**2)
            mag[mag == 0] = np.nan
            Ui, Uj = (Vi / mag) * arrow_len, (Vj / mag) * arrow_len

            ax.quiver(Gi, Gj, Ui, Uj, angles="xy", scale_units="xy", scale=1,
                      color=color, width=0.0025, alpha=0.85)

        ax.plot(anchor[var_pair[0]], anchor[var_pair[1]], ".", color=color,
                markersize=13, label=f"{label} state")

    ax.set_xlabel(LABELS[var_pair[0]])
    ax.set_ylabel(LABELS[var_pair[1]])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if title:
        ax.set_title(title)
    if len(states) > 1:
        ax.legend(loc="best")


def main():
    s = shock_speed()
    left = {"rho": cfg.rho_L, "u": cfg.u_L, "v": cfg.v_L}
    right = {"rho": cfg.rho_R, "u": cfg.u_R, "v": cfg.v_R}
    pairs = [("rho", "u"), ("rho", "v"), ("u", "v")]

    # per-state plots: one field at a time, only that state's own point shown
    for state_name, anchor, color in [("Left", left, "tab:blue"), ("Right", right, "tab:red")]:
        out_dir = cfg.state_dir(state_name)
        os.makedirs(out_dir, exist_ok=True)
        for pair in pairs:
            fig, ax = plt.subplots(figsize=(6, 6))
            title = f"{state_name} state: {LABELS[pair[0]]}-{LABELS[pair[1]]} plane"
            plot_field(ax, pair, [{"label": state_name, "anchor": anchor, "color": color}], s, title=title)
            fname = os.path.join(out_dir, f"{pair[0]}_{pair[1]}.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"saved {fname}")

    # combined plots: each shows ONE state's field, but BOTH states' points
    # marked for context -- 2 (left field / right field) x 3 (variable pairs)
    # = 6 plots total.
    state_info = {
        "Left": {"label": "Left", "anchor": left, "color": "tab:blue"},
        "Right": {"label": "Right", "anchor": right, "color": "tab:red"},
    }
    for pair in pairs:
        for main_name in ["Left", "Right"]:
            other_name = "Right" if main_name == "Left" else "Left"
            main_entry = {**state_info[main_name], "show_field": True}
            other_entry = {**state_info[other_name], "show_field": False}

            fig, ax = plt.subplots(figsize=(8, 7))
            title = (
                f"{main_name} state field: {LABELS[pair[0]]}-{LABELS[pair[1]]} view "
                f"(both states shown)"
            )
            plot_field(ax, pair, [main_entry, other_entry], s, title=title)
            fname = f"combined_{main_name.lower()}_{pair[0]}_{pair[1]}.png"
            fig.tight_layout()
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"saved {fname}")


if __name__ == "__main__":
    main()