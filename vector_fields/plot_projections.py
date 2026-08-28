"""
2D quiver plots of the vector field, projected onto each coordinate plane
(rho-u, rho-v, u-v). All plots draw from the same user-set ranges in
config.py (RHO_MIN/MAX, U_MIN/MAX, V_MIN/MAX) -- nothing here computes its
own window.

Produces two kinds of figures:
  - Per-state plots: one field at a time (left_state/*.png, right_state/*.png)
  - Combined plots: both fields overlaid on the same axes (combined_*.png)

On top of the vector field, the (rho, u) panels also get the two analytic
wave curves through that panel's state:
  - S curve: the shock (Rankine-Hugoniot) branch
  - R curve: the rarefaction branch (integrated characteristic speed)
Each is a single continuous curve over rho > 0, built from two piecewise
closed-form expressions (one valid for 0 < x < rho_s, one for x > rho_s)
that agree at x = rho_s.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from model import vector_field, shock_speed, A

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


def shock_curve(x, rho_s, u_s, v_s, a):
    """
    Shock (Rankine-Hugoniot) branch through (rho_s, u_s, v_s):

        u = u_s - sqrt( A(v_s) * (rho_s - x)/(rho_s*x) * (1/x^a - 1/rho_s^a) )

    This single expression is real-valued and continuous for all x > 0
    (both x < rho_s and x > rho_s give a nonnegative radicand), so no
    piecewise split is needed here.
    """
    x = np.asarray(x, dtype=float)
    Av = A(v_s)
    inner = Av * (rho_s - x) / (rho_s * x) * (1.0 / x**a - 1.0 / rho_s**a)
    inner = np.where(inner < 0, np.nan, inner)  # guard against tiny negative roundoff
    return u_s - np.sqrt(inner)


def rarefaction_curve(x, rho_s, u_s, v_s, a):
    """
    Rarefaction branch through (rho_s, u_s, v_s), i.e. the integrated
    characteristic speed. The closed form flips sign across x = rho_s:

        x > rho_s:      u = sqrt(a*A(v_s)) * ( -2/(a+1)*(x^-(a+1)/2 - rho_s^-(a+1)/2) + u_s/sqrt(a*A(v_s)) )
        0 < x < rho_s:  u = sqrt(a*A(v_s)) * (  2/(a+1)*(x^-(a+1)/2 - rho_s^-(a+1)/2) + u_s/sqrt(a*A(v_s)) )

    Both pieces agree at x = rho_s (bracket -> 0), so this stitches into
    one continuous curve over all x > 0.
    """
    x = np.asarray(x, dtype=float)
    Av = A(v_s)
    pref = np.sqrt(a * Av)
    sign = np.where(x > rho_s, -1.0, 1.0)
    bracket = (2.0 / (a + 1.0)) * (x**(-(a + 1.0) / 2.0) - rho_s**(-(a + 1.0) / 2.0))
    return pref * (sign * bracket + u_s / pref)


def plot_wave_curves(ax, rho_s, u_s, v_s, color, a=None, n=400):
    """
    Overlay the shock curve (solid) and rarefaction curve (dashed) through
    ONE state onto an existing (rho, u) axes. Which state's curves you get
    is entirely determined by the (rho_s, u_s, v_s) you pass in -- callers
    below pass the Left state's values on Left panels and the Right
    state's values on Right panels.
    """
    if a is None:
        a = cfg.a  # <-- rename to match the actual exponent name in config.py if different

    xmin, xmax = cfg.RANGES["rho"]
    xmin = max(xmin, 1e-6)  # avoid the x = 0 singularity
    x = np.linspace(xmin, xmax, n)

    u_shock = shock_curve(x, rho_s, u_s, v_s, a)
    u_raref = rarefaction_curve(x, rho_s, u_s, v_s, a)

    ax.plot(x, u_shock, "-", color=color, linewidth=1.5, label="S curve")
    ax.plot(x, u_raref, "--", color=color, linewidth=1.5, label="R curve")


def main():
    s = shock_speed()
    left = {"rho": cfg.rho_L, "u": cfg.u_L, "v": cfg.v_L}
    right = {"rho": cfg.rho_R, "u": cfg.u_R, "v": cfg.v_R}
    pairs = [("rho", "u"), ("rho", "v"), ("u", "v")]

    # per-state plots: one field at a time, only that state's own point shown
    for state_name, anchor, color in [("Left", left, "#999999"), ("Right", right, "black")]:
        out_dir = cfg.state_dir(state_name)
        os.makedirs(out_dir, exist_ok=True)
        for pair in pairs:
            fig, ax = plt.subplots(figsize=(6, 6))
            title = f"{state_name} state: {LABELS[pair[0]]}-{LABELS[pair[1]]} plane"
            plot_field(ax, pair, [{"label": state_name, "anchor": anchor, "color": color}], s, title=title)

            # wave curves only make sense in the (rho, u) plane; use THIS
            # panel's own state (Left panel -> Left curves, Right -> Right)
            if pair == ("rho", "u"):
                plot_wave_curves(ax, anchor["rho"], anchor["u"], anchor["v"], color)
                ax.legend(loc="best")

            fname = os.path.join(out_dir, f"{pair[0]}_{pair[1]}.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"saved {fname}")

    # combined plots: each shows ONE state's field, but BOTH states' points
    # marked for context -- 2 (left field / right field) x 3 (variable pairs)
    # = 6 plots total.
    state_info = {
    "Left": {"label": "Left", "anchor": left, "color": "#999999"},
    "Right": {"label": "Right", "anchor": right, "color": "black"},
    }
    for pair in pairs:
        for main_name in ["Left", "Right"]:
            other_name = "Right" if main_name == "Left" else "Left"
            main_entry = {**state_info[main_name], "show_field": True}
            other_entry = {**state_info[other_name], "show_field": False}

            fig, ax = plt.subplots(figsize=(8, 7))
            title = (
                f"{main_name} state field: {LABELS[pair[0]]}-{LABELS[pair[1]]} view "
            )
            plot_field(ax, pair, [main_entry, other_entry], s, title=title)

            # again, wave curves use whichever state is the "main" field
            # for this panel (main_name), not the reference-only state
            if pair == ("rho", "u"):
                m = state_info[main_name]
                plot_wave_curves(ax, m["anchor"]["rho"], m["anchor"]["u"], m["anchor"]["v"], m["color"])
                ax.legend(loc="best")

            fname = f"combined_{main_name.lower()}_{pair[0]}_{pair[1]}.png"
            fig.tight_layout()
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"saved {fname}")


if __name__ == "__main__":
    main()