"""
Run everything: generates the 6 2D projection quiver plots and the two
interactive 3D Plotly vector field figures.

Usage:
    python main.py
"""

import plot_projections
import plot_3d

if __name__ == "__main__":
    plot_projections.main()
    plot_3d.main()