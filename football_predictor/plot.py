"""Charts for simulation results. Needs matplotlib."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_positions(summary: pd.DataFrame, positions: np.ndarray, path: Path, title: str) -> Path:
    """Heatmap of each team's chance of finishing in each league position, best team on top.

    Row i of `positions` must belong to row i of `summary`, as returned by `simulate_season`.
    """
    if positions.shape != (len(summary), len(summary)):
        raise ValueError("positions must be a square matrix with one row per team in summary")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-summary["expected_points"].to_numpy())
    teams = summary.index[order]
    grid = positions[order] * 100
    n = len(teams)

    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.32 * n + 1.5))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(grid.max(), 1), aspect="auto")
    ax.set_xticks(range(n), [str(i + 1) for i in range(n)])
    ax.set_yticks(range(n), [f"{team}  ({summary['expected_points'].iloc[i]:.0f} pts)" for team, i in zip(teams, order)])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Final position")
    for row in range(n):
        for col in range(n):
            value = grid[row, col]
            if value >= 1:
                colour = "white" if value > grid.max() * 0.6 else "black"
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=7, color=colour)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
