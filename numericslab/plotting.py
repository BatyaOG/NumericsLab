
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_figure(fig, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_solution_heatmap(grid, filename):
    x = np.linspace(0, 1, grid.shape[1])
    y = np.linspace(0, 1, grid.shape[0])

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.contourf(x, y, grid, levels=30, cmap="viridis")
    fig.colorbar(image, ax=ax, label="Solution u(x, y)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Numerical Poisson Solution")
    ax.set_aspect("equal")

    save_figure(fig, filename)


def plot_convergence(grid_sizes, sine_errors, quartic_errors, filename):
    h = np.array([1 / (n + 1) for n in grid_sizes])

    reference = 1.7*sine_errors[0] * (h / h[0])**2

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.loglog(h, sine_errors, "o-", label="Sine")
    ax.loglog(h, quartic_errors, "s-", label="Quartic")
    ax.loglog(h, reference, "--", label="Second-order reference")

    ax.set_xlabel("Grid spacing h")
    ax.set_ylabel("Maximum absolute error")
    ax.set_title("Grid Convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    save_figure(fig, filename)


def plot_error_vs_runtime(median_times, errors, filename):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.loglog(median_times, errors, "o-")

    ax.set_xlabel("Median sparse solve time (seconds)")
    ax.set_ylabel("Maximum absolute error")
    ax.set_title("Accuracy vs. Computational Cost")
    ax.grid(True, which="both", alpha=0.3)

    save_figure(fig, filename)

def plot_overview(grid, grid_sizes, sine_errors,
                  quartic_errors, median_times, filename):

    h = np.array([1 / (n + 1) for n in grid_sizes])
    reference = 1.7 * sine_errors[0] * (h / h[0])**2

    fig, axes = plt.subplots(
        1, 3, figsize=(17, 5), constrained_layout=True
    )

    # Panel A: Numerical solution
    x = np.linspace(0, 1, grid.shape[1])
    y = np.linspace(0, 1, grid.shape[0])

    image = axes[0].contourf(
        x, y, grid, levels=30, cmap="viridis"
    )

    fig.colorbar(
        image, ax=axes[0],
        label="Solution u(x, y)", shrink=0.8
    )

    axes[0].set_title("(A) Numerical Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    # Panel B: Convergence
    axes[1].loglog(h, sine_errors, "o-", label="Sine")
    axes[1].loglog(h, quartic_errors, "s-", label="Quartic")
    axes[1].loglog(
        h, reference, "--", label=r"$O(h^2)$ reference"
    )

    axes[1].set_title("(B) Grid Convergence")
    axes[1].set_xlabel("Grid spacing h")
    axes[1].set_ylabel("Maximum absolute error")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    # Panel C: Accuracy vs. solve-only runtime
    axes[2].loglog(median_times, sine_errors, "o-")

    for n, t, error in zip(
        grid_sizes, median_times, sine_errors
    ):
        axes[2].annotate(
            f"n={n}",
            (t, error),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )

    axes[2].set_title("(C) Accuracy vs. Computational Cost")
    axes[2].set_xlabel("Median sparse solve time (s)")
    axes[2].set_ylabel("Maximum absolute error")
    axes[2].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "NumericsLab: Verification and Performance",
        fontsize=15
    )

    save_figure(fig, filename)
