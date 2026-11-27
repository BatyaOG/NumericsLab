
from pathlib import Path
import numpy as np

from numericslab import solver, cases, experiments, plotting


def main():
    grid_sizes = [3, 7, 15, 31]

    print("Running NumericsLab experiments...")

    sine_errors = experiments.run_convergence(
        grid_sizes, cases.sine_source,
        cases.zero_boundary, cases.manufactured_exact
    )

    quartic_errors = experiments.run_convergence(
        grid_sizes, cases.quartic_source,
        cases.quartic_boundary, cases.quartic_exact
    )

    u_quadratic = solver.solve_poisson(
        15, cases.quadratic_source, cases.quadratic_boundary
    )

    quadratic_error = np.max(
        np.abs(u_quadratic - cases.manufactured_exact_quadratic(15))
    )

    print(f"Quadratic maximum error: {quadratic_error:.3e}")
    assert quadratic_error < 1e-12

    median_times = experiments.benchmark_solve_times(
        grid_sizes, cases.sine_source, cases.zero_boundary
    )

    n = 31
    u = solver.solve_poisson(
        n, cases.sine_source, cases.zero_boundary
    )

    grid = solver.reconstruct_solution(
        u, n, cases.zero_boundary
    )

    output_dir = Path("figures")

    plotting.plot_solution_heatmap(
        grid, output_dir / "solution_heatmap.png"
    )

    plotting.plot_convergence(
        grid_sizes, sine_errors, quartic_errors,
        output_dir / "convergence.png"
    )

    plotting.plot_error_vs_runtime(
        median_times, sine_errors,
        output_dir / "error_vs_runtime.png"
    )

    plotting.plot_overview(
        grid, grid_sizes, sine_errors,
        quartic_errors, median_times,
        output_dir / "overview.png"
    )

    print("All experiments and figures completed successfully!")


if __name__ == "__main__":
    main()
