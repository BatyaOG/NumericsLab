
import numpy as np
import math
import time
from scipy.sparse.linalg import spsolve
from numericslab.solver import solve_poisson, build_sparse_poisson_matrix, build_rhs


def run_convergence(grid_sizes, source_fn, boundary_fn, exact_fn):
    errors = []

    for n in grid_sizes:
        h = 1/(n+1)

        u = solve_poisson(n,source_fn,boundary_fn)
        u_exact = exact_fn(n)

        error = np.max(np.abs(u-u_exact))
        errors.append(error)

        print(f"n={n}, h={h:.6f}, max error={error:.8e}")

    return errors


def convergence_orders(errors, h_values):
    if len(errors) != len(h_values):
        raise ValueError("Errors and grid spacings must have equal lengths")

    if len(errors) < 2:
        raise ValueError("At least two measurements are required")

    if any(not math.isfinite(error) for error in errors):
        raise ValueError("Errors must be finite")

    if any(not math.isfinite(h) for h in h_values):
        raise ValueError("Grid spacings must be finite")

    if any(error <= 0 for error in errors):
        raise ValueError("Errors must be positive")

    if any(h <= 0 for h in h_values):
        raise ValueError("Grid spacings must be positive")

    if any(h_values[i] <= h_values[i+1] for i in range(len(h_values)-1)):
        raise ValueError("Grid spacings must strictly decrease")

    orders = []

    for i in range(len(errors)-1):
        p = math.log(errors[i]/errors[i+1]) / math.log(h_values[i]/h_values[i+1])
        orders.append(p)

    return orders


def benchmark_solve_times(grid_sizes, source_fn, boundary_fn):
    median_times = []

    for n in grid_sizes:
        h = 1/(n+1)

        A = build_sparse_poisson_matrix(n,h)
        b = build_rhs(n,source_fn,boundary_fn)

        runtimes = []

        spsolve(A,b)

        for i in range(20):
            start = time.perf_counter()
            spsolve(A,b)
            end = time.perf_counter()

            runtimes.append(end-start)

        median_times.append(np.median(runtimes))

    return median_times
