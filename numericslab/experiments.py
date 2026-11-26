
import numpy as np
import math

from numericslab.solver import solve_poisson


def run_convergence(grid_sizes, source_fn, boundary_fn, exact_fn):
    errors = []

    for n in grid_sizes:
        h = 1 / (n + 1)

        u = solve_poisson(n, source_fn, boundary_fn)
        u_exact = exact_fn(n)

        error = np.max(np.abs(u - u_exact))
        errors.append(error)

        print(f"n={n}, h={h:.6f}, max error={error:.8e}")

    return errors


def convergence_orders(errors):
    orders = []

    for i in range(len(errors) - 1):
        p = math.log2(errors[i] / errors[i + 1])
        orders.append(p)

    return orders

import time
from scipy.sparse.linalg import spsolve
from numericslab.solver import build_sparse_poisson_matrix, build_rhs


def benchmark_solve_times(grid_sizes, source_fn, boundary_fn):
    median_times = []

    for n in grid_sizes:
        h = 1 / (n + 1)

        A = build_sparse_poisson_matrix(n, h)
        b = build_rhs(n, source_fn, boundary_fn)

        runtimes = []

        # Warm-up
        spsolve(A, b)

        # Measure 20 solve operations
        for i in range(20):
            start = time.perf_counter()
            spsolve(A, b)
            end = time.perf_counter()

            runtimes.append(end - start)

        median_times.append(np.median(runtimes))

    return median_times
