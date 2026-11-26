
import numpy as np

from numericslab import solver, cases, experiments


def test_grid_index():
    assert solver.grid_to_index(1, 2, 3) == 5
    assert solver.grid_to_index(2, 2, 3) == 8


def test_sparse_matrix():
    A = solver.build_sparse_poisson_matrix(3, 0.25)

    assert A.shape == (9, 9)
    assert A.nnz == 33

    # Center point has four neighbors
    assert np.isclose(A[4, 4], 64.0)

    for k in [1, 3, 5, 7]:
        assert np.isclose(A[4, k], -16.0)

    # Matrix must be symmetric
    np.testing.assert_allclose(A.toarray(), A.toarray().T)


def test_quadratic_rhs():
    b = solver.build_rhs(
        3, cases.quadratic_source, cases.quadratic_boundary
    )

    expected = np.array([
        -2, 0, 22,
         0, -4, 16,
        22, 16, 46
    ])

    np.testing.assert_allclose(b, expected)


def test_quadratic_solution():
    n = 7

    u = solver.solve_poisson(
        n, cases.quadratic_source, cases.quadratic_boundary
    )

    exact = cases.manufactured_exact_quadratic(n)

    assert np.max(np.abs(u - exact)) < 1e-12


def test_second_order_convergence():
    grid_sizes = [7, 15, 31]

    for source_fn, boundary_fn, exact_fn in [
        (cases.sine_source,
         cases.zero_boundary,
         cases.manufactured_exact),

        (cases.quartic_source,
         cases.quartic_boundary,
         cases.quartic_exact)
    ]:
        errors = experiments.run_convergence(
            grid_sizes, source_fn, boundary_fn, exact_fn
        )

        orders = experiments.convergence_orders(errors)

        assert all(1.8 < p < 2.2 for p in orders)


def test_solution_reconstruction():
    n = 3

    u = solver.solve_poisson(
        n, cases.quadratic_source, cases.quadratic_boundary
    )

    grid = solver.reconstruct_solution(
        u, n, cases.quadratic_boundary
    )

    assert grid.shape == (5, 5)
    assert np.isclose(grid[2, 2], 0.5)
    assert np.isclose(grid[2, 4], 1.25)
    assert np.isclose(grid[4, 4], 2.0)
