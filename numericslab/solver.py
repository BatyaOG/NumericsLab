
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def grid_to_index(row, col, n):
    return row*n + col


def build_sparse_poisson_matrix(n, h):
    A = lil_matrix((n*n, n*n))

    for row in range(n):
        for col in range(n):
            k = grid_to_index(row, col, n)

            A[k,k] = 4/h**2

            if col > 0:
                A[k,k-1] = -1/h**2
            if col < n-1:
                A[k,k+1] = -1/h**2
            if row > 0:
                A[k,k-n] = -1/h**2
            if row < n-1:
                A[k,k+n] = -1/h**2

    return A.tocsr()


def build_rhs(n, source_fn, boundary_fn):
    h = 1/(n+1)
    b = np.zeros(n*n)

    for row in range(n):
        for col in range(n):
            x = (col+1)*h
            y = (row+1)*h
            k = grid_to_index(row, col, n)

            b[k] = source_fn(x,y)

            if col == 0:
                b[k] += boundary_fn(0,y)/h**2
            if col == n-1:
                b[k] += boundary_fn(1,y)/h**2
            if row == 0:
                b[k] += boundary_fn(x,0)/h**2
            if row == n-1:
                b[k] += boundary_fn(x,1)/h**2

    return b


def solve_poisson(n, source_fn, boundary_fn):
    h = 1/(n+1)

    A = build_sparse_poisson_matrix(n,h)
    b = build_rhs(n,source_fn,boundary_fn)

    u = spsolve(A,b)
    return u


def reconstruct_solution(u, n, boundary_fn):
    grid = np.zeros((n+2,n+2))

    grid[1:-1,1:-1] = u.reshape((n,n))
    h = 1/(n+1)

    for row in range(n+2):
        y = row*h
        grid[row,0] = boundary_fn(0,y)
        grid[row,n+1] = boundary_fn(1,y)

    for col in range(n+2):
        x = col*h
        grid[0,col] = boundary_fn(x,0)
        grid[n+1,col] = boundary_fn(x,1)

    return grid
