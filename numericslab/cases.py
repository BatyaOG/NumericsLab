from numericslab.solver import grid_to_index

import numpy as np
def quadratic_source(x, y):
    return -4.0

def quadratic_boundary(x, y):
    return x**2 + y**2

def manufactured_exact_quadratic(n):
    h=1/(1+n)
    b = np.zeros(n*n)
    for row in range(n):
        for col in range(n):
            x=(col+1)*h
            y=(row+1)*h
            k=grid_to_index(row, col, n)
            b[k]=x**2+y**2
    return b

def sine_source(x, y):
    return 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)

def zero_boundary(x, y):
    return 0.0

def manufactured_exact(n):
    h = 1 / (n + 1)
    u_exact = np.zeros(n*n)

    for row in range(n):
        for col in range(n):
            x = (col + 1) * h
            y = (row + 1) * h
            k = grid_to_index(row, col, n)
            u_exact[k] = np.sin(np.pi*x) * np.sin(np.pi*y)

    return u_exact

def quartic_source(x, y):
    return -12 * (x**2 + y**2)

def quartic_boundary(x, y):
    return x**4 + y**4

def quartic_exact(n):
    h = 1 / (n + 1)
    u_exact = np.zeros(n*n)

    for row in range(n):
        for col in range(n):
            x = (col + 1) * h
            y = (row + 1) * h
            k = grid_to_index(row, col, n)
            u_exact[k] = x**4 + y**4

    return u_exact
