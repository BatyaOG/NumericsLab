# NumericsLab — Technical Report

**Verification and Performance Analysis of a Two-Dimensional Finite-Difference Poisson Solver**

**Date:** November 26, 2026

## Abstract

This project implements and verifies a two-dimensional Poisson solver using second-order finite differences and sparse linear algebra in Python. Three manufactured solutions are used to evaluate the numerical implementation, including zero and nonzero Dirichlet boundary conditions. The sine and quartic cases demonstrate convergence toward second-order accuracy, while the quadratic case is reproduced to approximately floating-point precision. Automated numerical tests, solve-time benchmarks, and reproducible scientific visualizations complement the verification study.

## 1. Problem Formulation

The governing equation is

$$
-\nabla^2u=f
$$

on the unit square $\Omega=[0,1]^2$, subject to prescribed Dirichlet boundary conditions $u=g$ on $\partial\Omega$.

The source function $f$ and boundary function $g$ are passed to the solver as user-defined Python functions.

## 2. Numerical Method

The standard five-point finite-difference approximation is applied at interior grid nodes:

$$
\frac{4u_{i,j}-u_{i-1,j}-u_{i+1,j}-u_{i,j-1}-u_{i,j+1}}{h^2}=f_{i,j}.
$$

The grid spacing is $h=1/(n+1)$, where $n$ represents the number of interior nodes along each coordinate direction.

The resulting system contains $n^2$ unknowns and is expressed as

$$
A\mathbf{u}=\mathbf{b}.
$$

The matrix is assembled in SciPy's LIL sparse format, converted to CSR, and solved using `scipy.sparse.linalg.spsolve`.

Prescribed boundary values contribute to the right-hand side for interior nodes adjacent to the domain boundary.

## 3. Verification Methodology

Three manufactured solutions are evaluated:

| Case | Exact solution | Source function |
|---|---|---|
| Sine | $\sin(\pi x)\sin(\pi y)$ | $2\pi^2\sin(\pi x)\sin(\pi y)$ |
| Quadratic | $x^2+y^2$ | $-4$ |
| Quartic | $x^4+y^4$ | $-12(x^2+y^2)$ |

The maximum absolute error is calculated over the interior nodes.

The observed convergence order for successive halving of grid spacing is

$$
p=\log_2(E_{\mathrm{coarse}}/E_{\mathrm{fine}}).
$$

The sine and quartic cases are used to measure convergence. The quadratic case is used to verify exact reproduction of a polynomial whose Laplacian is represented exactly by the five-point stencil, apart from floating-point effects.

## 4. Numerical Results

| Interior nodes per direction | Sine error | Quartic error |
|---|---:|---:|
| 3 | 5.30293e-2 | 1.75781e-2 |
| 7 | 1.29507e-2 | 4.54891e-3 |
| 15 | 3.21896e-3 | 1.14759e-3 |
| 31 | 8.03578e-4 | 2.87558e-4 |

Observed convergence orders:

| Refinement | Sine | Quartic |
|---|---:|---:|
| 3 → 7 | 2.034 | 1.950 |
| 7 → 15 | 2.008 | 1.987 |
| 15 → 31 | 2.002 | 1.997 |

Both error sequences approach the theoretical second-order convergence rate.

For the quadratic case at $n=15$, the fresh-environment experiment reported a maximum error of approximately $4.44\times10^{-16}$.

These results support correct implementation of the five-point discretization and Dirichlet boundary treatment for the tested problems.

A small algebraic residual alone would not establish discretization accuracy. The comparisons with manufactured analytical solutions provide direct evidence about solution error.

## 5. Performance Analysis

Performance is measured using the sparse linear-system solution operation.

For each grid resolution, the implementation performs one untimed warm-up solve followed by 20 timed solves. The median elapsed time is used as the representative measurement.

Matrix assembly, right-hand-side construction, reconstruction, and visualization are excluded from timing.

The error-versus-runtime figure illustrates the accuracy–computational-cost tradeoff across the tested grid resolutions.

Because timings depend on hardware, software versions, and operating-system conditions, these benchmarks should be interpreted as local demonstrations rather than general performance predictions.

No asymptotic runtime complexity is inferred from the four measured grid resolutions.

## 6. Reproducibility and Software Quality

The implementation separates solver operations, analytical test cases, experiments, plotting, and workflow execution into individual Python modules.

Six pytest tests verify indexing, matrix structure, boundary contributions, polynomial solution accuracy, convergence behavior, and reconstruction.

A fresh Git clone was tested in an isolated Python virtual environment on macOS 15.3 (ARM64), with the following versions:

| Software | Version |
|---|---|
| Python | 3.13.9 |
| NumPy | 2.5.5 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| pytest | 9.2.0 |

All six tests passed. The full experiment workflow executed successfully and regenerated four scientific figures.

The repository additionally contains a GitHub Actions workflow for automated testing on an Ubuntu runner.

## 7. Limitations and Conclusions

The solver is restricted to a two-dimensional uniform grid, a constant-coefficient Laplacian, prescribed Dirichlet conditions, and sparse direct linear solution.

It does not model time dependence, heterogeneous permeability, or physical reservoir wells. Although the governing equation is mathematically related to steady constant-coefficient Darcy flow, the present implementation is not a reservoir simulator.

The verification results demonstrate the expected second-order convergence behavior for two manufactured solutions, including nonzero boundary conditions.

The project illustrates a complete small-scale scientific-computing workflow: numerical implementation, analytical verification, automated testing, performance measurement, visualization, and reproducible execution.

**Repository:** https://github.com/BatyaOG/NumericsLab