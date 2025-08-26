# Krylov-bits

This repository is a small testbed for experimenting with Krylov subspace algorithms on both CPU and GPU.  Convenience and clarity take precedence over squeezing out every last FLOP.  GPU execution is provided through [CuPy](https://cupy.dev/), and we make a best effort to keep all functionality working with either the NumPy/SciPy or the CuPy backend.

## Components

* **Chebyshev method** – block Chebyshev iteration for polynomial filtering.  Useful as a lightweight preconditioner or as a building block in other iterations.
* **Batched MINRES** – a synchronized multi right-hand-side MINRES solver that can serve as a standalone routine or as a callback in other algorithms.
* **Power iteration with callbacks** – `power_solve_cb` performs block power iteration where the inverse step is supplied by a user-provided linear solver callback.  A convergence callback can monitor residuals after each outer iteration.

The `examples/` directory demonstrates these pieces in action.

## Goals

* Support both CPU and GPU backends with minimal code changes.
* Favor approachability and experimentation over peak performance.
* Keep dependencies light and rely on standard scientific Python tools.
