# Guidance for Coding Agents

This project is a playground for Krylov subspace methods that run on both CPUs and GPUs.  The code aims for clarity and ease of experimentation; performance optimizations are secondary.

## Key points

* **Backend duality** – Most functions should work with either NumPy/SciPy or CuPy.  Use the helpers in `util.py` (for example `_Backend` and `_detect_backend`) to keep new code backend-agnostic.
* **CuPy optionality** – GPU support is provided through CuPy but the repository must remain usable without it.  Guard CuPy imports appropriately.
* **Existing pieces** – The repository currently includes:
  * Block Chebyshev iteration (`block_chebyshev`/`power_cheb`).
  * A batched multi-RHS MINRES solver (`MinresMultiRHS`).
  * `power_solve_cb`, a power method that accepts a user-supplied linear solver callback and exposes a convergence callback.
* **Development style** – Prefer readable, well-commented code over highly tuned implementations.  Examples belong in the `examples/` directory.
* **Efficiency** – Make every effort to minimize temporary matrices/vectors and avoid unnecessary data copies.
* **Checks** – Run `python -m py_compile` on modified Python files.  When practical, execute any relevant scripts in `examples/` to ensure they still run.
