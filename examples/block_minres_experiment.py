import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.append(str(Path(__file__).resolve().parent.parent))

from util import (
    preconditioned_block_power,
    MinresMultiRHS,
    block_minres,
    power_solve_cb,
    make_minres_callback,
)


def _build_symmetric_matrix(n: int, nnz_per_row: int, spread: float, seed: int, dtype) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)

    rows = np.repeat(np.arange(n), nnz_per_row)
    cols = np.empty((n, nnz_per_row), dtype=np.int64)

    offsets = rng.normal(loc=0.0, scale=spread, size=(n, nnz_per_row))

    for i in range(n):
        raw = i + offsets[i]
        clipped = np.clip(raw, 0, n - 1)
        cols[i] = clipped.round().astype(np.int64)
        cols[i, 0] = i

    cols = cols.reshape(-1)

    rows = rows.astype(np.int64)
    data = rng.uniform(-1.0, 1.0, size=rows.size).astype(dtype)

    base = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = (base + base.T).tocsr()
    A.sum_duplicates()
    return A


def main() -> None:
    parser = argparse.ArgumentParser(description="Block MINRES experiment with configurable initial subspace generation.")
    parser.add_argument("--n", type=int, required=True, help="Matrix dimension")
    parser.add_argument("--nnz-per-row", type=int, required=True, help="Nonzeros per row in the base pattern")
    parser.add_argument("--spread", type=float, required=True, help="Standard deviation for band offsets")
    parser.add_argument("--block-size", type=int, required=True, help="Number of vectors in the block method")
    parser.add_argument("--outer-iters", type=int, required=True, help="Outer iterations / sweeps for subspace refinement")
    parser.add_argument("--inner-iters", type=int, default=1, help="Inner iterations for eigensolver acceleration (if used)")
    parser.add_argument("--precond-maxiter", type=int, required=True, help="Iterations for the MinresMultiRHS preconditioner")
    parser.add_argument("--precond-rtol", type=float, required=True, help="Relative tolerance for the MinresMultiRHS preconditioner")
    parser.add_argument("--block-minres-iters", type=int, required=True, help="Iterations for block MINRES")
    parser.add_argument("--block-minres-tol", type=float, required=True, help="Stopping tolerance for block MINRES")
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--use-eigensolver", action="store_true", help="Use Rayleigh-Ritz block eigensolver for subspace construction")
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    if args.outer_iters <= 0:
        raise ValueError("--outer-iters must be positive")
    if args.inner_iters <= 0:
        raise ValueError("--inner-iters must be positive")
    if args.block_minres_iters <= 0:
        raise ValueError("--block-minres-iters must be positive")

    A = _build_symmetric_matrix(args.n, args.nnz_per_row, args.spread, seed=args.seed, dtype=dtype)

    rng = np.random.default_rng(args.seed + 1)
    x_true = rng.standard_normal(args.n).astype(dtype)
    b = A @ x_true

    precond_solver = MinresMultiRHS(A, backend="numpy")

    def apply_preconditioner(_, V):
        X, _ = precond_solver.solve(V, rtol=args.precond_rtol, maxiter=args.precond_maxiter)
        return X

    if args.use_eigensolver:
        rng = np.random.default_rng(args.seed + 2)
        V_current = rng.standard_normal((args.n, args.block_size)).astype(dtype)
        V_current, _ = np.linalg.qr(V_current, mode="reduced")

        apply_inverse_cb = make_minres_callback(
            rtol=args.precond_rtol,
            maxiter=args.precond_maxiter,
            backend="numpy",
        )

        eig_iter = {"count": 0}

        def eig_callback(V_iter, w_iter, residuals):
            eig_iter["count"] += 1
            w_np = np.asarray(w_iter, dtype=float)
            res_np = np.asarray(residuals, dtype=float) if residuals is not None else None
            neg = w_np[w_np < 0]
            pos = w_np[w_np > 0]
            min_neg = neg.min() if neg.size else float("nan")
            max_neg = neg.max() if neg.size else float("nan")
            min_pos = pos.min() if pos.size else float("nan")
            max_pos = pos.max() if pos.size else float("nan")
            mid_idx = len(w_np) // 2
            mid_rel = res_np[mid_idx] if res_np is not None and mid_idx < res_np.size else float("nan")
            print(
                f"Eigen iter {eig_iter['count']:02d}: min_neg={min_neg:.3e}, max_neg={max_neg:.3e}, "
                f"min_pos={min_pos:.3e}, max_pos={max_pos:.3e}, mid_relres={mid_rel:.3e}"
            )

        for _ in range(args.outer_iters):
            _, V_current = power_solve_cb(
                A,
                V_current,
                apply_inverse_cb=apply_inverse_cb,
                outer_iter=max(1, args.inner_iters),
                backend="numpy",
                callback=eig_callback,
            )

        W = V_current[:, : args.block_size]
    else:
        W_current = None
        for outer in range(args.outer_iters):
            seed = args.seed + 2 + outer if W_current is None else None
            W_current = preconditioned_block_power(
                A,
                apply_preconditioner=apply_preconditioner,
                block_size=args.block_size,
                iterations=max(1, args.inner_iters),
                backend="numpy",
                seed=seed,
                V0=W_current,
            )
        W = W_current

    b_vec = np.asarray(b, dtype=dtype)
    V_aug = np.concatenate([W, b_vec.reshape(-1, 1)], axis=1)
    Q, R = np.linalg.qr(V_aug, mode="reduced")

    rhs_proj = Q.T @ b_vec
    c = np.linalg.solve(R, rhs_proj)
    Rc = R @ c

    Y, info = block_minres(
        A,
        Q,
        iterations=args.block_minres_iters,
        backend="numpy",
        tol=args.block_minres_tol,
    )

    x = Y @ Rc

    residual = A @ x - b_vec
    res_norm = np.linalg.norm(residual)
    rel_res = res_norm / np.linalg.norm(b_vec)

    mode_label = "Rayleigh-Ritz eigensolver" if args.use_eigensolver else "preconditioned power sweeps"

    print(f"Matrix dimension: {args.n}")
    print(f"Block size: {args.block_size}")
    print(f"Initial subspace generator: {mode_label}")
    print(f"Outer iterations: {args.outer_iters}")
    if args.use_eigensolver:
        print(f"Inner iterations (per outer): {args.inner_iters}")
    print(f"Block MINRES iterations used: {info['iterations']}")
    history = np.asarray(info["residual_history"], dtype=dtype)
    print("Block MINRES aggregate residuals per iteration:")
    if history.size:
        for step, res_vec in enumerate(history, start=1):
            aggregate = np.linalg.norm(res_vec)
            print(f"  iter {step:02d}: {aggregate:.3e}")
    else:
        print("  (no iterations performed)")
    print(f"Final solve residual: {res_norm:.3e} (relative {rel_res:.3e})")
    print(f"Error to ground truth x: {np.linalg.norm(x - x_true):.3e}")


if __name__ == "__main__":
    main()
