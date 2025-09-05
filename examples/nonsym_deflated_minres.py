import argparse
import logging
import os
import sys
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from util import (
    random_nonsymmetric_matrix,
    power_solve_cb,
    make_minres_callback,
    DeflatedMinres,
    _detect_backend,
)

logger = logging.getLogger(__name__)


def build_symmetrized(A, *, Bk):
    xp = Bk.xp
    sp = Bk.sp
    n = A.shape[0]
    Asym = sp.bmat([[None, A.T], [A, None]], format="csr")
    return Asym


def main():
    parser = argparse.ArgumentParser(
        description="Deflated MINRES on a symmetrized nonsymmetric system"
    )
    parser.add_argument("--n", type=int, default=64, help="matrix dimension")
    parser.add_argument(
        "--nnz-per-row", type=int, default=4, help="nonzeros per row in A"
    )
    parser.add_argument(
        "--k", type=int, default=4, help="number of eigenpairs/deflation vectors"
    )
    parser.add_argument("--backend", choices=["numpy", "cupy"], default=None, help="array backend")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--power-iters", type=int, default=5, help="block power iterations")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp64"],
        default="fp32",
        help="floating point precision",
    )
    parser.add_argument(
        "--inner-rtol", type=float, default=1e-6, help="tolerance for inner MINRES solves"
    )
    parser.add_argument(
        "--inner-maxiter", type=int, default=50, help="max iterations for inner MINRES solves"
    )
    parser.add_argument(
        "--minres-tol", type=float, default=1e-8, help="tolerance for final MINRES solve"
    )
    parser.add_argument(
        "--minres-maxiter", type=int, default=None, help="max iterations for final MINRES solve"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="path to log file (default: nonsym_minres_<timestamp>.log)",
    )
    args = parser.parse_args()

    if args.log_file is None:
        args.log_file = f"nonsym_minres_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.log_file),
        ],
    )
    logger.info("logging to %s", args.log_file)

    dtype = np.float32 if args.dtype == "fp32" else np.float64

    A = random_nonsymmetric_matrix(
        args.n,
        args.nnz_per_row,
        seed=args.seed,
        backend=args.backend,
        dtype=dtype,
    )
    Bk = _detect_backend(A, prefer=args.backend)
    xp = Bk.xp
    sp = Bk.sp

    rng = np.random.default_rng(args.seed)
    x_true = xp.asarray(rng.standard_normal(args.n), dtype=dtype)
    b = A @ x_true

    Asym = build_symmetrized(A, Bk=Bk)
    n = args.n

    rhs = xp.zeros(2 * n, dtype=A.dtype)
    rhs[n:] = b

    V0 = xp.asarray(rng.standard_normal((2 * n, args.k)), dtype=dtype)
    V0, _ = xp.linalg.qr(V0)

    Asym_abs = Asym.copy()
    Asym_abs.data = xp.abs(Asym_abs.data)
    row_sums = Asym_abs.sum(axis=1)
    row_sums = xp.asarray(row_sums).ravel()
    Anorm = float(row_sums.max())

    def power_cb(V, w, _):
        AV = Asym @ V
        R = AV - V * w[xp.newaxis, :]
        norms = xp.sqrt(xp.sum(xp.abs(R) ** 2, axis=0))
        denom = Anorm * xp.sqrt(xp.sum(xp.abs(V) ** 2, axis=0))
        be = norms / denom
        pos = w[w > 0]
        neg = w[w < 0]
        if pos.size:
            max_pos = float(pos[xp.argmax(xp.abs(pos))])
            min_pos = float(pos[xp.argmin(xp.abs(pos))])
        else:
            max_pos = min_pos = float("nan")
        if neg.size:
            max_neg = float(neg[xp.argmin(neg)])
            min_neg = float(neg[xp.argmax(neg)])
        else:
            max_neg = min_neg = float("nan")
        logger.info(
            "eig max+ %.3e min+ %.3e max- %.3e min- %.3e",
            max_pos,
            min_pos,
            max_neg,
            min_neg,
        )
        thresh = 1e-8
        converged = be < thresh
        num = int(xp.count_nonzero(converged))
        if num < len(be):
            next_be = float(be[~converged].min())
            logger.info("%d eigenpairs converged, next backward err %.3e", num, next_be)
        else:
            logger.info("all %d eigenpairs converged", num)

    inv_cb = make_minres_callback(
        rtol=args.inner_rtol, maxiter=args.inner_maxiter, backend=args.backend
    )
    w, V = power_solve_cb(
        Asym,
        V0,
        inv_cb,
        outer_iter=args.power_iters,
        backend=args.backend,
        callback=power_cb,
    )

    solver = DeflatedMinres(Asym, V, backend=args.backend)

    def solve_cb(xh):
        xh_top = xh[:n]
        relerr = xp.max(
            xp.abs(x_true - xh_top) / xp.maximum(1e-14, xp.abs(x_true))
        )
        relres = float(xp.linalg.norm(b - A @ xh_top) / xp.linalg.norm(b))
        logger.info("rel err %.3e rel resid %.3e", float(relerr), relres)

    x_sol, info = solver.solve(
        rhs, tol=args.minres_tol, maxiter=args.minres_maxiter, callback=solve_cb
    )
    logger.info("MINRES info: %s", info)


if __name__ == "__main__":
    main()
