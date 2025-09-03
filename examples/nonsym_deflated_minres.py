import argparse
import logging
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from util import (
    random_nonsymmetric_matrix,
    power_solve_cb,
    make_minres_callback,
    DeflatedMinres,
    _detect_backend,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_symmetrized(A, *, Bk):
    xp = Bk.xp
    sp = Bk.sp
    n = A.shape[0]
    Asym = sp.bmat([[None, A.T], [A, None]], format="csr")
    return Asym


def main():
    parser = argparse.ArgumentParser(description="Deflated MINRES on a symmetrized nonsymmetric system")
    parser.add_argument("n", type=int, nargs="?", default=64, help="matrix dimension")
    parser.add_argument("nnz_per_row", type=int, nargs="?", default=4, help="nonzeros per row in A")
    parser.add_argument("k", type=int, nargs="?", default=4, help="number of eigenpairs/deflation vectors")
    parser.add_argument("--backend", choices=["numpy", "cupy"], default=None, help="array backend")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--power-iters", type=int, default=5, help="block power iterations")
    args = parser.parse_args()

    A = random_nonsymmetric_matrix(
        args.n, args.nnz_per_row, seed=args.seed, backend=args.backend
    )
    Bk = _detect_backend(A, prefer=args.backend)
    xp = Bk.xp
    sp = Bk.sp

    rng = np.random.default_rng(args.seed)
    x_true = xp.asarray(rng.standard_normal(args.n))
    b = A @ x_true

    Asym = build_symmetrized(A, Bk=Bk)
    n = args.n

    rhs = xp.zeros(2 * n, dtype=A.dtype)
    rhs[n:] = b

    V0 = xp.asarray(rng.standard_normal((2 * n, args.k)))
    V0, _ = xp.linalg.qr(V0)

    Anorm = float(xp.abs(Asym).sum(axis=1).max())

    def power_cb(V, w, _):
        AV = Asym @ V
        R = AV - V * w[xp.newaxis, :]
        norms = xp.sqrt(xp.sum(xp.abs(R) ** 2, axis=0))
        denom = Anorm * xp.sqrt(xp.sum(xp.abs(V) ** 2, axis=0))
        be = norms / denom
        thresh = 1e-8
        converged = be < thresh
        num = int(xp.count_nonzero(converged))
        if num < len(be):
            next_be = float(be[~converged].min())
            logger.info("%d eigenpairs converged, next backward err %.3e", num, next_be)
        else:
            logger.info("all %d eigenpairs converged", num)

    inv_cb = make_minres_callback(rtol=1e-6, backend=args.backend)
    w, V = power_solve_cb(
        Asym,
        V0,
        inv_cb,
        outer_iter=args.power_iters,
        backend=args.backend,
        callback=power_cb,
    )

    mask = w > 0
    if not xp.any(mask):
        raise RuntimeError("No positive eigenvalues found for deflation")
    V = V[:, mask]

    solver = DeflatedMinres(Asym, V, backend=args.backend)

    def solve_cb(xh):
        xh_top = xh[:n]
        relerr = xp.max(xp.abs(x_true - xh_top) / xp.maximum(1e-14, xp.abs(x_true)))
        logger.info("relative error %.3e", float(relerr))

    x_sol, info = solver.solve(rhs, tol=1e-8, callback=solve_cb)
    logger.info("MINRES info: %s", info)


if __name__ == "__main__":
    main()
