import logging
import os
import sys
import numpy as np
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from util import random_expander_like_matrix, power_solve_cb, make_minres_callback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_spd_matrix(n: int, nnz_per_row: int, seed: int = 0):
    A = random_expander_like_matrix(n, nnz_per_row, seed=seed, backend="numpy")
    rng = np.random.default_rng(seed)
    A.data = rng.uniform(-1.0, 1.0, size=A.nnz)
    A = (A + A.T).tocsr() * 0.5
    A = A + sp.eye(n, format="csr", dtype=A.dtype)
    A = A.T @ A
    return A

def main():
    n = 64
    k = 4
    A = make_spd_matrix(n, 4, seed=1)
    V = np.linalg.qr(np.random.standard_normal((n, k)))[0]

    def cb(V, w, relres):
        thresh = 1e-6
        converged = relres < thresh
        num = int(np.count_nonzero(converged))
        if num < len(relres):
            next_rel = float(relres[~converged].min())
            logger.info("%d eigenvalues converged, next relres %.3e", num, next_rel)
        else:
            logger.info("all %d eigenvalues converged", num)

    inv_cb = make_minres_callback(rtol=1e-4, maxiter=50, backend="numpy")
    power_solve_cb(A, V, inv_cb, outer_iter=5, backend="numpy", callback=cb)

if __name__ == "__main__":
    main()
