# coding: utf-8
from __future__ import annotations

from typing import Callable, Optional, Tuple

# Optional CuPy import (only if available)
try:
    import cupy as _cp
    import cupyx.scipy.sparse as _cpxs
    from cupyx.scipy.sparse.linalg import LinearOperator as _CuLinearOperator, minres as _cu_minres
    from cupyx.scipy.linalg import solve_triangular as _cu_solve_tri
    _CUPY_AVAILABLE = True
except Exception:
    _cp = None
    _cpxs = None
    _CuLinearOperator = None
    _cu_minres = None
    _cu_solve_tri = None
    _CUPY_AVAILABLE = False

import numpy as _np
import scipy.sparse as _sps
from scipy.sparse.linalg import LinearOperator as _NpLinearOperator, minres as _np_minres
from scipy.linalg import solve_triangular as _np_solve_tri
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------ Backend plumbing ------------------------

class _Backend:
    """Holds function/module handles for a specific array backend."""
    def __init__(self, name: str):
        if name == "cupy":
            if not _CUPY_AVAILABLE:
                raise RuntimeError("CuPy backend requested but CuPy is not available.")
            self.name = "cupy"
            self.xp = _cp
            self.sp = _cpxs
            self.LinearOperator = _CuLinearOperator
            self.minres = _cu_minres
            self.solve_tri = _cu_solve_tri
            self.asfortranarray = _cp.asfortranarray
            self.copyto = _cp.copyto
        elif name == "numpy":
            self.name = "numpy"
            self.xp = _np
            self.sp = _sps
            self.LinearOperator = _NpLinearOperator
            self.minres = _np_minres
            self.solve_tri = _np_solve_tri
            self.asfortranarray = _np.asfortranarray
            self.copyto = _np.copyto
        else:
            raise ValueError("backend name must be 'numpy' or 'cupy'")

def _detect_backend(*arrays, prefer: str | None = None) -> _Backend:
    """
    Choose backend from the first array that looks like a CuPy or NumPy array,
    unless `prefer` is explicitly given ('numpy' or 'cupy').
    """
    if prefer in ("numpy", "cupy"):
        return _Backend(prefer)

    for a in arrays:
        if a is None:
            continue
        # crude but reliable detection
        if _CUPY_AVAILABLE and isinstance(a, _cp.ndarray):
            return _Backend("cupy")
        if isinstance(a, _np.ndarray):
            return _Backend("numpy")

    # default: prefer cupy if available, else numpy
    return _Backend("cupy" if _CUPY_AVAILABLE else "numpy")


# ------------------------ Helper / utility ------------------------

def _to_xp_array(x, xp):
    if x is None:
        return None
    # move between host/device as needed
    if _CUPY_AVAILABLE and xp is _cp:
        return _cp.asarray(x)
    else:
        return _np.asarray(x)


# ------------------------ Public API ------------------------



def random_expander_like_matrix(
    n: int,
    nnz_per_row: int,
    seed: int | None = None,
    *,
    backend: str | None = None,
    dtype=None,
):
    """
    Create an n x n sparse matrix A with:
      - `nnz_per_row` randomly sampled off-diagonal entries per row (no replacement)
      - symmetrized as (A + A.T)
      - all nonzero values set to -1.0
    Works with NumPy/SciPy and CuPy/cupyx. Select with `backend='numpy'|'cupy'` or auto-detect.

    Returns
    -------
    csr_matrix (scipy.sparse.csr_matrix or cupyx.scipy.sparse.csr_matrix)
    """
    if not (0 <= nnz_per_row < n):
        raise ValueError("Require 0 <= nnz_per_row < n (must exclude diagonal).")

    Bk = _Backend(backend or ("cupy" if _CUPY_AVAILABLE else "numpy"))
    xp = Bk.xp
    sp = Bk.sp

    rng = _np.random.default_rng(seed)

    rows = _np.repeat(_np.arange(n), nnz_per_row)
    samples = _np.empty((n, nnz_per_row), dtype=_np.int64)
    base = _np.arange(n - 1)
    for i in range(n):
        cols = rng.choice(base, size=nnz_per_row, replace=False)
        cols = cols + (cols >= i)  # skip diagonal
        samples[i, :] = cols
    cols = samples.ravel()

    rows_x = xp.asarray(rows, dtype=xp.int64)
    cols_x = xp.asarray(cols, dtype=xp.int64)
    if dtype is None:
        dtype = xp.float32

    data_x = xp.full(rows_x.shape, -1.0, dtype=dtype)
    A = sp.coo_matrix((data_x, (rows_x, cols_x)), shape=(n, n)).tocsr()
    A.sum_duplicates()

    B = (A + A.T).tocsr()
    B.sum_duplicates()
    # Set all nonzeros to -1.0 (preserve sparsity pattern)
    if B.nnz:
        B.data[...] = xp.array(-1.0, dtype=dtype)

    # Force zero diagonal (should be zero already)
    d = B.diagonal()
    if xp.any(d != 0):
        B = B - sp.diags(d, offsets=0, shape=B.shape, dtype=B.dtype)

    return B


def block_chebyshev(
    A, B,
    eig_min: float, eig_max: float, maxiter: int,
    *, backend: str | None = None
):
    """
    Block Chebyshev iteration (SPD A, spectrum in [eig_min, eig_max]).
    `A` can be a dense/sparse array with @, or a LinearOperator with matvec/matmat.
    Works for NumPy or CuPy based on backend or B's type.
    """
    Bk = _detect_backend(B, prefer=backend)
    xp = Bk.xp

    theta = 0.5 * (eig_max + eig_min)
    delta = 0.5 * (eig_max - eig_min)
    sigma1 = theta / delta

    X = xp.zeros_like(B)
    R = B - (A @ X if hasattr(A, "__matmul__") else A.matmat(X))

    rho = 1.0 / sigma1
    D = (1.0 / theta) * R

    for _ in range(maxiter):
        X = X + D
        R = R - (A @ D if hasattr(A, "__matmul__") else A.matmat(D))
        rho_next = 1.0 / (2.0 * sigma1 - rho)
        D = rho_next * rho * D + (2.0 * rho_next / delta) * R
        rho = rho_next
    return X


def power_cheb(
    A, V,
    eig_min: float, eig_max: float,
    outer_iter: int = 10, inner_iter: int = 10,
    *, backend: str | None = None, plots = False,
    callback: Optional[Callable[[object, object, object], None]] = None,
) -> Tuple:
    """
    Block power + Chebyshev accelerator with Rayleigh–Ritz each outer step.
    Returns (w, V), with w ascending.
    If `callback` is provided, it is invoked as ``callback(V, w, relres)`` after
    each outer iteration.
    """
    Bk = _detect_backend(V, prefer=backend)
    xp = Bk.xp

    V = _to_xp_array(V, xp)

    iplot = 0
    for _ in range(outer_iter):
        V = block_chebyshev(A, V, eig_min, eig_max, maxiter=inner_iter, backend=Bk.name)
        # Orthonormalize
        Q, _ = xp.linalg.qr(V, mode="reduced")
        V = Q
        # Rayleigh–Ritz
        VAV = V.T @ (A @ V if hasattr(A, "__matmul__") else A.matmat(V))
        w, W = xp.linalg.eigh(VAV)
        V = V @ W

        # Residuals for callback/plots
        if callback is not None or plots:
            AV = (A @ V) if hasattr(A, "__matmul__") else A.matmat(V)
            R = AV - V * w[xp.newaxis, :]
            norms = xp.sqrt(xp.sum(xp.abs(R) ** 2, axis=0))
            residuals = norms / xp.maximum(xp.abs(w), xp.finfo(V.dtype).eps)
        else:
            residuals = None

        if callback is not None:
            callback(V, w, residuals)

        if plots:
            plt.close()
            plt.loglog(w, residuals)
            plt.xlabel("approximate eigenvalues")
            plt.ylabel("eigenvector residual")
            plt.title(f"min(eig)={xp.amin(w)},max(eig)={xp.amax(w)}")
            plt.xlim(1e-8,1e-3)
            plt.ylim(1e-12,1.0)
            plt.savefig(f"plots/{str(iplot).zfill(3)}.svg")
            iplot = iplot + 1

    return w, V



class DeflatedMinres:
    """
    Backend-agnostic deflated MINRES with minimal copying.
    - Supports NumPy/SciPy and CuPy/cupyx.
    Assumptions:
      - A is symmetric.
      - V has orthonormal columns.
      - A supports @ with (n,) and (n,k) arrays or is a LinearOperator with matvec/matmat.
    """

    def __init__(self, A, V, *, backend: str | None = None) -> None:
        Bk = _detect_backend(V, A, prefer=backend)
        self._Bk = Bk
        xp = Bk.xp

        self.A = A
        self.V = Bk.asfortranarray(_to_xp_array(V, xp))
        n, k = self.V.shape
        self.n, self.k = n, k

        AV = self._A_matmat(self.V)
        self.AV = Bk.asfortranarray(AV)

        VAV = self.V.T @ self.AV
        self.L = xp.linalg.cholesky(VAV)

        # Work buffers
        self._buf_Ax = xp.empty((n,), dtype=self.V.dtype, order='C')
        self._buf_t  = xp.empty((k,), dtype=self.V.dtype, order='C')
        self._buf_z  = xp.empty((k,), dtype=self.V.dtype, order='C')
        self._buf_y  = xp.empty((n,), dtype=self.V.dtype, order='C')

        self._buf_tb = xp.empty((k,), dtype=self.V.dtype, order='C')
        self._buf_yb = xp.empty((k,), dtype=self.V.dtype, order='C')
        self._buf_xl = xp.empty((n,), dtype=self.V.dtype, order='C')

        self._buf_Axh = xp.empty((n,), dtype=self.V.dtype, order='C')
        self._buf_txh = xp.empty((k,), dtype=self.V.dtype, order='C')
        self._buf_zxh = xp.empty((k,), dtype=self.V.dtype, order='C')
        self._buf_Vz  = xp.empty((n,), dtype=self.V.dtype, order='C')

    # ---- A application helpers ----
    def _A_matvec(self, x, out=None):
        y = self.A @ x if hasattr(self.A, "__matmul__") else self.A.matvec(x)
        if out is None:
            return y
        self._Bk.copyto(out, y)
        return out

    def _A_matmat(self, X):
        return self.A @ X if hasattr(self.A, "__matmul__") else self.A.matmat(X)

    # ---- Deflated operator (I - A V (VAV)^{-1} V^T) A x ----
    def _apply_deflated(self, x):
        xp = self._Bk.xp

        # 1) Ax -> _buf_Ax
        self._A_matvec(x, out=self._buf_Ax)

        # 2) t = V^T Ax  -> _buf_t
        self._buf_t = self.V.T @ self._buf_Ax

        # 3) z = (VAV)^{-1} t via Cholesky
        y = self._Bk.solve_tri(self.L, self._buf_t, lower=True, overwrite_b=False, check_finite=False)
        z = self._Bk.solve_tri(self.L.T, y, lower=False, overwrite_b=False, check_finite=False)
        self._Bk.copyto(self._buf_z, z)

        # 4) y = Ax - AV @ z  (return a fresh vector)
        y_out = self._buf_Ax.copy()
        y_out -= self.AV @ self._buf_z
        return y_out

    # rhs = (I - A V (VAV)^{-1} V^T) b
    def _deflated_rhs(self, b):
        self._buf_tb = self.V.T @ b
        y = self._Bk.solve_tri(self.L, self._buf_tb, lower=True, overwrite_b=False, check_finite=False)
        y = self._Bk.solve_tri(self.L.T, y, lower=False, overwrite_b=False, check_finite=False)
        self._Bk.copyto(self._buf_yb, y)
        rhs = b - (self.AV @ self._buf_yb)
        return rhs

    # reconstruct x from xh:
    # xl = V y_b, xr = xh - V z_xh with z_xh = (VAV)^{-1} V^T A xh
    def _reconstruct(self, b, xh):
        xp = self._Bk.xp
        self._Bk.copyto(self._buf_xl, self.V @ self._buf_yb)  # V y_b

        self._A_matvec(xh, out=self._buf_Axh)         # A xh
        self._buf_txh = self.V.T @ self._buf_Axh      # V^T A xh
        y = self._Bk.solve_tri(self.L, self._buf_txh, lower=True, overwrite_b=False, check_finite=False)
        y = self._Bk.solve_tri(self.L.T, y, lower=False, overwrite_b=False, check_finite=False)
        self._Bk.copyto(self._buf_zxh, y)

        self._Bk.copyto(self._buf_Vz, self.V @ self._buf_zxh)
        x = xh - self._buf_Vz
        x += self._buf_xl
        return x

    def solve(
        self,
        b,
        *,
        tol: float = 1e-12,
        maxiter: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> Tuple:
        Bk = self._Bk
        xp = Bk.xp
        n = b.shape[0]
        dtype = getattr(self.A, "dtype", None) or b.dtype

        def mv(x):
            return self._apply_deflated(x)

        op = Bk.LinearOperator((n, n), matvec=mv, dtype=dtype)

        rhs = self._deflated_rhs(b)

        def cb(xh):
            if callback is not None:
                callback(self._reconstruct(b, xh))

        xh, info = Bk.minres(op, rhs, tol=tol, maxiter=maxiter, callback=cb)
        x = self._reconstruct(b, xh)
        return x, info

class MinresMultiRHS:
    """
    Multiple independent MINRES solves advanced in lockstep, with a single SPMM per
    iteration.  Solves (A - shift_i * I) x_i = b_i for i=1..m, without coupling
    between RHSs (this is NOT block MINRES).

    - A can be a sparse/dense array supporting '@' with (n,k), or a LinearOperator
      with 'matmat'. If only 'matvec' exists, we fall back to columnwise matvecs
      (loses the single-SPMM benefit).
    - No preconditioner in this minimal version.
    - Global convergence: stops when max_i ||r_i|| / ||b_i|| <= rtol, or maxiter.

    Parameters
    ----------
    A : matrix-like or LinearOperator
      Symmetric (indefinite ok). For shifts, solves (A - shift_i I) x_i = b_i.
    backend : {"numpy","cupy"} or None
      Force backend; otherwise auto-detect from B or A.

    Usage
    -----
    solver = MinresMultiRHS(A, backend="cupy")
    X, info = solver.solve(B, shifts=[0.0, 0.5, ...], rtol=1e-6, maxiter=200)

    Returns
    -------
    X : (n, m) array
    info : dict { "it": int, "converged": bool, "final_relres": float, "relres_per_rhs": array }
    """

    def __init__(self, A, *, backend: str | None = None):
        self.A = A
        self._Bk = _detect_backend(getattr(A, "dtype", None), prefer=backend)

    # ---- A application helpers ----
    def _A_matvec(self, x, out=None):
        A = self.A
        y = (A @ x) if hasattr(A, "__matmul__") else A.matvec(x)
        if out is None:
            return y
        self._Bk.copyto(out, y)
        return out

    def _A_matmat(self, X):
        A = self.A
        if hasattr(A, "__matmul__"):
            # prefer '@' (SciPy/CuPy sparse supports dense matmat)
            return A @ X
        elif hasattr(A, "matmat"):
            return A.matmat(X)
        else:
            # fallback: columnwise matvec (no single-SPMM, but keeps correctness)
            cols = [self._A_matvec(X[:, i]) for i in range(X.shape[1])]
            return self._Bk.xp.stack(cols, axis=1)

    def solve(
        self,
        B,
        shifts=None,
        *,
        rtol: float = 1e-8,
        maxiter: int | None = None,
        callback: Optional[Callable[[object], None]] = None,  # gets X each iter
    ):
        Bk = _detect_backend(B, self.A, prefer=self._Bk.name)
        xp = Bk.xp

        # --- shape & inputs ---
        B = _to_xp_array(B, xp)
        if B.ndim == 1:
            B = B[:, None]
        n, m = B.shape
        if shifts is None:
            shifts = xp.zeros((m,), dtype=B.dtype)
        else:
            shifts = _to_xp_array(shifts, xp).reshape(-1)
            if shifts.size == 1 and m > 1:
                shifts = xp.full((m,), shifts.item(), dtype=B.dtype)
            if shifts.size != m:
                raise ValueError("len(shifts) must equal number of RHS columns in B")

        # --- maxiter default ---
        if maxiter is None:
            maxiter = 5 * n

        # --- Allocate work ---
        X   = xp.zeros((n, m), dtype=B.dtype)
        w   = xp.zeros((n, m), dtype=B.dtype)
        w2  = xp.zeros((n, m), dtype=B.dtype)
        w1  = xp.zeros((n, m), dtype=B.dtype)

        r1  = B.copy()
        r2  = r1.copy()
        y   = r2.copy()  # preconditioner is identity here

        # Scalars as length-m arrays
        eps = xp.finfo(B.dtype).eps
        oldb   = xp.zeros((m,), dtype=B.dtype)
        beta   = xp.sqrt(xp.maximum(xp.sum(xp.conjugate(r2) * y, axis=0).real, 0))
        dbar   = xp.zeros((m,), dtype=B.dtype)
        epsln  = xp.zeros((m,), dtype=B.dtype)
        phibar = beta.copy()
        cs     = -xp.ones((m,), dtype=B.dtype)
        sn     = xp.zeros((m,), dtype=B.dtype)

        bnorm  = xp.maximum(xp.sqrt(xp.maximum(xp.sum(xp.abs(B)**2, axis=0), 0)), eps)
        relres = xp.full((m,), xp.inf, dtype=B.dtype)

        # mask of "active" systems (beta>0); keep everyone in loop but avoid div-by-0
        active = beta > 0

        # --- Iterate ---
        it = 0
        while it < maxiter:
            it += 1

            # v = y / beta   (safe: inactive cols get v=0)
            s = xp.where(beta > 0, 1.0 / beta, 0.0)
            v = y * s[None, :]

            # Y = A @ v   (single SPMM)  then apply per-RHS shift
            Y = self._A_matmat(v)
            Y = Y - v * shifts[None, :]

            # Y -= (beta/oldb) * r1   for k>=2; where oldb>0
            if it >= 2:
                fac = xp.where(oldb > 0, beta / xp.where(oldb == 0, 1.0, oldb), 0.0)
                Y = Y - r1 * fac[None, :]

            # alfa = <v, Y>
            alfa = xp.sum(xp.conjugate(v) * Y, axis=0).real

            # Y -= (alfa/beta) * r2
            ab = xp.where(beta > 0, alfa / beta, 0.0)
            Y = Y - r2 * ab[None, :]

            # r1 <- r2 ; r2 <- Y ; y <- r2 (identity preconditioner)
            r1, r2 = r2, Y
            y = r2

            # shift oldb/beta & update norms
            oldb = beta
            beta_sq = xp.maximum(xp.sum(xp.conjugate(r2) * y, axis=0).real, 0)
            beta = xp.sqrt(beta_sq)

            # ----- MINRES orthogonal transformations (vectorized per column) -----
            oldeps = epsln
            delta  = cs * dbar + sn * alfa
            gbar   = sn * dbar - cs * alfa
            epsln  = sn * beta
            dbar   = -cs * beta

            # Compute next plane rotation
            gamma  = xp.sqrt(xp.maximum(gbar * gbar + beta * beta, 0))
            gamma  = xp.where(gamma > 0, gamma, eps)
            cs     = gbar / gamma
            sn     = beta / gamma
            phi    = cs * phibar
            phibar = sn * phibar

            # Update directions and solution
            denom = 1.0 / gamma
            w1, w2 = w2, w
            w = (v - oldeps[None, :] * w1 - delta[None, :] * w2) * denom[None, :]
            X = X + w * phi[None, :]

            # --- Global convergence (optional): use MINRES residual proxy |phibar|
            relres = xp.abs(phibar) / bnorm
            glob = float(xp.max(relres).item())
            if callback is not None:
                # Pass a view of X; caller can copy if they want to stash it
                callback(X)

            if glob <= rtol:
                break

        info = {
            "it": it,
            "converged": bool(glob <= rtol),
            "final_relres": float(glob),
            "relres_per_rhs": relres,
        }
        return X.squeeze() if X.shape[1] == 1 else X, info




from typing import Callable, Tuple, Optional

ApplyInverseCB = Callable[[object, object], object]  # (A, V) -> X ≈ A^{-1} V

def power_solve_cb(
    A, V,
    apply_inverse_cb: ApplyInverseCB,
    *,
    outer_iter: int = 10,
    backend: str | None = None,
    plots: bool = False,
    callback: Optional[Callable[[object, object, object], None]] = None,
) -> Tuple[object, object]:
    """
    Block power + user-supplied inverse callback with Rayleigh–Ritz each outer it.
    `apply_inverse_cb(A, V)` must return X ≈ A^{-1} V (same shape as V).
    If `callback` is given, it receives ``(V, w, relres)`` after each outer
    iteration.
    """
    Bk = _detect_backend(V, A, prefer=backend)
    xp = Bk.xp
    V = _to_xp_array(V, xp)

    iplot = 0
    for _ in range(outer_iter):
        # Inverse-like step: X ≈ A^{-1} V  (caller controls method, tolerance, etc.)
        V = apply_inverse_cb(A, V)

        # Orthonormalize
        Q, _ = xp.linalg.qr(V, mode="reduced")
        V = Q

        # Rayleigh–Ritz
        AV = (A @ V) if hasattr(A, "__matmul__") else A.matmat(V)
        VAV = V.T @ AV
        w, W = xp.linalg.eigh(VAV)
        V = V @ W

        # Residuals for callback/plots
        if callback is not None or plots:
            AV = (A @ V) if hasattr(A, "__matmul__") else A.matmat(V)
            R = AV - V * w[xp.newaxis, :]
            norms = xp.sqrt(xp.sum(xp.abs(R) ** 2, axis=0))
            absw = xp.abs(w)
            # Guard near-zero Ritz values so we don't blow up; scale by the spectrum size.
            absw_floor = xp.finfo(V.dtype).eps * xp.maximum(1.0, xp.max(absw))
            den = xp.maximum(absw, absw_floor)
            residuals = norms / den
        else:
            residuals = None

        if callback is not None:
            callback(V, w, residuals)

        if plots:
            plt.close()
            plt.loglog(w, residuals)
            plt.xlabel("approximate eigenvalues")
            plt.ylabel("eigenvector residual")
            plt.xlim(1e-9,1e-3)
            plt.ylim(1e-12,1.0)
            plt.title(f"min(eig)={float(xp.amin(w))}, max(eig)={float(xp.amax(w))}")
            plt.savefig(f"plots/{str(iplot).zfill(3)}.svg")
            iplot += 1

    return w, V


# ---- Convenience callback factories (optional) ----

def make_cheb_callback(eig_min: float, eig_max: float, iters: int, *, backend: str | None = None) -> ApplyInverseCB:
    """
    Returns a callback (A, V) -> X using Chebyshev iteration over [eig_min, eig_max].
    """
    def _cb(A, V):
        return block_chebyshev(A, V, eig_min=eig_min, eig_max=eig_max, maxiter=iters, backend=backend)
    return _cb


def make_minres_callback(rtol: float = 1e-6, maxiter: int | None = None, *, backend: str | None = None) -> ApplyInverseCB:
    """
    Returns a callback (A, V) -> X using synchronized multi-RHS MINRES (MinresMultiRHS).
    """
    def _cb(A, V):
        solver = MinresMultiRHS(A, backend=backend)
        X, _info = solver.solve(V, shifts=None, rtol=rtol, maxiter=maxiter)
        return X
    return _cb

# ------------------------ Tiny usage notes ------------------------
# CPU (NumPy/SciPy):
#   A = random_expander_like_matrix(1024, 4, seed=0, backend='numpy')
#   V = _np.linalg.qr(_np.random.randn(1024, 16))[0]
#   solver = DeflatedMinres(A, V, backend='numpy')
#   b = _np.random.randn(1024)
#   x, info = solver.solve(b)
#
# GPU (CuPy):
#   A = random_expander_like_matrix(1024, 4, seed=0, backend='cupy')
#   V = _cp.linalg.qr(_cp.random.standard_normal((1024,16)))[0]
#   solver = DeflatedMinres(A, V, backend='cupy')
#   b = _cp.random.standard_normal(1024)
#   x, info = solver.solve(b)

