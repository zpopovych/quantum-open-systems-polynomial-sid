"""Refit the Lindblad ansatz at Q = 316 with an exponentially fitted quadrature
rule in place of Simpson's rule (Python check of the Julia/TSSOS refit).

The loss in the notebooks is the double-step rule
    rho_{j+1} - rho_{j-1} = h * L[a rho_{j-1} + b rho_j + a rho_{j+1}]
with Simpson weights a = 1/3, b = 4/3. Simpson is exact for polynomials up to
cubic, but the data are oscillations e^{-i w t} with w h = 0.5 at Q = 316, and
the fitted frequency comes out low by ~ w (w h)^4 / 180.

Exponential fitting keeps the same three-point form and picks a, b so that the
rule is exact for 1 and e^{+-i w0 t}:
    2a + b = 2,   2a cos(th) + b = 2 sin(th) / th,   th = w0 h
    =>  a = (1 - sin(th)/th) / (1 - cos(th)),   b = 2 - 2a
As th -> 0 this gives a -> 1/3, b -> 4/3 (Simpson). The weights are numbers,
so the loss is still a quartic polynomial in (H, J) and TSSOS applies unchanged.

Here the loss is minimised with scipy least squares, started from the saved
TSSOS model, so this is a local check of the global TSSOS result.

Run from the repo root:  python3 py/09_expfit_refit_Q316.py
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from sid_tools import (ALL_STATES, EVOL_DATA, liouvillian, read_model,
                       read_timeevolution, trace_distance, evolve_L)

GAMMA = "0.079477"  # Q = 316
TRAIN = ["B1", "B2", "B3", "B4"]


def simpson_weights(theta=None):
    return 1 / 3, 4 / 3


def expfit_weights(theta):
    """Three-point weights exact for 1, cos(w0 t), sin(w0 t) on [-h, h]."""
    if theta < 1e-3:  # series, avoids 0/0
        return 1 / 3 + theta**2 / 90, 4 / 3 - theta**2 / 45
    a = (1 - np.sin(theta) / theta) / (1 - np.cos(theta))
    return a, 2 - 2 * a


def unpack(x):
    """Lindblad ansatz of 01_E_Lindblad_CONSTR_train-test.ipynb, 5 real parameters:
    H = [[e, al + i be], [al - i be, -e]] / 2,  J1 = r |0><1|,  J2,3,4 = p * (sx, sy, sz)."""
    e, al, be, r, p = x
    H = np.array([[e, al + 1j * be], [al - 1j * be, -e]]) / 2
    J = [np.array([[0, r], [0, 0]], dtype=complex),
         p * np.array([[0, 1], [1, 0]], dtype=complex),
         p * np.array([[0, -1j], [1j, 0]]),
         p * np.array([[1, 0], [0, -1]], dtype=complex)]
    return H, J


def pack(H, J):
    """Inverse of unpack for a saved model (J as written by the notebook)."""
    return np.array([2 * H[0, 0].real, 2 * H[0, 1].real, 2 * H[0, 1].imag,
                     J[0][0, 1].real, J[3][0, 0].real])


def residuals(x, data, a, b):
    H, J = unpack(x)
    L = liouvillian(H, J)
    out = []
    for t, rho in data:
        v = rho.reshape(len(t), -1, order="F")    # vec(rho_k), column-stacked
        h = t[1:-1] - t[:-2]                       # step, as (t[i]-t[i-1]) in kossak_obj
        lhs = v[2:] - v[:-2]
        rhs = h[:, None] * ((a * v[:-2] + b * v[1:-1] + a * v[2:]) @ L.T)
        r = (lhs - rhs).ravel()
        out += [r.real, r.imag]
    return np.concatenate(out)


def fit(x0, data, a, b):
    sol = least_squares(residuals, x0, args=(data, a, b), method="lm",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return sol.x


def score(x):
    H, J = unpack(x)
    L = liouvillian(H, J)
    td_mean, td_max = [], []
    for s in ALL_STATES:
        t, rho = read_timeevolution(EVOL_DATA, s, GAMMA)
        td = trace_distance(rho, evolve_L(L, rho[0], t))
        td_mean.append(td.mean()); td_max.append(td.max())
    return np.mean(td_mean), np.max(td_max)


def main():
    data = [read_timeevolution(EVOL_DATA, s, GAMMA) for s in TRAIN]
    h = np.median(np.diff(data[0][0]))
    H0, J0, status = read_model("lindblad", GAMMA)
    x_tssos = pack(H0, J0)
    print(f"Q = 316, h = {h},  saved TSSOS Lindblad fit ({status}): "
          f"eps = {x_tssos[0]:.5f}, mean/max TD = %.5f / %.4f" % score(x_tssos))

    x_simp = fit(x_tssos, data, *simpson_weights())
    print(f"Simpson refit (local):        eps = {x_simp[0]:.5f}, mean/max TD = %.5f / %.4f"
          % score(x_simp))

    w0 = x_simp[0]                     # frequency taken from the Simpson fit, as in the Julia run
    a, b = expfit_weights(w0 * h)
    print(f"exp-fitted weights at w0 h = {w0 * h:.4f}: a = {a:.6f}, b = {b:.6f} "
          f"(Simpson: 0.333333, 1.333333)")
    x_exp = fit(x_simp, data, a, b)
    print(f"Exp-fitted refit (local):     eps = {x_exp[0]:.5f}, mean/max TD = %.5f / %.4f"
          % score(x_exp))
    print("Kurt's Lamb-shifted frequency: 25.13513")


if __name__ == "__main__":
    main()
