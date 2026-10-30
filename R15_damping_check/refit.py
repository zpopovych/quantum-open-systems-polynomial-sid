"""Python re-implementation of the Lindblad-4 ansatz fit (01_E_Lindblad_CONSTR_train-test.ipynb)
with a choice of quadrature rule, solved by local least squares from a given start point.

Simpson (the rule used in the paper):
    rho[i] - rho[i-2] = h * L((rho[i-2] + 4 rho[i-1] + rho[i]) / 3)
Boole (next closed Newton-Cotes rule, still linear in L so the objective stays a quartic polynomial):
    rho[i] - rho[i-4] = (2h/45) * L(7 rho[i-4] + 32 rho[i-3] + 12 rho[i-2] + 32 rho[i-1] + 7 rho[i])
"""
import numpy as np
from scipy.optimize import least_squares

from r15_common import BASIS, SX, SY, SZ, liouvillian, read_evolution

RULES = {
    "simpson": (2, np.array([1, 4, 1]) / 3),
    "boole": (4, 2 * np.array([7, 32, 12, 32, 7]) / 45),
}


def lindblad4(x):
    eps, a, b, r, p = x
    H = 0.5 * np.array([[eps, a + 1j * b], [a - 1j * b, -eps]])
    J1 = np.array([[0, r], [0, 0]], dtype=complex)
    return H, [J1, p * SX, p * SY, p * SZ]


def _stencils(gamma, rule, states=BASIS):
    span, wts = RULES[rule]
    lhs, rhs = [], []
    for s in states:
        t, rho = read_evolution(s, gamma)
        h = t[1] - t[0]
        v = rho.reshape(len(t), 4)
        n = len(t) - span
        lhs.append(v[span:] - v[:n])
        rhs.append(h * sum(w * v[k:k + n] for k, w in enumerate(wts)))
    return np.vstack(lhs), np.vstack(rhs)


def fit(gamma, x0, rule="simpson"):
    lhs, rhs = _stencils(gamma, rule)

    def resid(x):
        R = lhs - rhs @ liouvillian(*lindblad4(x)).T
        return np.concatenate([R.real.ravel(), R.imag.ravel()])

    sol = least_squares(resid, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return sol.x, 0.5 * np.sum(sol.fun ** 2)


def quadrature_frequency(w, gamma, h, rule):
    """Frequency a noiseless damped mode exp(-(i w + gamma/2) t) is fitted with by the rule."""
    span, wts = RULES[rule]
    z = np.exp((-1j * w - gamma / 2) * h)
    lam = (z ** span - 1) / (h * sum(c * z ** k for k, c in enumerate(wts)))
    return -lam.imag
