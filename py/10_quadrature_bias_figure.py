"""Frequency bias of the three-point quadrature loss vs omega*h, with the
sampling steps of the 8 spin-boson couplings marked.

For data rotating at omega, a rule with weights (a, b, a) puts the loss optimum at
    omega_fit = 2 sin(th) / (h (2 a cos(th) + b)),   th = omega h.
Simpson: a = 1/3, b = 4/3. Exponentially fitted: a, b exact for omega0.

Run from the repo root:  python3 py/10_quadrature_bias_figure.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sid_tools import EVOL_DATA, GAMMAS, Q_of, read_timeevolution

OUT = "py/out"


def omega_fit(omega, h, a, b):
    th = omega * h
    return 2 * np.sin(th) / (h * (2 * a * np.cos(th) + b))


def expfit_weights(theta):
    a = (1 - np.sin(theta) / theta) / (1 - np.cos(theta))
    return a, 2 - 2 * a


def main():
    os.makedirs(OUT, exist_ok=True)
    th = np.logspace(-2.5, np.log10(0.8), 300)
    simpson = 1 - omega_fit(1.0, th, 1 / 3, 4 / 3)
    # exponentially fitted rule with omega0 taken from a Simpson fit (as in the refit)
    w0 = omega_fit(1.0, th, 1 / 3, 4 / 3)
    ef = np.array([abs(1 - omega_fit(1.0, x, *expfit_weights(w * x))) for x, w in zip(th, w0)])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(th, simpson, color="#2a78d6", lw=1.6, label=r"Simpson, $\approx(\omega h)^4/180$")
    keep = ef > 1e-13  # below this the curve is round-off
    ax.loglog(th[keep], ef[keep], color="#eb6834", lw=1.6, ls="--",
              label=r"exp.-fitted, $\omega_0$ from Simpson fit")
    for g in GAMMAS[:4]:
        t, _ = read_timeevolution(EVOL_DATA, "B3", g)
        x = 8 * np.pi * np.median(np.diff(t))
        ax.plot(x, 1 - omega_fit(1.0, x, 1 / 3, 4 / 3), "o", color="#0b0b0b", ms=5, zorder=3)
        ax.annotate(f"Q = {Q_of(g):.3g}", (x, 1 - omega_fit(1.0, x, 1 / 3, 4 / 3)),
                    textcoords="offset points", xytext=(-8, 7), fontsize=7, ha="right")
    ax.set_xlabel(r"$\omega h$ (rad per sample)")
    ax.set_ylabel(r"relative frequency error $|\omega_{\rm fit}-\omega|/\omega$")
    ax.set_ylim(1e-13, 1e-2)
    ax.grid(True, which="major", color="#e5e5e5", lw=0.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/quadrature_frequency_bias.pdf")
    fig.savefig(f"{OUT}/quadrature_frequency_bias.png", dpi=150)
    for g in GAMMAS:
        t, _ = read_timeevolution(EVOL_DATA, "B3", g)
        h = np.median(np.diff(t))
        print(f"{g:9s} Q={Q_of(g):8.3f} h={h:.3g} omega*h={8*np.pi*h:.4g} "
              f"simpson rel.err={1-omega_fit(1.0, 8*np.pi*h, 1/3, 4/3):.2e}")


if __name__ == "__main__":
    main()
