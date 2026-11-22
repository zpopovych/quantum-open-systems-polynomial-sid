r"""Check of the Breuer-Laine-Piilo (BLP) non-Markovianity numbers in NonMarkovianity.ipynb.

    N = max_{rho1(0), rho2(0)} \int_{sigma>0} sigma dt,   sigma = dD/dt,  D = trace distance

The notebook computes, per pair of the 14 stored initial states, sum(diff(D)[diff(D) > 0]).
That sum is the exact integral of sigma over sigma>0 for the piecewise-linear D(t) through the
samples (no Riemann/Simpson error), so the summation itself is fine. This script checks the
other ingredients:

  1. time window: Julia's zip() truncates each pair to its own shortest series, and the series
     lengths differ between states, so different pairs are integrated over different windows;
  2. maximisation over pairs: only the 91 pairs of the 14 stored states are tried. For a qubit
     the evolution is affine on the Bloch vector, b(t) = A(t) b0 + c(t), so D(t) = |A(t) n| for
     the antipodal pure pair b0 = +-n, and the optimum (Wissmann et al. PRA 86, 062108 (2012))
     is found by maximising over the unit vector n. A(t), c(t) come from the 4 basis states;
  3. noise floor: increments below the data precision (~1e-5 .. 1e-4, see the linearity check)
     are counted as revivals; N_eps ignores wiggles smaller than eps (hysteresis);
  4. grid: the sum is recomputed on every 2nd and 4th sample.

Usage: python py/blp_nonmarkovianity_check.py [DATA/ALL_GAMMAS_B4_D10.h5]
Writes py/out/blp_check.csv and py/out/blp_check.pdf.
"""
import itertools, os, sys
import h5py
import numpy as np
from scipy.optimize import minimize

FILE = sys.argv[1] if len(sys.argv) > 1 else "DATA/ALL_GAMMAS_B4_D10.h5"
GAMMAS = ["0.079477", "0.25133", "0.79477", "2.5133", "7.9477", "25.133", "79.477", "251.33"]
STATES = ["B1", "B2", "B3", "B4"] + [f"D{i}" for i in range(1, 11)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load(f, state, g):
    d = f[state][g]
    t = d["t"][()].ravel()
    p0 = d["p0"][()].ravel()
    c = d["s_re"][()].ravel() + 1j * d["s_im"][()].ravel()
    return t, np.stack([2 * c.real, -2 * c.imag, 2 * p0 - 1], 1)  # Bloch vectors


def trdist(b1, b2):  # qubit: D = |b1 - b2| / 2
    return 0.5 * np.linalg.norm(b1 - b2, axis=1)


def npos(D):  # = integral of sigma over sigma>0 for piecewise-linear D
    d = np.diff(D)
    return d[d > 0].sum()


def npos_eps(D, eps):  # same, but rises/dips smaller than eps are treated as noise
    tot, lo, hi, up = 0.0, D[0], D[0], None
    for x in D[1:]:
        if up is None:
            if x - lo >= eps: up, hi = True, x
            elif x < lo: lo = x
        elif up:
            if x > hi: hi = x
            elif hi - x >= eps: tot += hi - lo; up, lo = False, x
        else:
            if x < lo: lo = x
            elif x - lo >= eps: up, hi = True, x
    return tot + (hi - lo if up else 0.0)


def fib_sphere(n):
    i = np.arange(n) + 0.5
    ph, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, curves = [], {}
    with h5py.File(FILE, "r") as f:
        for g in GAMMAS:
            S = {s: load(f, s, g) for s in STATES}
            # (a) notebook method, reproduced: per-pair truncation, 91 pairs
            nb = []
            for a, b in itertools.combinations(STATES, 2):
                m = min(len(S[a][0]), len(S[b][0]))
                nb.append((npos(trdist(S[a][1][:m], S[b][1][:m])), a, b, S[a][0][m - 1]))
            nb.sort(reverse=True)
            # common window = where all 4 basis states exist
            n = min(len(S[s][0]) for s in STATES[:4])
            t = S["B1"][0][:n]
            bb = {s: S[s][1][:n] for s in STATES[:4]}
            # (b) same 91 pairs, but only pairs covering the common window, cut to it
            nb_common = max(npos(trdist(S[a][1][:n], S[b][1][:n]))
                            for a, b in itertools.combinations(STATES, 2)
                            if min(len(S[a][0]), len(S[b][0])) >= n)
            # affine map from B1=+z, B2=-z, B3=+x, B4=-y
            c = (bb["B1"] + bb["B2"]) / 2
            A = np.stack([bb["B3"] - c, -(bb["B4"] - c), (bb["B1"] - bb["B2"]) / 2], 2)
            lin_err = 0.0
            for s in STATES[4:]:
                m = min(n, len(S[s][0]))
                b0 = np.linalg.solve(A[0], S[s][1][0] - c[0])
                pred = np.einsum("tij,j->ti", A[:m], b0) + c[:m]
                lin_err = max(lin_err, np.abs(pred - S[s][1][:m]).max())
            # (c) optimum over antipodal pure pairs
            P = fib_sphere(6000)
            Dp = np.linalg.norm(np.einsum("tij,pj->pti", A, P), axis=2)
            dd = np.diff(Dp, axis=1)
            k = np.where(dd > 0, dd, 0).sum(1).argmax()
            obj = lambda x: -npos(np.linalg.norm(A @ (x / np.linalg.norm(x)), axis=1))
            r = minimize(obj, P[k], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-13))
            nopt = r.x / np.linalg.norm(r.x)
            D = np.linalg.norm(A @ nopt, axis=1)
            curves[g] = (t, D)
            rows.append(dict(gamma=g, t_start=t[0], t_end=t[-1], dt=t[1] - t[0],
                             N_notebook=nb[0][0], notebook_pair=f"{nb[0][1]}-{nb[0][2]}",
                             notebook_pair_t_end=nb[0][3], N_pairs_common_window=nb_common,
                             N_opt=npos(D), N_opt_every2=npos(D[::2]), N_opt_every4=npos(D[::4]),
                             N_opt_eps1e5=npos_eps(D, 1e-5), N_opt_eps1e4=npos_eps(D, 1e-4),
                             n_opt=" ".join(f"{x:+.3f}" for x in nopt), linearity_err=lin_err))
    keys = list(rows[0])
    with open(os.path.join(OUT, "blp_check.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    for r in rows:
        print(f"gamma={r['gamma']:>8}  notebook={r['N_notebook']:.3e} ({r['notebook_pair']}, t_end={r['notebook_pair_t_end']:.3g})"
              f"  pairs/common={r['N_pairs_common_window']:.3e}  opt={r['N_opt']:.3e}"
              f"  opt,eps=1e-4={r['N_opt_eps1e4']:.3e}  every4={r['N_opt_every4']:.3e}  linErr={r['linearity_err']:.1e}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 4, figsize=(14, 6))
        for a, r in zip(ax.flat, rows):
            t, D = curves[r["gamma"]]
            d = np.diff(D); cum = np.concatenate([[0], np.cumsum(np.where(d > 0, d, 0))])
            a.plot(t, D, lw=1, label="D(t), optimal pair")
            a2 = a.twinx(); a2.plot(t, cum, "C3", lw=1); a2.tick_params(axis="y", colors="C3")
            a.set_title(f"γ={r['gamma']}  N_opt={r['N_opt']:.2e}", fontsize=9); a.set_xlabel("t")
        ax[0, 0].set_ylabel("D(t)"); fig.text(0.995, 0.5, "running ∫σ>0 σ dt", color="C3", rotation=90, va="center")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "blp_check.pdf")); fig.savefig(os.path.join(OUT, "blp_check.png"), dpi=80)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
