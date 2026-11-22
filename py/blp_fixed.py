r"""Fixed Breuer-Laine-Piilo (BLP) non-Markovianity for the stored spin-boson data.

Three simple steps, on existing data only (DATA/ALL_GAMMAS_B4_D10.h5):

  Step 1  One time window per Q for every pair: keep only times where the four basis states
          B1..B4 all have data (Step 2 needs only these four).
  Step 2  Best pair of initial states: the qubit evolution is affine on the Bloch vector,
          b(t) = A(t) b0 + c(t), with A, c built from the basis states B1=+z, B2=-z, B3=+x, B4=-y.
          For the orthogonal pure pair b0 = +-n the trace distance is D(t) = |A(t) n|, and the BLP
          maximum lies on such pairs. We maximise over the unit vector n.
  Step 3  Noise threshold: rises of D smaller than eps are not counted (eps = 1e-4, the order of
          the data precision measured by the linearity residual).

The integral of sigma over sigma>0 is the sum of rises of D between samples, as before.
Error bar: spread of {eps = 0, eps = 1e-4} x {every sample, every 2nd sample}.

Writes NonMark_v2.pkl (numpy array, same order as NonMark.pkl) and py/out/blp_fixed.csv,
py/out/blp_old_vs_fixed.pdf, py/out/blp_2nu_oscillation.pdf.
Usage (from the repo root): python py/blp_fixed.py
"""
import itertools, os, pickle
import h5py
import numpy as np
from scipy.optimize import minimize

FILE = "DATA/ALL_GAMMAS_B4_D10.h5"
GAMMAS = ["0.079477", "0.25133", "0.79477", "2.5133", "7.9477", "25.133", "79.477", "251.33"]
Q = [316, 100, 31.6, 10, 3.16, 1, 0.316, 0.1]
STATES = ["B1", "B2", "B3", "B4"] + [f"D{i}" for i in range(1, 11)]
EPS = 1e-4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load(f, s, g):
    d = f[s][g]
    c = d["s_re"][()].ravel() + 1j * d["s_im"][()].ravel()
    return d["t"][()].ravel(), np.stack([2 * c.real, -2 * c.imag, 2 * d["p0"][()].ravel() - 1], 1)


def rises(D, eps=0.0):
    """Sum of rises of D; with eps > 0, rises and dips smaller than eps are ignored (hysteresis)."""
    if eps == 0:
        d = np.diff(D)
        return d[d > 0].sum()
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


def sphere(n):
    i = np.arange(n) + 0.5
    ph, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)


def best_pair(A):
    P = sphere(6000)
    Dp = np.linalg.norm(np.einsum("tij,pj->pti", A, P), axis=2)
    d = np.diff(Dp, axis=1)
    k = np.where(d > 0, d, 0).sum(1).argmax()
    r = minimize(lambda x: -rises(np.linalg.norm(A @ (x / np.linalg.norm(x)), axis=1)), P[k],
                 method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-13))
    n = r.x / np.linalg.norm(r.x)
    return n, np.linalg.norm(A @ n, axis=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    old = pickle.load(open("NonMark.pkl", "rb"))
    rows, curves = [], {}
    with h5py.File(FILE, "r") as f:
        for g, q, N_old in zip(GAMMAS, Q, old):
            S = {s: load(f, s, g) for s in STATES}
            n = min(len(S[s][0]) for s in STATES[:4])                   # Step 1
            n14 = min(len(S[s][0]) for s in STATES)
            t = S["B1"][0][:n]
            b = {s: S[s][1][:n] for s in STATES[:4]}
            b14 = {s: S[s][1][:n14] for s in STATES}
            c = (b["B1"] + b["B2"]) / 2                                   # Step 2
            A = np.stack([b["B3"] - c, -(b["B4"] - c), (b["B1"] - b["B2"]) / 2], 2)
            lin = 0.0
            for s in STATES[4:]:
                m = min(n, len(S[s][0]))
                pred = np.einsum("tij,j->ti", A[:m], np.linalg.solve(A[0], S[s][1][0] - c[0])) + c[:m]
                lin = max(lin, np.abs(pred - S[s][1][:m]).max())
            N_pairs = max(rises(0.5 * np.linalg.norm(b14[x] - b14[y], axis=1))
                          for x, y in itertools.combinations(STATES, 2))
            nvec, D = best_pair(A)
            N = rises(D, EPS)                                             # Step 3
            N_14 = rises(D[:n14], EPS)  # same, cut to where all 14 states have data
            spread = [rises(D, e) for e in (0.0, EPS)] + [rises(D[::2], e) for e in (0.0, EPS)]
            curves[g] = (t, D)
            rows.append(dict(Q=q, gamma=g, t_start=t[0], t_end=t[-1], dt=t[1] - t[0], N_old=N_old,
                             t_end_14states=S['B1'][0][n14 - 1], N_pairs_14window=N_pairs, N_fixed_14window=N_14, N_best_eps0=rises(D), N_fixed=N,
                             N_low=min(spread), N_high=max(spread), linearity_err=lin,
                             n_best=" ".join(f"{x:+.3f}" for x in nvec)))
    N_fixed = np.array([r["N_fixed"] for r in rows])
    pickle.dump(N_fixed, open("NonMark_v2.pkl", "wb"))
    keys = list(rows[0])
    with open(os.path.join(OUT, "blp_fixed.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(f"{'Q':>6} {'window':>16} {'old':>9} {'91pr,14w':>9} {'best':>9} {'fixed':>9}  [low, high]")
    for r in rows:
        print(f"{r['Q']:>6} [{r['t_start']:.3g}, {r['t_end']:.3g}]".ljust(24) +
              f"{r['N_old']:9.2e} {r['N_pairs_14window']:9.2e} {r['N_best_eps0']:9.2e} {r['N_fixed']:9.2e}"
              f"  [{r['N_low']:.1e}, {r['N_high']:.1e}]")
    plots(rows, curves)


def plots(rows, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [r["N_old"] for r in rows], "x--", color="0.5", label="current (NonMark.pkl)")
    Nf = np.array([r["N_fixed"] for r in rows]); lo = np.array([r["N_low"] for r in rows]); hi = np.array([r["N_high"] for r in rows])
    pos = Nf > 0
    ax.errorbar(x[pos], Nf[pos], yerr=[Nf[pos] - np.maximum(lo[pos], 1e-7), hi[pos] - Nf[pos]], fmt="o-", color="C3",
                capsize=3, label=f"fixed (shared window, best pair, ε={EPS:g})")
    ax.plot(x[~pos], np.full((~pos).sum(), EPS), "v", color="C3", ms=8, label=f"below threshold (< {EPS:g})")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 0.2)
    ax.set_xticks(x, [r["Q"] for r in rows]); ax.set_xlabel("Q = ν/γ"); ax.set_ylabel("BLP non-Markovianity N")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, "blp_old_vs_fixed.pdf"))
    fig.savefig(os.path.join(OUT, "blp_old_vs_fixed.png"), dpi=90)
    # what produces the rises at high Q: a small 2ν oscillation riding on the decay of D
    from scipy.signal import savgol_filter
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, g, q in zip(axs, ["0.25133", "0.079477"], [100, 316]):
        t, D = curves[g]
        w = int(round(1.0 / (t[1] - t[0]))) | 1
        S = savgol_filter(D, w, 2)
        m = (t > 19) & (t < 20.5)
        a.plot(t[m], D[m], ".-", ms=3, lw=0.8, label="D(t), best pair")
        a.plot(t[m], S[m], "k--", lw=0.8, label="smooth decay")
        a.set_title(f"Q = {q}: oscillation at 2ν on top of the decay"); a.set_xlabel("t"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "blp_2nu_oscillation.pdf"))
    fig.savefig(os.path.join(OUT, "blp_2nu_oscillation.png"), dpi=90)


if __name__ == "__main__":
    main()
