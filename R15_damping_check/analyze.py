"""R15: reproduce Kurt's damping-only master-equation check (email of 23 Sep 2026) and
compare it with our TSSOS fits on the same window.

Run from anywhere:  python R15_damping_check/analyze.py
Writes results.csv, results_window_from_t0.csv and three PDFs next to this file.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from r15_common import (DODECA, GAMMAS, N_END, N_START, Q_LABELS, W0, damping_model, kossak_fit,
                        kurt_frequency, lindblad_fit, model_params, propagate, read_evolution,
                        trace_distance)
from refit import fit, lindblad4, quadrature_frequency

HERE = os.path.dirname(os.path.abspath(__file__))
# Categorical slots 1-4 of the colour-blind-checked default palette, in fixed order
C_KURT, C_LIND, C_KOSS, C_DEB, C_BOOLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"


def window(t, k, mode):
    """Indices of Kurt's window [n = 11, N_END[k]] in our stored arrays.

    mode "index": Kurt's sample numbers index our stored arrays directly (1-based).
    mode "t0":    Kurt's sample n sits at t = (n - 1) dt with t = 0 at n = 1; our files start
                  later (the first points were cut), so take our samples inside that time span.
    """
    if mode == "index":
        return np.arange(N_START - 1, N_END[k])
    dt = t[1] - t[0]
    t_lo, t_hi = (N_START - 1) * dt, (N_END[k] - 1) * dt
    return np.where((t >= t_lo - 1e-9) & (t <= t_hi + 1e-9))[0]


def scores(model, g, k, mode):
    """Trace distance per state and time, propagating each model from the data at window start."""
    H, Ls = model
    tds = []
    for s in DODECA:
        t, rho = read_evolution(s, g)
        idx = window(t, k, mode)
        tds.append(trace_distance(rho[idx], propagate(H, Ls, rho[idx[0]], t[idx])))
    return np.array(tds)


def best_frequency(g, k, mode):
    """Kurt chose w to fit best; here: w minimising the average trace distance (gamma fixed)."""
    gm = float(g)
    res = minimize_scalar(lambda w: scores(damping_model(w, gm), g, k, mode).mean(),
                          bounds=(W0 - 0.5, W0 + 0.5), method="bounded", options={"xatol": 1e-6})
    return res.x


def debias_simpson(H, gamma, h):
    """Replace the fitted frequency w_fit by the w whose Simpson-fitted value is w_fit."""
    w_fit = (H[0, 0] - H[1, 1]).real
    w = brentq(lambda x: quadrature_frequency(x, gamma, h, "simpson") - w_fit, w_fit - 0.5, w_fit + 0.5)
    H2 = H.copy()
    H2[0, 0], H2[1, 1] = w / 2, -w / 2
    return H2, w


def tssos_start(g):
    H, Ls, _ = lindblad_fit(g)
    return [2 * H[0, 0].real, 2 * H[0, 1].real, 2 * H[0, 1].imag, Ls[0][0, 1].real, Ls[1][0, 1].real]


def run(mode):
    rows, series = [], {}
    for k, g in enumerate(GAMMAS):
        gm = float(g)
        t, _ = read_evolution("D1", g)
        h = t[1] - t[0]
        H_l, L_l, st_l = lindblad_fit(g)
        H_k, L_k, st_k = kossak_fit(g)
        x_boole, _ = fit(g, fit(g, tssos_start(g), "simpson")[0], "boole")
        w_best = best_frequency(g, k, mode)
        H_d, w_d = debias_simpson(H_l, gm, h)
        models = {
            "kurt_ME": damping_model(kurt_frequency(gm), gm),
            "bare_ME": damping_model(W0, gm),
            "bestw_ME": damping_model(w_best, gm),
            "lindblad_tssos": (H_l, L_l),
            "kossak_tssos": (H_k, L_k),
            "lindblad_boole": lindblad4(x_boole),
            "lindblad_debiased": (H_d, L_l),
        }
        row = {"Q": Q_LABELS[k], "gamma": gm, "dt": h, "w_dt": W0 * h,
               "t_start": t[window(t, k, mode)[0]], "t_end": t[window(t, k, mode)[-1]],
               "w_kurt": kurt_frequency(gm), "w_best_ME": w_best,
               "w_simpson_pred": quadrature_frequency(kurt_frequency(gm), gm, h, "simpson"),
               "w_lindblad": model_params(H_l, L_l)[0], "w_kossak": model_params(H_k, L_k)[0],
               "w_lindblad_boole": x_boole[0], "w_lindblad_debiased": w_d, "status_lindblad": st_l, "status_kossak": st_k}
        for name, m in models.items():
            td = scores(m, g, k, mode)
            row[f"avg_{name}"], row[f"max_{name}"] = td.mean(), td.max()
            series[(k, name)] = td
        rows.append(row)
    return rows, series


def write_csv(rows, path):
    keys = list(rows[0])
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in keys) + "\n")


def plot_vs_q(rows, path):
    q = [float(r["Q"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, label, c, mk in [("kurt_ME", "damping-only ME (Kurt's w)", C_KURT, "o"),
                               ("lindblad_tssos", "Lindblad fit (TSSOS, Simpson)", C_LIND, "s"),
                               ("kossak_tssos", "Kossakowski fit (TSSOS, Simpson)", C_KOSS, "^"),
                               ("lindblad_debiased", "Lindblad fit, Simpson bias removed", C_DEB, "v")]:
        ax.plot(q, [r[f"avg_{name}"] for r in rows], "-", color=c, marker=mk, ms=6, lw=2, label=f"{label}: average")
        ax.plot(q, [r[f"max_{name}"] for r in rows], "--", color=c, marker=mk, ms=6, lw=1.5, mfc="none",
                label=f"{label}: max")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Q"); ax.set_ylabel("trace distance to spin-boson data")
    ax.set_title("10 dodeca states, Kurt's sample windows", fontsize=10)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_q316_time(series, path):
    t, _ = read_evolution("D1", GAMMAS[0])
    tw = t[window(t, 0, "index")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, (name, title, c) in zip(axes, [("kurt_ME", "damping-only ME (Kurt's w)", C_KURT),
                                           ("lindblad_tssos", "Lindblad fit (TSSOS, Simpson)", C_LIND),
                                           ("lindblad_debiased", "Lindblad fit, Simpson bias removed", C_DEB)]):
        td = series[(0, name)]
        for s in td:
            ax.plot(tw, s, color=c, lw=0.8, alpha=0.55)
        ax.plot(tw, td.mean(0), color="k", lw=1.5, label="average over states")
        ax.set_title(title, fontsize=9); ax.set_xlabel("t"); ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("trace distance, Q = 316"); axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_bias(rows, path):
    th = np.logspace(-2.2, np.log10(0.7), 200)
    w = W0
    simp = [(quadrature_frequency(w, 0.0, x / w, "simpson") - w) / w for x in th]
    boole = [(quadrature_frequency(w, 0.0, x / w, "boole") - w) / w for x in th]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(th, np.abs(simp), color=C_LIND, lw=2, label="Simpson rule (paper)")
    ax.plot(th, np.abs(boole), color=C_BOOLE, lw=2, ls="--", label="Boole rule")
    for r in rows:
        ax.plot(r["w_dt"], abs(r["w_lindblad"] - r["w_kurt"]) / r["w_kurt"], "s", color=C_LIND, ms=7,
                mec="white", mew=1)
        ax.annotate(f"Q={r['Q']}", (r["w_dt"], abs(r["w_lindblad"] - r["w_kurt"]) / r["w_kurt"]),
                    textcoords="offset points", xytext=(-8, 7), fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-10, 5e-2)
    ax.set_xlabel(r"$\omega\,\Delta t$ (phase advance per sample)")
    ax.set_ylabel(r"relative frequency error $|\omega_{fit}-\omega|/\omega$")
    ax.set_title("Quadrature bias in fitted frequency; squares = TSSOS fits vs Kurt's w", fontsize=9)
    ax.grid(alpha=0.25, which="both", lw=0.5); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    rows, series = run("index")
    write_csv(rows, f"{HERE}/results.csv")
    plot_vs_q(rows, f"{HERE}/trace_dist_vs_Q.pdf")
    plot_q316_time(series, f"{HERE}/trace_dist_Q316_vs_time.pdf")
    plot_bias(rows, f"{HERE}/simpson_frequency_bias.pdf")
    rows_t0, _ = run("t0")
    write_csv(rows_t0, f"{HERE}/results_window_from_t0.csv")
    for label, rr in [("window = sample indices of our files", rows), ("window = times from t = 0", rows_t0)]:
        print(f"\n{label}")
        print(f"{'Q':>5} {'t0':>6} {'t1':>6} | {'Kurt avg':>9} {'max':>7} | {'bestw avg':>9} | "
              f"{'Lind avg':>9} {'max':>7} | {'Koss avg':>9} {'max':>7} | {'Boole avg':>9} {'max':>7}")
        for r in rr:
            print(f"{r['Q']:>5} {r['t_start']:6.3f} {r['t_end']:6.2f} | {r['avg_kurt_ME']:9.5f} {r['max_kurt_ME']:7.4f} | "
                  f"{r['avg_bestw_ME']:9.5f} | {r['avg_lindblad_tssos']:9.5f} {r['max_lindblad_tssos']:7.4f} | "
                  f"{r['avg_kossak_tssos']:9.5f} {r['max_kossak_tssos']:7.4f} | "
                  f"{r['avg_lindblad_boole']:9.5f} {r['max_lindblad_boole']:7.4f} | "
                  f"{r['avg_lindblad_debiased']:10.5f} {r['max_lindblad_debiased']:7.4f}")
    print("\nfrequencies")
    for r in rows:
        print(f"Q={r['Q']:>5} w_dt={r['w_dt']:.3f} w_kurt={r['w_kurt']:.5f} w_best={r['w_best_ME']:.5f} "
              f"simpson_pred={r['w_simpson_pred']:.5f} lindblad={r['w_lindblad']:.5f} kossak={r['w_kossak']:.5f} "
              f"boole={r['w_lindblad_boole']:.5f} debiased={r['w_lindblad_debiased']:.5f} [{r['status_lindblad']}/{r['status_kossak']}]")
