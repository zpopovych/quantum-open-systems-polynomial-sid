"""R11 (Kurt, 6 Sep 2026): error vs time at the weakest damping, Q = 316.

For every initial state (B1-B4, D1-D10) plots model minus spin-boson data for
  1. the excited-state population rho_11,
  2. Re and Im of the coherence rho_01,
all states on the same axes, for the Kossakowski and Lindblad TSSOS fits.
Also writes the trace distance of the fitted model and of the same model with
the Simpson frequency bias removed (see sid_tools.simpson_corrected_liouvillian).

Run from the repo root:  python3 py/07_Q316_error_vs_time.py
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sid_tools import (ALL_STATES, EVOL_DATA, Q_of, evolve_L, liouvillian,
                       read_model, read_timeevolution,
                       simpson_corrected_liouvillian, trace_distance)

GAMMA = "0.079477"  # Q = 316
OUT = "py/out"
NAMES = {"kossak": "Kossakowski", "lindblad": "Lindblad"}
BASIS_COLORS = {"B1": "#2a78d6", "B2": "#eb6834", "B3": "#1baf7a", "B4": "#eda100"}
BASIS_LABELS = {"B1": r"B1 $|0\rangle$", "B2": r"B2 $|1\rangle$ (excited)",
                "B3": r"B3 $|x\rangle$", "B4": r"B4 $|y\rangle$"}
DODECA_GREYS = plt.cm.Greys(np.linspace(0.35, 0.75, 10))

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e5e5", "grid.linewidth": 0.5})


def style(state):
    if state in BASIS_COLORS:
        return dict(color=BASIS_COLORS[state], lw=1.2, label=BASIS_LABELS[state], zorder=3)
    i = int(state[1:]) - 1
    return dict(color=DODECA_GREYS[i], lw=0.6, zorder=2,
                label="D1-D10 (dodecahedron)" if i == 0 else None)


def run(kind):
    H, ops, status = read_model(kind, GAMMA)
    L = liouvillian(H, ops)
    res = {}
    for s in ALL_STATES:
        t, rho = read_timeevolution(EVOL_DATA, s, GAMMA)
        dt = np.median(np.diff(t))
        fit = evolve_L(L, rho[0], t)
        cor = evolve_L(simpson_corrected_liouvillian(L, dt), rho[0], t)
        res[s] = dict(t=t, d_pe=(fit[:, 1, 1] - rho[:, 1, 1]).real,
                      d_re=(fit[:, 0, 1] - rho[:, 0, 1]).real,
                      d_im=(fit[:, 0, 1] - rho[:, 0, 1]).imag,
                      td=trace_distance(rho, fit), td_cor=trace_distance(rho, cor))
    return res, status, L


def plot_errors(kind, res, status):
    fig, axs = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True)
    keys = [("d_pe", r"$\Delta\rho_{11}$ (excited population)"),
            ("d_re", r"$\Delta\,\mathrm{Re}\,\rho_{01}$"),
            ("d_im", r"$\Delta\,\mathrm{Im}\,\rho_{01}$")]
    for ax, (k, lab) in zip(axs, keys):
        for s in ALL_STATES:
            ax.plot(res[s]["t"], res[s][k], **style(s))
        ax.set_ylabel(lab)
        ax.axhline(0, color="#52514e", lw=0.5)
    axs[-1].set_xlabel("time")
    axs[0].legend(ncol=3, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"{NAMES[kind]} fit minus spin-boson data, Q = {Q_of(GAMMA):.0f} "
                 f"(TSSOS {status})", fontsize=10)
    fig.tight_layout()
    fn = f"{OUT}/Q316_error_vs_time_{kind}.pdf"
    fig.savefig(fn); fig.savefig(fn.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    return fn


def plot_tracedist(all_res):
    fig, axs = plt.subplots(2, 2, figsize=(9, 5.5), sharex=True, sharey=True)
    for row, kind in enumerate(["kossak", "lindblad"]):
        for col, (k, title) in enumerate([("td", "as fitted"),
                                          ("td_cor", "Simpson frequency bias removed")]):
            ax = axs[row, col]
            for s in ALL_STATES:
                ax.plot(all_res[kind][s]["t"], all_res[kind][s][k], **style(s))
            ax.set_title(f"{NAMES[kind]}, {title}", fontsize=9)
            if col == 0:
                ax.set_ylabel("trace distance")
    for ax in axs[-1]:
        ax.set_xlabel("time")
    axs[0, 0].legend(ncol=2, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"Trace distance to spin-boson data, Q = {Q_of(GAMMA):.0f}", fontsize=10)
    fig.tight_layout()
    fn = f"{OUT}/Q316_tracedist_fitted_vs_simpson_corrected.pdf"
    fig.savefig(fn); fig.savefig(fn.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    return fn


def main():
    os.makedirs(OUT, exist_ok=True)
    all_res, rows = {}, []
    for kind in ["kossak", "lindblad"]:
        res, status, L = run(kind)
        all_res[kind] = res
        print("wrote", plot_errors(kind, res, status))
        ev = np.linalg.eigvals(L)
        print(f"  {NAMES[kind]} Liouvillian eigenvalues: {np.round(ev, 5)}")
        for s in ALL_STATES:
            r = res[s]
            rows.append([kind, s, status] + [f"{np.max(np.abs(r[k])):.5f}" for k in
                        ("d_pe", "d_re", "d_im", "td", "td_cor")] + [f"{np.mean(r['td']):.5f}",
                        f"{np.mean(r['td_cor']):.5f}"])
    print("wrote", plot_tracedist(all_res))
    fn = f"{OUT}/Q316_error_summary.csv"
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ansatz", "state", "tssos_status", "max_abs_d_excited_pop",
                    "max_abs_d_re_coh", "max_abs_d_im_coh", "max_tracedist",
                    "max_tracedist_simpson_corrected", "mean_tracedist",
                    "mean_tracedist_simpson_corrected"])
        w.writerows(rows)
    print("wrote", fn)


if __name__ == "__main__":
    main()
