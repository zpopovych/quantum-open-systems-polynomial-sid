"""R10 (Kurt, 6 Sep 2026): excited-state population of the best Markovian fit,
excited initial state (B2 = |1><1|), for gamma values 5, 6, 7, 8.

"gamma 5..8" is read as the 5th to 8th coupling in GAMMAS (Q = 3.16, 1, 0.316, 0.1).
The best Markovian fit is the Kossakowski TSSOS model: it has the lower mean
trace distance over all 14 initial states at each of these couplings. The
Lindblad fit is exported too for comparison.

Files use Kurt's Lorentzian-data format, one line per time point:
    time pop_excited real_coherence imag_coherence
(the header line starts with % so MATLAB load() skips it), with the evolution started from the first spin-boson sample, on the same time
grid as the spin-boson data. The spin-boson data itself is written in the same
format so both can be overlaid directly.

Run from the repo root:  python3 py/08_export_excited_pop_gamma5-8.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sid_tools import (ALL_STATES, EVOL_DATA, GAMMAS, Q_of, evolve_L,
                       liouvillian, read_model, read_timeevolution,
                       trace_distance)

OUT = "py/out/excited_pop_gamma5-8"
STATE = "B2"
HEADER = "time pop_excited real_coherence imag_coherence"


def four_columns(t, rho):
    return np.column_stack([t, rho[:, 1, 1].real, rho[:, 0, 1].real, rho[:, 0, 1].imag])


def mean_td_all_states(L, gamma):
    tds = []
    for s in ALL_STATES:
        t, rho = read_timeevolution(EVOL_DATA, s, gamma)
        tds.append(trace_distance(rho, evolve_L(L, rho[0], t)).mean())
    return np.mean(tds)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(8, 5.5))
    lines = ["gamma_index gamma Q ansatz tssos_status mean_tracedist_all_states "
             "max_abs_pop_error_excited_state file"]
    for k in range(5, 9):
        gamma = GAMMAS[k - 1]
        t, rho = read_timeevolution(EVOL_DATA, STATE, gamma)
        sb_file = f"{OUT}/spin_boson_gamma{k}_excited.dat"
        np.savetxt(sb_file, four_columns(t, rho), fmt="%.10e", header=HEADER, comments="% ")
        ax = axs.flat[k - 5]
        ax.plot(t, rho[:, 1, 1].real, color="#0b0b0b", lw=1.4, label="spin-boson")
        for kind, color in [("kossak", "#2a78d6"), ("lindblad", "#eb6834")]:
            H, ops, status = read_model(kind, gamma)
            L = liouvillian(H, ops)
            fit = evolve_L(L, rho[0], t)
            fn = f"{OUT}/markov_fit_{kind}_gamma{k}_excited.dat"
            np.savetxt(fn, four_columns(t, fit), fmt="%.10e", header=HEADER, comments="% ")
            ax.plot(t, fit[:, 1, 1].real, color=color, lw=1.1, ls="--",
                    label={"kossak": "Kossakowski", "lindblad": "Lindblad"}[kind] + " fit")
            err = np.max(np.abs(fit[:, 1, 1] - rho[:, 1, 1]))
            lines.append(f"{k} {gamma} {Q_of(gamma):.4g} {kind} {status} "
                         f"{mean_td_all_states(L, gamma):.5f} {err:.5f} {os.path.basename(fn)}")
        ax.set_title(f"gamma {k} = {gamma}, Q = {Q_of(gamma):.3g}", fontsize=9)
        ax.set_xlabel("time")
        ax.set_ylabel("excited population")
    axs.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/excited_pop_gamma5-8.pdf")
    fig.savefig(f"{OUT}/excited_pop_gamma5-8.png", dpi=150)
    with open(f"{OUT}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
