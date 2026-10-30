"""Check the Python propagation against the Julia/QuantumOptics rescores of 12 Jul 2026."""
import h5py
import numpy as np

from r15_common import BASIS, DODECA, GAMMAS, REPO, kossak_fit, lindblad_fit, propagate, read_evolution, trace_distance

for fit, ref in [(lindblad_fit, "E_LINDBLAD_TRACEDIST_ALLSTATES_2026-07-12.h5"),
                 (kossak_fit, "E_KOSSAK_TRACEDIST_ALLSTATES_2026-07-12.h5")]:
    worst = 0.0
    with h5py.File(f"{REPO}/{ref}", "r") as f:
        for g in GAMMAS:
            H, Ls, _ = fit(g)
            for s in BASIS + DODECA:
                t, rho = read_evolution(s, g)
                td = trace_distance(rho, propagate(H, Ls, rho[0], t))
                worst = max(worst, np.abs(td - f[g][s]["TraceDist"][()]).max())
    print(f"{ref}: max |Python - Julia| trace distance = {worst:.1e}")
