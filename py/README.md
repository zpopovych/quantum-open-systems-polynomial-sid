# Python diagnostics (no Julia needed)

`sid_tools.py` reads `DATA/ALL_GAMMAS_B4_D10.h5` and the saved TSSOS models
(`E_KOSSAK_..._16-44.h5`, `E_LINDBLAD4_..._17-08.h5`, the same files used by
`03B_rescore_tracedist_allstates.ipynb`) and evolves them exactly with
`expm` of the Liouvillian. It reproduces the Julia/QuantumOptics trace distances
in `E_{KOSSAK,LINDBLAD}_TRACEDIST_ALLSTATES_2026-07-12.h5` to within 1.3e-5 for
all 8 couplings and 14 initial states.

Note: matrices written by Julia come back transposed in h5py; `read_model`
transposes them back.

Run from the repo root:

```
pip install h5py numpy scipy matplotlib
python3 py/07_Q316_error_vs_time.py         # Kurt's R11
python3 py/08_export_excited_pop_gamma5-8.py  # Kurt's R10
```

## R11: Q = 316 error vs time (`out/Q316_*`)

Model minus spin-boson data for the excited population and for Re, Im of the
coherence, all 14 initial states on one plot (`Q316_error_vs_time_{kossak,lindblad}.pdf`).

| | Lindblad fit | Kossakowski fit |
|---|---|---|
| max excited-population error | 0.0015 | 0.054 |
| max Re/Im coherence error | 0.043 | 0.050 |
| max trace distance | 0.044 | 0.065 |
| max trace distance, Simpson bias removed | **0.0023** | 0.055 |

What the plots show:

1. **The coherence error comes from a frequency error, and that error is the
   Simpson discretisation of the loss.** The spin-boson coherences rotate at
   25.1353 (the same as Kurt's Lamb-shifted 8π + 0.0155γ/(2π)(Om/w + log(Om/w − 1))
   = 25.1351), while both fits rotate at 25.1261. At Q = 316 the data are
   sampled with dt = 0.02, so ω·dt ≈ 0.50 rad per step (about 12 samples per
   period). The double-step Simpson loss maps a true eigenvalue λ to
   μ = 3(e^{2λdt} − 1) / (dt(1 + 4e^{λdt} + e^{2λdt})); for the data's λ this gives
   exactly 25.1261. TSSOS found the global optimum of a biased objective. The
   phase error grows to 0.23 rad over the run, which is the growing envelope in
   the coherence plots. All other couplings are sampled 3 to 400 times more
   finely, so the bias there is 0.003 rad or less over the whole run.
2. **Inverting that map on the fitted Liouvillian fixes the Lindblad fit**:
   the max trace distance falls from 0.044 to 0.0023 over all 14 states, with
   no refit (`Q316_tracedist_fitted_vs_simpson_corrected.pdf`).
3. **The Kossakowski fit has a second, separate error.** Its decay rates are
   too large (population 0.0886 vs γ = 0.0795, coherence 0.0470 vs γ/2 = 0.0397)
   and its steady state is not the ground state: the ground state B1 is heated
   to 5% excited population. This is not a discretisation effect and needs a
   look at the Kossakowski objective or constraints at this coupling.
4. The excited-population error of the Lindblad fit is at most 1.5e-3, which
   agrees with Kurt's observation that the excited-state population fits well.

## R10: excited population for γ = 5 to 8 (`out/excited_pop_gamma5-8/`)

Initial state B2 (excited), γ indices 5 to 8 = 7.9477, 25.133, 79.477, 251.33
(Q = 3.16, 1, 0.316, 0.1). One file per γ and ansatz, in Kurt's four-column
format `time pop_excited real_coherence imag_coherence`, on the spin-boson time
grid, plus the spin-boson data in the same format and a preview plot.

The Kossakowski fit is the best Markovian fit at all four couplings (lower
mean trace distance over the 14 states: 0.017, 0.063, 0.037, 0.024 against
0.021, 0.104, 0.158, 0.266 for Lindblad). At γ5 the data start at t = 0.0505
(the first samples were dropped for the initial slip), so the excited
population starts at 0.78, not 1.

## Exponentially fitted refit at Q = 316 (`09_expfit_refit_Q316.py`)

This replaces the Simpson weights (1/3, 4/3, 1/3) in the loss with weights that are exact for 1 and e^{±iω₀t}: a = (1 − sinθ/θ)/(1 − cosθ), b = 2 − 2a, θ = ω₀h. The loss stays a quartic polynomial. This script runs a local least-squares check from the TSSOS start, using the notebook's 5-parameter Lindblad ansatz.

- Simpson: ε = 25.12607 and mean trace distance 0.0216 over the 14 states. This is the same as TSSOS.
- Exponentially fitted: ε = 25.13526 and mean trace distance 0.00041 (max 0.0015).
