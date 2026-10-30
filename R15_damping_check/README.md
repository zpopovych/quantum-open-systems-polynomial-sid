# R15: Kurt's damping-only master-equation check (Q = 316 to 3.16)

Kurt's request (email of 23 Sep 2026): simulate the two-parameter master equation
(damping rate = the spin-boson γ, Lamb-shifted frequency
`w = 8π + 0.0155·γ/(2π)·(Om/w + log(Om/w − 1))`, `Om = 10w = 80π`) for the 10 dodeca states,
from sample n = 11 to samples 1001 / 2801 / 2516 / 1591 / 1258 for Q = 316 / 100 / 31.6 / 10 / 3.16.
Report the average and max trace distance to the spin-boson data, check that we get his numbers,
then find what is wrong with our Q = 316 fit.

Everything here is Python (numpy/scipy/h5py/matplotlib), no Julia.

```
python R15_damping_check/check_propagation.py   # Python propagation vs the Julia rescores
python R15_damping_check/analyze.py             # tables, CSVs and the three PDFs (~40 s)
```

## Result

**1. Kurt's check reproduces.** The damping-only ME is accurate at weak damping and gets
steadily better with Q, as Kurt says. At Q = 316 it beats our TSSOS fit by a factor of ~35.

| Q | ME avg | ME max | Lindblad fit avg | max | Kossakowski fit avg | max |
|---|---|---|---|---|---|---|
| 316 | 0.00058 | 0.0012 | **0.0214** | **0.041** | 0.033 | 0.060 |
| 100 | 0.0015 | 0.0026 | 0.00084 | 0.0026 | 0.018 | 0.026 |
| 31.6 | 0.0043 | 0.0084 | 0.0024 | 0.0084 | 0.016 | 0.021 |
| 10 | 0.0135 | 0.0255 | 0.0068 | 0.0246 | 0.014 | 0.019 |
| 3.16 | 0.048 | 0.081 | 0.024 | 0.070 | 0.021 | 0.049 |

Trace distance between the spin-boson data and each model, propagated from the data at the
window start, over the 10 dodeca states (`results.csv`, `trace_dist_vs_Q.pdf`). At Q ≤ 100 the
Lindblad fit is as good as the ME or better, so only Q = 316 is anomalous.

**Comparison with Kurt's plot** (`trace_dist_to_damped_qubit.pdf`, values read off the log axis):

| Q | Kurt avg | ours avg | Kurt max | ours max |
|---|---|---|---|---|
| 316 | ~8e-5 | 5.8e-4 | ~9e-4 | 1.2e-3 |
| 100 | ~2.3e-4 | 1.5e-3 | ~2.5e-3 | 2.6e-3 |
| 31.6 | ~8.7e-4 | 4.3e-3 | ~9.5e-3 | 8.4e-3 |
| 10 | ~3.2e-3 | 1.35e-2 | ~3.6e-2 | 2.55e-2 |
| 3.16 | ~1.2e-2 | 4.8e-2 | ~0.115 | 8.1e-2 |

The maxima agree within about 30%, and both curves fall with Q at a similar rate.
Our averages are 4 to 7 times higher than his, and his max/avg ratio is about 10
where ours is about 2. So the reproduction **does not match yet**.

The reason is in the data. Our spin-boson coherences contain a counter-rotating (e^{+iωt})
component that a secular damping ME cannot produce. Its amplitude is 3.1e-4 / 9.7e-4 / 3.0e-3 /
9.1e-3 / 2.4e-2 for Q = 316 to 3.16. That alone puts a floor of a few 1e-4 on the average
trace distance at Q = 316, above Kurt's 8e-5. Choosing the best initial state by least squares
instead of the data point at n = 11 does not change this (avg 3.7e-4 at Q = 316). So Kurt is
probably using different data or a different average. Questions for him: which data files
(sampling and cutoff), what the ME starts from at n = 11, and how the average is taken (over
which times and states).

The diagnosis below does not depend on this. At Q = 316, both his ME and ours are 35 to 250 times
better than the TSSOS fit, and the fit's frequency error is fully explained by the Simpson bias.

**2. Cause of the poor Q = 316 fit: Simpson discretisation bias in the frequency.**
The Q = 316 data is sampled at dt = 0.02, so the qubit phase advances by ω·dt ≈ 0.50 rad per
sample. Ten times more than at Q = 100 (0.17 rad), and the Simpson rule in the loss
`ρ[i] − ρ[i−2] = dt·L((ρ[i−2] + 4ρ[i−1] + ρ[i])/3)` is biased at that step. For a noiseless
damped oscillation with the true ω, the Simpson loss is minimised at

| | ω |
|---|---|
| Kurt's Lamb-shifted ω | 25.13513 |
| ω minimising the ME's average trace distance | 25.13531 |
| Simpson-loss optimum predicted for Kurt's ω | 25.12594 |
| TSSOS Lindblad fit | 25.12607 |
| TSSOS Kossakowski fit | 25.12610 |

The fitted frequency is low by 0.009, which is the Simpson prediction to 1e-4. A 0.009 rad
frequency error accumulates to ~0.2 rad of phase by t = 20, which is the whole 0.02 to 0.04
coherence error (`trace_dist_Q316_vs_time.pdf`: the error grows smoothly with t, and is smallest for the
two states closest to the poles, which have the least coherence). The relative bias scales as (ω·dt)⁴/180
(`simpson_frequency_bias.pdf`), so it is ~10⁻⁵ at Q = 100 and negligible below. The damping rate
is fitted correctly (0.0797 vs γ = 0.0795).

**3. Fix check.** Keeping our fitted Lindblad operators and only inverting the Simpson map for the
frequency (ω = 25.13526) brings Q = 316 to avg 0.00062, max 0.0012, the same as Kurt's ME.
Refitting with the 5-point Boole rule instead of Simpson (still linear in L, so the objective stays
a quartic polynomial for TSSOS) gives avg 0.0022, max 0.0046: better, but still biased at
ω·dt = 0.5. Python least squares on the Simpson loss reproduces the TSSOS optimum (ε = 25.12607),
so TSSOS found the right minimum of the wrong loss.

So the spin-boson simulation is fine, and the fit is optimal for the loss we wrote down. Options for
the paper: (a) correct the frequency analytically after the fit (cheapest), (b) use a higher-order
rule or a finer time grid at Q = 316, (c) state the bias and its (ω·dt)⁴ scaling in Sec. II A and
III D (this also answers referee 3's discretisation point).

## Other findings

- **Kossakowski fit is worse than the Lindblad fit at every Q ≤ 100** (avg 0.014 to 0.018 vs
  0.0008 to 0.007). Its population decay rate is too large (0.271 vs γ = 0.251 at Q = 100;
  0.089 vs 0.0795 at Q = 316). This is separate from the Simpson bias and worth a look before the
  Kossakowski curves go into the paper.
- **Window choice does not matter.** Our files start at t = 0.06 / 0.12 / 0.0775 / 0.064 / 0.056
  (the first points were cut), so "sample n" is ambiguous. Reading n as an index into our files
  (`results.csv`) or as t = (n − 1)·dt from t = 0 (`results_window_from_t0.csv`) changes the
  numbers by under 10%.
- **Dodeca numbering.** Our D1 to D10 are taken as Kurt's states 1 3 5 7 9 10 11 13 15 20 in order.
  The averages and maxima do not depend on this mapping.
- **At Q = 3.16** the "best" ME frequency (25.63) is far from Kurt's formula (25.37) and the TSSOS
  fits disagree with each other; weak-coupling theory is not expected to hold there.

## Validation

`check_propagation.py` compares the Python propagation (exact matrix exponential) with the Julia
QuantumOptics rescores `E_*_TRACEDIST_ALLSTATES_2026-07-12.h5` for all 14 states and 5 γ:
Lindblad agrees to 1.3e-5 everywhere, Kossakowski to 6e-6 at Q ≥ 10. At Q = 3.16 the Kossakowski
values differ by up to 0.017. The Python model was checked against the explicit Kossakowski
dissipator to 1e-15, so the difference is on the Julia side (probably ODE solver tolerance on the
fast dynamics) and does not affect Q = 316.

Conventions are in `r15_common.py`: matrices written by Julia are transposed when read by h5py
(the Kossakowski `C` is used as read, which is the orientation that matches the Julia rescore).

## Files

| File | What |
|---|---|
| `r15_common.py` | data loading, Kurt's frequency, models, exact propagation, trace distance |
| `refit.py` | Python Lindblad-4 fit with Simpson or Boole loss; quadrature frequency bias |
| `analyze.py` | runs everything, writes CSVs and PDFs |
| `check_propagation.py` | Python vs Julia trace distances |
| `results.csv`, `results_window_from_t0.csv` | per-Q numbers for every model |
| `analyze_output.txt` | console output of `analyze.py` |
| `trace_dist_vs_Q.pdf` | average (solid) and max (dashed) trace distance vs Q, Kurt's style |
| `trace_dist_Q316_vs_time.pdf` | trace distance vs time at Q = 316, all 10 states |
| `simpson_frequency_bias.pdf` | Simpson and Boole frequency bias vs ω·dt with the TSSOS fits |
