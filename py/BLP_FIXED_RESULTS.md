# Fixed Breuer (BLP) non-Markovianity, existing data only

Scripts: `py/blp_fixed.py` (writes `NonMark_v2.pkl`, `py/out/blp_fixed.csv`, figures),
`py/blp_fixed.jl` (Julia port, same steps), `py/plot_article_fig_blp_v2.py` (article figure with
the new values, from `03_plot_violin.ipynb`).

## The three steps

1. **One time window per Q.** Every pair is evaluated on the same times: where the four basis
   states B1 to B4 all have data. Before, each pair was cut to its own shortest series.
2. **Best pair of initial states.** The qubit evolution is affine, b(t) = A(t) b₀ + c(t), and the
   four basis states give A and c. For the orthogonal pure pair b₀ = ±n, D(t) = |A(t) n|. We
   maximise over n instead of trying the 91 pairs of stored states. (Check: the 10 dodecahedron
   trajectories are reproduced from A, c to 2e-5 to 2.5e-4.)
3. **Noise threshold ε = 1e-4.** Rises of D smaller than ε are not counted (about the data precision).

The integral of σ over σ > 0 is still the sum of rises of D between samples, which is exact for the
sampled D. Error bar = spread over {ε = 0, ε = 1e-4} × {every sample, every 2nd sample}.

## Results

| Q | window | current N | fixed N | range |
|---|---|---|---|---|
| 316 | 0.06 – 25.2 | 1.5e-4 | < 1e-4 | 0 – 2.1e-4 |
| 100 | 0.05 – 23.9 | 1.3e-2 | 1.2e-2 | 1.1e-2 – 1.3e-2 |
| 31.6 | 0.05 – 8.8 | 5.9e-5 | < 1e-4 | 0 – 6.5e-5 |
| 10 | 0.05 – 2.8 | 7.1e-4 | 1.1e-4 | 1.1e-4 – 7.6e-4 |
| 3.16 | 0.05 – 0.75 | 3.5e-2 | 3.8e-2 | ±0.1 % |
| 1 | 0 – 0.110 | 4.2e-2 | 6.5e-3 | ±1 % |
| 0.316 | 0 – 0.050 | 3.0e-3 | 2.3e-4 | ±0.1 % |
| 0.1 | 0 – 0.064 | 6.4e-4 | 1.3e-3 | ±0.3 % |

Figures: `py/out/blp_old_vs_fixed.pdf`, `py/out/SB_SID_LINDBLAD_KOSSAK_CONSTR_TSSOS_vs_Bench_color-blind_BLPv2.pdf`.

## What produces N at high Q (no recurrence)

At Q = 100 and 316, D(t) of the best pair decays smoothly with a small oscillation at 2ν
(50.3 rad per unit time) on top (`py/out/blp_2nu_oscillation.pdf`). The oscillation is there from
the first sample with the same frequency, so it is not a bath recurrence. It looks like the
counter-rotating (non-secular) part of the spin-boson coupling. Its slope is about equal to the
decay slope, so D only turns upward once the decay slows down: after t ≈ 17 at Q = 100. At
Q = 316 and 31.6 the upward bits are below 1e-4.

## Caveats that only new data can remove (request to Kurt)

- **Window length decides N at Q ≤ 1 and at Q = 100.** Values grow with a longer record
  (Q = 0.1: 2e-4 up to t = 0.025, 1.3e-3 up to t = 0.064). For Q ≤ 1 the four basis states have
  different end times now. Ask: B1 to B4 run to the same end time, as long as the other states.
- **Early times missing for Q ≥ 3.16.** Data start at t ≈ 0.05 (the initial slip was cut). Ask:
  B1 to B4 from t = 0 to 0.06 for γ = 0.079477, 0.25133, 0.79477, 2.5133, 7.9477, same dt.
- **Q = 316 sampling.** dt = 0.02 gives only 6 samples per 2ν period. Ask (optional): B1 to B4
  at Q = 316 with dt ≤ 0.005.
