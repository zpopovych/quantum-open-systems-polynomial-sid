# Check of the BLP non-Markovianity numbers (`NonMarkovianity.ipynb`)

Run: `python py/blp_nonmarkovianity_check.py` (needs `h5py numpy scipy`, optional `matplotlib`).
Output: `py/out/blp_check.csv`, `py/out/blp_check.pdf`.

## What the notebook does

```julia
dD = diff([TrDist(ρ₁, ρ₂) for (ρ₁, ρ₂) in zip(ρs₁, ρs₂)])
append!(N, sum(dD[dD .> 0]))        # then maximum(N) over the 91 pairs of 14 stored states
```

`sum(dD[dD .> 0])` is **not** a crude Riemann sum: it is the exact value of ∫_{σ>0} σ dt for the
piecewise-linear D(t) through the samples (a sum of D(t_end) − D(t_start) over increasing steps),
and a lower bound on the continuous value. On every 2nd / 4th sample it changes by < 1 % for
γ ≥ 7.95 and for the Lorentzian set, so the step size is not the problem. The Python port
reproduces `NonMark.pkl` to all digits.

## What is wrong

| γ | notebook N (pair, window end) | 91 pairs, common window | optimum over antipodal pure pairs | optimum, wiggles < 1e-4 ignored |
|---|---|---|---|---|
| 0.079477 | 1.54e-4 (D7-D8) | 1.54e-4 | 2.11e-4 | **0** |
| 0.25133 | 1.29e-2 (B3-D8) | 1.29e-2 | 1.34e-2 | 1.21e-2 |
| 0.79477 | 5.91e-5 (D5-D6) | 5.91e-5 | 6.52e-5 | **0** |
| 2.5133 | 7.10e-4 (D5-D6) | 7.10e-4 | 7.63e-4 | 1.1e-4 |
| 7.9477 | 3.46e-2 (B4-D6) | 3.46e-2 | 3.85e-2 | 3.85e-2 |
| 25.133 | **4.19e-2** (B2-B3, t ≤ 0.163) | 3.64e-3 (t ≤ 0.110) | 6.46e-3 | 6.46e-3 |
| 79.477 | **3.02e-3** (D3-D5, t ≤ 0.100) | 4.69e-5 (t ≤ 0.050) | 2.30e-4 | 2.30e-4 |
| 251.33 | 6.40e-4 (B1-B3) | 6.36e-4 | 1.28e-3 | 1.28e-3 |

1. **Different time windows per pair (largest error).** For γ = 25.133, 79.477, 251.33 the 14
   series have different lengths (e.g. 626 … 3517 samples at γ = 79.477). `zip` cuts each pair to
   its own shortest series, so the maximum is taken over pairs integrated over different windows
   and is won by whichever pair has the longest data. At γ = 25.133 all of 4.19e-2 comes from
   t ∈ (0.110, 0.163], where the basis states B1, B4 have no data; at γ = 79.477 all of 3.02e-3
   comes from t ∈ (0.05, 0.10]. Fix: truncate all states to one common window per γ.
2. **Maximisation over only 91 fixed pairs.** The qubit map is affine, b(t) = A(t) b₀ + c(t)
   (checked: the 10 dodecahedron states are reproduced from the 4 basis states to 2e-5 … 2.5e-4),
   so D(t) = |A(t) n| for the antipodal pair b₀ = ±n, which is where the optimum lies
   (Wißmann et al., PRA 86, 062108 (2012)). Maximising over n raises N by 5–100 %, by ×2 at
   γ = 251.33 and ×5 at γ = 79.477 (same window).
3. **Noise counted as revivals.** At γ = 0.079 and 0.79 the "revivals" are step-to-step zig-zags
   of 1e-6 … 5e-5 (below the 2e-5 … 1e-4 data precision); they vanish on every 2nd/4th sample or
   with a 1e-4 hysteresis. These two N values are zero within numerical precision.
4. **Missing start of the evolution.** For γ ≤ 7.9477 the stored series start at t ≈ 0.05, not 0.
   At γ = 7.9477 the optimal pair already has D = 0.86 at the first sample, so any revival in
   [0, 0.05] is not counted.
5. **Late-time rises to check.** At γ = 0.079 and 0.25 all of N comes from t > 17–20, at the end
   of the record. Worth checking whether this is a recurrence of the discretised bath rather than
   physical memory.

The Lorentzian set (`NonMarkovianity_Lorentzian.ipynb`) has equal series lengths and starts at
t = 0; there the stored values are within 2 % of the optimum (0.185 → 0.188, 0.627 → 0.637).

## Recommended Julia fix

```julia
# common window, optimum over antipodal pure pairs b₀ = ±n, noise threshold ε
function blp_N(ρB1, ρB2, ρB3, ρB4; ε = 1e-4, ngrid = 4000)   # B1=+z, B2=-z, B3=+x, B4=-y
    n = minimum(length.((ρB1, ρB2, ρB3, ρB4)))
    b(ρ) = [2real(ρ[1,2]), -2imag(ρ[1,2]), real(ρ[1,1] - ρ[2,2])]
    A = map(1:n) do k
        c = (b(ρB1[k]) + b(ρB2[k])) / 2
        hcat(b(ρB3[k]) - c, -(b(ρB4[k]) - c), (b(ρB1[k]) - b(ρB2[k])) / 2)
    end
    best = 0.0
    for i in 1:ngrid                                           # Fibonacci sphere; refine with Optim if needed
        z = 1 - (2i - 1)/ngrid; φ = π*(1 + √5)*i; r = √(1 - z^2)
        v = [r*cos(φ), r*sin(φ), z]
        D = [norm(Ak * v) for Ak in A]                         # trace distance of the pair ±v
        best = max(best, rises_above(D, ε))
    end
    best
end
```

`rises_above(D, ε)` is the hysteresis sum `npos_eps` in the Python script (with ε = 0 it equals
`sum(dD[dD .> 0])`).
