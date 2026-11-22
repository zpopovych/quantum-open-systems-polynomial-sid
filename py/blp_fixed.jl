# Fixed BLP non-Markovianity (Julia port of py/blp_fixed.py, grid search over pairs only).
# Step 1: one time window per Q (where B1..B4 all have data).
# Step 2: best orthogonal pure pair ±n via the affine Bloch map from B1=+z, B2=-z, B3=+x, B4=-y.
# Step 3: rises of D smaller than ε are not counted.
# Usage from the repo root:  julia py/blp_fixed.jl
using HDF5, LinearAlgebra

function bloch_series(file, state, γ)
    h5open(file, "r") do f
        g = f[state][γ]
        p0 = vec(read(g["p0"])); sre = vec(read(g["s_re"])); sim = vec(read(g["s_im"]))
        [[2sre[k], -2sim[k], 2p0[k] - 1] for k in eachindex(p0)]
    end
end

"Sum of rises of D; rises and dips smaller than ε are ignored (ε = 0 gives sum(dD[dD .> 0]))."
function rises(D, ε = 0.0)
    ε == 0 && return sum(max.(diff(D), 0.0))
    tot = 0.0; lo = hi = D[1]; up = 0            # 0 undecided, 1 rising, -1 falling
    for x in D[2:end]
        if up == 0
            x - lo >= ε ? (up = 1; hi = x) : (x < lo && (lo = x))
        elseif up == 1
            if x > hi; hi = x
            elseif hi - x >= ε; tot += hi - lo; up = -1; lo = x; end
        else
            if x < lo; lo = x
            elseif x - lo >= ε; up = 1; hi = x; end
        end
    end
    up == 1 ? tot + hi - lo : tot
end

function blp_fixed(file, γ; ε = 1e-4, ngrid = 20000)
    B = [bloch_series(file, s, γ) for s in ("B1", "B2", "B3", "B4")]
    n = minimum(length.(B))                                        # Step 1
    A = map(1:n) do k                                              # Step 2
        c = (B[1][k] + B[2][k]) / 2
        hcat(B[3][k] - c, -(B[4][k] - c), (B[1][k] - B[2][k]) / 2)
    end
    best = (-1.0, [0.0, 0.0, 1.0])
    for i in 1:ngrid                                               # Fibonacci sphere
        z = 1 - (2i - 1) / ngrid; φ = π * (1 + √5) * i; r = sqrt(1 - z^2)
        v = [r * cos(φ), r * sin(φ), z]
        N0 = rises([norm(Ak * v) for Ak in A])
        N0 > best[1] && (best = (N0, v))
    end
    D = [norm(Ak * best[2]) for Ak in A]
    return rises(D, ε), best[1], best[2]                           # Step 3
end

file = "DATA/ALL_GAMMAS_B4_D10.h5"
γs = ["0.079477", "0.25133", "0.79477", "2.5133", "7.9477", "25.133", "79.477", "251.33"]
Qs = [316, 100, 31.6, 10, 3.16, 1, 0.316, 0.1]
for (γ, Q) in zip(γs, Qs)
    N, N0, v = blp_fixed(file, γ)
    println(rpad("Q = $Q", 12), "N_fixed = ", round(N, sigdigits = 3), "   (ε = 0: ", round(N0, sigdigits = 3), ")")
end
