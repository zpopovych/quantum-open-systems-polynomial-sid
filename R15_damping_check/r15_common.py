"""Shared helpers for the R15 damping-only master-equation check (Python, no Julia).

Conventions follow LiPoSID.read_timeevolution:
    rho = [[p0, s_re + i s_im], [s_re - i s_im, 1 - p0]]
In this representation population decays into index 0 and rho[0,1] rotates as exp(-i w t),
so the damping-only master equation is
    H = (w/2) diag(1, -1),   L = sqrt(gamma) |0><1|.
Arrays written by Julia's HDF5.jl are column-major, so h5py returns them transposed:
every matrix read from a fit file is transposed back with `.T`.
"""
import h5py
import numpy as np
from scipy.linalg import expm

REPO = __file__.rsplit("/R15_damping_check/", 1)[0]
DATA = f"{REPO}/DATA/ALL_GAMMAS_B4_D10.h5"
LINDBLAD_FIT = f"{REPO}/E_LINDBLAD4_CONSTR_TSSOS_treshold_1e-9_FROB_QO_2026-Jun-14_at_17-08.h5"
KOSSAK_FIT = f"{REPO}/E_KOSSAK_CONSTR_TSSOS_treshold_1e-15_FROB_QO_2026-Jun-14_at_16-44.h5"

# The five largest Q values, Q = w0/gamma with w0 = 8*pi
GAMMAS = ["0.079477", "0.25133", "0.79477", "2.5133", "7.9477"]
Q_LABELS = ["316", "100", "31.6", "10", "3.16"]
W0 = 8 * np.pi
OMEGA_C = 10 * W0  # cutoff Om = 10 w = 80 pi

# Kurt's 23 Sep 2026 window: start at sample n = 11, end at these samples (1-based, MATLAB)
N_START = 11
N_END = [1001, 2801, 2516, 1591, 1258]
DODECA = [f"D{i}" for i in range(1, 11)]  # Kurt's dodeca states 1 3 5 7 9 10 11 13 15 20
BASIS = [f"B{i}" for i in range(1, 5)]

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def kurt_frequency(gamma):
    """Lamb-shifted frequency from Kurt's email; Om/w uses the bare w = 8 pi."""
    x = OMEGA_C / W0
    return W0 + 0.0155 * gamma / (2 * np.pi) * (x + np.log(x - 1))


def read_evolution(state, gamma, path=DATA):
    with h5py.File(path, "r") as f:
        g = f[state][gamma]
        t = np.ravel(g["t"][()])
        p0 = np.ravel(g["p0"][()])
        s = np.ravel(g["s_re"][()]) + 1j * np.ravel(g["s_im"][()])
    rho = np.empty((len(t), 2, 2), dtype=complex)
    rho[:, 0, 0] = p0
    rho[:, 1, 1] = 1 - p0
    rho[:, 0, 1] = s
    rho[:, 1, 0] = np.conj(s)
    return t, rho


def trace_distance(r1, r2):
    """0.5 * ||r1 - r2||_1 for stacks of 2x2 Hermitian matrices."""
    ev = np.linalg.eigvalsh(r1 - r2)
    return 0.5 * np.abs(ev).sum(axis=-1)


def liouvillian(H, Ls):
    """Row-major vectorisation: vec(A X B) = kron(A, B.T) vec(X)."""
    I = np.eye(H.shape[0])
    Lv = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for L in Ls:
        LdL = L.conj().T @ L
        Lv += np.kron(L, L.conj()) - 0.5 * (np.kron(LdL, I) + np.kron(I, LdL.T))
    return Lv


def propagate(H, Ls, rho0, t):
    """Exact propagation on the (possibly non-uniform) grid t, starting at t[0]."""
    Lv = liouvillian(H, Ls)
    out = np.empty((len(t), 2, 2), dtype=complex)
    v = rho0.reshape(-1)
    out[0] = rho0
    dts = np.diff(t)
    cache = {}
    for k, dt in enumerate(dts, start=1):
        key = round(dt, 12)
        if key not in cache:
            cache[key] = expm(Lv * dt)
        v = cache[key] @ v
        out[k] = v.reshape(2, 2)
    return out


def damping_model(w, gamma):
    H = 0.5 * w * SZ
    L = np.sqrt(gamma) * np.array([[0, 1], [0, 0]], dtype=complex)
    return H, [L]


def lindblad_fit(gamma, path=LINDBLAD_FIT):
    with h5py.File(path, "r") as f:
        g = f[gamma]
        H = g["H"][()].T
        Ls = [g[f"J{k}"][()].T for k in range(1, 5)]
        status = g["status"][()].decode()
    return H, Ls, status


def kossak_fit(gamma, path=KOSSAK_FIT):
    """Same construction as LiPoSID.get_lindblad_operators with F = sigma/2."""
    with h5py.File(path, "r") as f:
        g = f[gamma]
        H = g["H"][()].T
        # Used as read (no transpose): this is the orientation that reproduces the Julia
        # rescore E_KOSSAK_TRACEDIST_ALLSTATES_2026-07-12.h5 (checked in check_propagation.py)
        C = g["C"][()]
        status = g["status"][()].decode()
    F = [SX / 2, SY / 2, SZ / 2]
    lam, V = np.linalg.eigh(C)
    Ls = [np.sqrt(l) * sum(V[j, i] * F[j] for j in range(3)) for i, l in enumerate(lam) if l > 1e-10]
    return H, Ls, status


def model_params(H, Ls):
    """Effective frequency, population decay rate and extra coherence decay of a model."""
    Lv = liouvillian(H, Ls)
    # rho[0,1] is element 1 of the row-major vector; its eigen-rate is the coherence pole
    ev = np.linalg.eigvals(Lv)
    coh = ev[np.argmin(ev.imag)]  # pole with exp(-i w t)
    pop = ev[np.argmin(np.abs(ev.imag) + 1e3 * (np.abs(ev) < 1e-9))]
    return -coh.imag, -pop.real, -coh.real
