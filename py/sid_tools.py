"""Python helpers that mirror LiPoSID.jl for reading spin-boson data and
evolving the TSSOS-identified Markovian models (no Julia needed).

Conventions (same as LiPoSID.read_timeevolution):
    rho = [[p0, s], [conj(s), 1 - p0]],  s = s_re + i s_im
    excited population = rho[1,1] = 1 - p0   (state B2 = |1><1| is "excited")
    coherence          = rho[0,1]
"""
import h5py
import numpy as np
from scipy.linalg import expm

GAMMAS = ["0.079477", "0.25133", "0.79477", "2.5133",
          "7.9477", "25.133", "79.477", "251.33"]
OMEGA = 8 * np.pi  # bare qubit frequency, Q = OMEGA / gamma
ALL_STATES = [f"B{i}" for i in range(1, 5)] + [f"D{i}" for i in range(1, 11)]

EVOL_DATA = "DATA/ALL_GAMMAS_B4_D10.h5"
KOSSAK_FITTED = "E_KOSSAK_CONSTR_TSSOS_treshold_1e-15_FROB_QO_2026-Jun-14_at_16-44.h5"
LINDBLAD_FITTED = "E_LINDBLAD4_CONSTR_TSSOS_treshold_1e-9_FROB_QO_2026-Jun-14_at_17-08.h5"

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, 1j], [-1j, 0]], dtype=complex)  # sign as in the notebooks
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
F_KOSSAK = [SX / 2, SY / 2, SZ / 2]


def Q_of(gamma):
    return OMEGA / float(gamma)


def read_timeevolution(file_name, state, gamma):
    """Return t (N,), rho (N,2,2) for one initial state and coupling."""
    with h5py.File(file_name, "r") as f:
        g = f[state][gamma]
        t = np.ravel(g["t"][()]).astype(float)
        p0 = np.ravel(g["p0"][()])
        s = np.ravel(g["s_re"][()]) + 1j * np.ravel(g["s_im"][()])
    rho = np.empty((len(t), 2, 2), dtype=complex)
    rho[:, 0, 0] = p0
    rho[:, 0, 1] = s
    rho[:, 1, 0] = np.conj(s)
    rho[:, 1, 1] = 1 - p0
    return t, rho


def lindblad_ops_from_kossak(C, basis_ops=F_KOSSAK, tol=1e-10):
    """Same as LiPoSID.get_lindblad_operators: L_i = sqrt(l_i) sum_j v_ji f_j."""
    w, v = np.linalg.eig(C)
    ops = []
    for i in range(len(w)):
        if w[i].real > tol:
            ops.append(sum(np.sqrt(w[i]) * v[j, i] * basis_ops[j]
                           for j in range(len(basis_ops))))
    return ops


def read_model(kind, gamma):
    """kind = 'kossak' or 'lindblad'. Returns H, [L_k], status.

    Matrices were written by Julia (column-major), so h5py returns them
    transposed; .T restores the Julia orientation.
    """
    fn = KOSSAK_FITTED if kind == "kossak" else LINDBLAD_FITTED
    with h5py.File(fn, "r") as f:
        g = f[gamma]
        H = g["H"][()].T.astype(complex)
        status = g["status"][()].decode()
        if kind == "kossak":
            ops = lindblad_ops_from_kossak(g["C"][()].T.astype(complex))
        else:
            ops = [g[f"J{k}"][()].T.astype(complex) for k in range(1, 5)]
    return H, ops, status


def liouvillian(H, ops):
    """Superoperator for column-stacked vec(rho)."""
    I = np.eye(H.shape[0])
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for J in ops:
        JdJ = J.conj().T @ J
        L += np.kron(J.conj(), J) - 0.5 * (np.kron(I, JdJ) + np.kron(JdJ.T, I))
    return L


def evolve(H, ops, rho0, t):
    """Exact propagation rho(t) = exp(L (t - t0)) rho0 on the grid t."""
    return evolve_L(liouvillian(H, ops), rho0, t)


def simpson_corrected_liouvillian(L, dt):
    """Undo the frequency bias of the double-step Simpson objective.

    The loss enforces rho_j - rho_{j-2} = (dt/3) L (rho_{j-2} + 4 rho_{j-1} + rho_j),
    so a fitted eigenvalue mu of L corresponds to a true eigenvalue lam with
    Z = exp(lam dt) solving (1 - a) Z^2 - 4 a Z - (1 + a) = 0, a = mu dt / 3.
    Taking the root near Z = 1 and lam = log(Z) / dt gives the generator whose
    exact evolution matches the samples that the fit saw.
    """
    w, V = np.linalg.eig(L)
    a = w * dt / 3
    Z = (4 * a + np.sqrt(16 * a**2 + 4 * (1 - a) * (1 + a))) / (2 * (1 - a))
    return V @ np.diag(np.log(Z) / dt) @ np.linalg.inv(V)


def evolve_L(L, rho0, t):
    v0 = rho0.reshape(-1, order="F")
    return np.array([(expm(L * (tk - t[0])) @ v0).reshape(2, 2, order="F")
                     for tk in t])


def trace_distance(r1, r2):
    """0.5 * ||r1 - r2||_1 for stacks of Hermitian 2x2 matrices."""
    A = r1 - r2
    A = 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))
    return 0.5 * np.abs(np.linalg.eigvalsh(A)).sum(axis=-1)
