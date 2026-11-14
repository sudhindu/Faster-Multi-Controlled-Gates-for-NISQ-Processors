"""
noise.py — open-system (Lindblad) model shared by Parts 2, 4 and 5
==================================================================

Master equation (rotating frame at the drive, RWA on the drive; Part 1 shows RWA vs lab frame agree):
    dρ/dt = −i[H(t), ρ] + Σ_k ( C_k ρ C_k† − ½{C_k† C_k, ρ} )

Collapse operators for EACH transmon (levels |0>..|3>):
    C = sqrt(Γ_10) |0><1| ,  sqrt(Γ_21) |1><2| ,  sqrt(Γ_32) |2><3|      (energy relaxation, level by level)
    C = sqrt(2 γ_φ) n̂                                                   (pure dephasing)
  * Γ_10 = 1/T1(1→0), Γ_21 = 1/T1(2→1): the |2> level is given its own, SHORTER lifetime
    (measured reference transmon: 27 µs and 17 µs), Γ_32 = 3 Γ_10 (harmonic-oscillator scaling).
  * pure dephasing via n̂: the coherence between |0> and |n> decays at rate γ_φ n², i.e. |0>-|2>
    coherences dephase 4× faster than |0>-|1> (charge/flux-noise scaling of higher transmon levels).
Quasi-static control errors (frequency offsets from flux noise, coupler-strength and drive-amplitude
errors, residual coupling) enter through common.unitary-style error dictionaries.

The gate channel is the 256×256 superoperator obtained by exact exponentiation of the Liouvillian on each
piecewise-constant time step (dt = 0.25 ns, the grid the pulses were optimised on), multiplied in time order.
It was validated against QuTiP `propagator(H, T, c_ops)` (see p2_noisy_gates.py --validate).
"""
import numpy as np, itertools
from scipy.linalg import expm
from common import *

def coherence(T1_10=27.0, T1_21=17.0, Tphi=40.0):
    """times in µs -> rates in 1/ns"""
    return dict(G10=1/(T1_10*1e3), G21=1/(T1_21*1e3), G32=3/(T1_10*1e3), gphi=1/(Tphi*1e3), T1_10=T1_10, T1_21=T1_21, Tphi=Tphi)

MEASURED = coherence(27.0, 17.0, 40.0)        # reference device (measured T1; Tphi assumed)
IMPROVED = coherence(100.0, 60.0, 150.0)      # state-of-the-art transmon qutrits
IBM_LIKE = coherence(200.0, 100.0, 200.0)     # qubit coherence comparable to IBM Heron devices (|2> at half T1)

def _sup_H(H): return -1j*(np.kron(I_D, H) - np.kron(H.T, I_D))
def _sup_D(c):
    cdc = c.conj().T @ c
    return np.kron(c.conj(), c) - 0.5*np.kron(I_D, cdc) - 0.5*np.kron(cdc.T, I_D)

def collapse_ops(coh):
    ops = []; E = lambda m, n: np.outer(I_N[m], I_N[n]); nop = np.diag(np.arange(N)).astype(float)
    rates = {1: coh['G10'], 2: coh['G21'], 3: coh['G32']}
    for which in (0, 1):
        emb = (lambda m: np.kron(m, I_N)) if which == 0 else (lambda m: np.kron(I_N, m))
        for nlev, r in rates.items():
            if nlev < N: ops.append(np.sqrt(r)*emb(E(nlev-1, nlev)))
        ops.append(np.sqrt(2*coh['gphi'])*emb(nop))
    return ops

_LC, _LX, _LY, _LN = _sup_H(CPL_RWA), _sup_H(HXt), _sup_H(HYt), _sup_H(NT.astype(complex))

def gate_superop(gate, T, x, coh=None, err=None, dt=0.25, chunk=120):
    """256x256 superoperator (column-stacking vec) in the nominal idle interaction picture"""
    err = err or {}
    pu = pulses(T, x, dt)
    wd = drive_freq(gate, x) + err.get('dwd', 0)*1e-3
    pe = dict(P0); pe['wc'] += err.get('dfc', 0)*1e-3; pe['wt'] += err.get('dft', 0)*1e-3
    L0 = _sup_H(h0_rot(pe, wd))
    if coh is not None:
        for c in collapse_ops(coh): L0 = L0 + _sup_D(c)
    fg = pu['fg']*(1+err.get('gsc', 0)) + TWO_PI*err.get('goff', 0)*1e-3
    amp = 1+err.get('amp', 0)
    S = np.eye(D*D, dtype=complex); n = len(fg)
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0+chunk))
        Ls = (L0[None] + fg[sl, None, None]*_LC + amp*pu['fI'][sl, None, None]*_LX
              + amp*pu['fQ'][sl, None, None]*_LY + pu['fD'][sl, None, None]*_LN)
        for E_ in expm(dt*Ls): S = E_ @ S
    E0 = np.real(np.diag(h0_rot(P0, drive_freq(gate, x))))          # nominal calibrated frame
    ph = np.exp(1j*E0*T); F = np.kron(ph.conj(), ph)                  # rho_ij -> e^{i(E_i-E_j)T} rho_ij
    return F[:, None]*S

def logical_channel(S, ph=None):
    """restrict to the 9 logical states, apply virtual-Z vector; returns S4[c,d,a,b]: rho'_{cd} = Σ S4 rho_{ab}"""
    Dv = phase_vector(ph) if ph is not None else np.ones(9)
    S4 = np.zeros((9, 9, 9, 9), complex)
    for (c, io), (d, jo) in itertools.product(enumerate(LOGICAL), repeat=2):
        row = jo*D + io
        for (a_, ki), (b_, li) in itertools.product(enumerate(LOGICAL), repeat=2):
            S4[c, d, a_, b_] = S[row, li*D+ki]
        S4[c, d] *= Dv[c]*np.conj(Dv[d])
    return S4

def channel_metrics(S4, gate):
    """per-input populations, leakage, entanglement (process) fidelity, average fidelity"""
    U = ideal_unitary9(gate); d = 9
    sig = ideal_perm(gate)
    per = {}; ret = []
    for a_, kk in enumerate(KEYS9):
        o = KEYS9.index(sig[kk]); per[kk] = float(np.real(S4[o, o, a_, a_]))
        ret.append(float(np.real(sum(S4[c, c, a_, a_] for c in range(9)))))
    Fe = 0
    for a_ in range(d):
        for b_ in range(d):
            Fe += U[:, a_].conj() @ S4[:, :, a_, b_] @ U[:, b_]
    Fe = float(np.real(Fe))/d**2; pret = float(np.mean(ret))
    return dict(per_input={"|%d,%d>" % k: per[k] for k in KEYS9}, pop_mean=float(np.mean(list(per.values()))),
                pop_min=float(min(per.values())), leak=1-pret, F_pro=Fe, F_avg=(d*Fe+pret)/(d+1))

def rwa_phases(gate, T, x):
    L = unitary_rwa(gate, T, x, dt=0.05)[np.ix_(LOGICAL, LOGICAL)]
    return best_phases(L, gate, x[2+4*K:])[0]

def single_idle(t, coh):
    """3-level idle channel of a spectator qutrit for time t (ns): S1[i,j,k,l] (rho'_ij = Σ S1 rho_kl)"""
    if coh is None or t <= 0: return np.einsum('ik,jl->ijkl', np.eye(3), np.eye(3)).astype(complex)
    E = lambda m, n: np.outer(np.eye(3)[m], np.eye(3)[n])
    cs = [np.sqrt(coh['G10'])*E(0, 1), np.sqrt(coh['G21'])*E(1, 2), np.sqrt(2*coh['gphi'])*np.diag([0., 1., 2.])]
    I3 = np.eye(3); Lm = np.zeros((9, 9), complex)
    for c in cs:
        cdc = c.conj().T@c; Lm += np.kron(c.conj(), c) - 0.5*np.kron(I3, cdc) - 0.5*np.kron(cdc.T, I3)
    S9 = expm(Lm*t)
    return np.array([[[[S9[j*3+i, l*3+k] for l in range(3)] for k in range(3)] for j in range(3)] for i in range(3)])
