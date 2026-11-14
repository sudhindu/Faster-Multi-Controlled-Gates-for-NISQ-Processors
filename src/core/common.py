"""
common.py — shared physical model for the level-selective qubit–qutrit gates
=============================================================================

Device model (two transmons, each truncated to N levels, default N = 4 so that leakage
to |3> is included):

    H_lab(t)/2π = Σ_q [ ω_q n_q + (α_q/2) n_q (n_q − 1) ]                      (bare transmons)
                + g(t) (a_c + a_c†)(a_t + a_t†)                               (tunable coupler, full form, no RWA)
                + Ω(t) (a_t + a_t†)                                           (microwave drive on the target)
                + Δ(t) n_t                                                    (flux pulse on the target frequency)

    Ω(t) = Ω_I(t) cos(ω_d t) − Ω_Q(t) sin(ω_d t)

Units: frequencies in GHz, time in ns; Hamiltonians are stored in rad/ns (factor 2π included).

Gate principle ("tunable-coupler + direct conditional drive"):
  * idle: coupler OFF (g = 0); target flux-parked at 5.20 GHz, control at 5.00 GHz;
  * gate: coupler ramps to g_on ≈ 45–55 MHz (10 ns cos² ramps). The dressed target transition
    j→k then depends on the control level by tens of MHz;
  * a shaped drive at the conditional frequency ω_d (control in level c0) swaps target j<->k;
  * spline-shaped coupler modulation and a small target flux pulse are optimised together
    with the drive so that BOTH populations and phases are correct (phase-aware design).
    The remaining single-qutrit phases are virtual-Z (frame) updates and cost no time.

The three gates:  C^1X_12 (control |1> swaps target 1<->2),
                  C^2X_01 (control |2> swaps target 0<->1),
                  C^2X_12 (control |2> swaps target 1<->2).

Pulse parameter vector x (length 2 + 4K + 4, K = 6 spline knots), identical to the optimiser:
  x[0]          g_on  (MHz)
  x[1]          drive detuning from the dressed conditional frequency (MHz)
  x[2:2+K]      in-phase drive knots       (×20 MHz)
  x[2+K:2+2K]   quadrature drive knots     (×20 MHz)
  x[2+2K:2+3K]  coupler-modulation knots   (×0.5, fractional change of g_on)
  x[2+3K:2+4K]  target flux-detuning knots (×40 MHz)
  x[2+4K:]      four virtual-Z phases (control levels 1,2 ; target levels 1,2), radians
"""
import os, json, itertools
import numpy as np
import qutip as qt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

TWO_PI = 2*np.pi
N = int(os.environ.get("SIM_N", "4"))          # levels per transmon
D = N*N
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- device parameters (GHz)
# device parameters: control 5.000 GHz, target 5.55 GHz (parked at 5.20 GHz for these gates),
# anharmonicities -300 / -320 MHz.
P0 = dict(wc=5.000, wt=5.20, ac=-0.300, at=-0.320)
RC, RD, K = 10.0, 6.0, 6                        # coupler ramp (ns), drive edge (ns), spline knots
GATES = {'C1X12': (1, 1, 2), 'C2X01': (2, 0, 1), 'C2X12': (2, 1, 2)}   # (control level, j, k)

# ---------------------------------------------------------------- operators (numpy and QuTiP)
a = np.diag(np.sqrt(np.arange(1, N)), 1)
I_N = np.eye(N)
AC, AT = np.kron(a, I_N), np.kron(I_N, a)
NC, NT = AC.conj().T @ AC, AT.conj().T @ AT
I_D = np.eye(D)
CPL_RWA = AC.conj().T @ AT + AC @ AT.conj().T          # exchange coupling (rotating frame)
HXt = 0.5*(AT + AT.conj().T)                            # drive quadratures in rotating frame
HYt = 0.5j*(AT - AT.conj().T)
def Q(m): return qt.Qobj(m, dims=[[N, N], [N, N]])

def idx(c, t): return c*N + t
LOGICAL = [idx(c, t) for c in range(3) for t in range(3)]     # the 9 logical states |c,t>, c,t in {0,1,2}
KEYS9 = [(c, t) for c in range(3) for t in range(3)]

def h0_lab(p):
    """bare (coupler-off) lab-frame Hamiltonian, rad/ns, diagonal"""
    return TWO_PI*(p['wc']*NC + 0.5*p['ac']*NC@(NC-I_D) + p['wt']*NT + 0.5*p['at']*NT@(NT-I_D))

def h0_rot(p, wd, g=0.0):
    """rotating frame at wd for both transmons, RWA exchange coupling g (GHz)"""
    return TWO_PI*((p['wc']-wd)*NC + 0.5*p['ac']*NC@(NC-I_D) + (p['wt']-wd)*NT
                   + 0.5*p['at']*NT@(NT-I_D) + g*CPL_RWA)

def dressed_order(H):
    E, V = np.linalg.eigh(H); order = {}; used = set()
    for c in range(N):
        for t in range(N):
            w = np.abs(V[idx(c, t), :])**2*np.array([0 if j in used else 1 for j in range(D)])
            k = int(np.argmax(w)); used.add(k); order[(c, t)] = k
    return E, order

def cond_freq(p, gate, g):
    """dressed target transition frequency (GHz) j->k with the control in level c0, coupler at g (GHz)"""
    c0, j, k = GATES[gate]
    E, o = dressed_order(h0_rot(p, 0.0, g))
    return (E[o[(c0, k)]] - E[o[(c0, j)]])/TWO_PI

# ---------------------------------------------------------------- pulses
def envelope(T, dt, rise):
    n = int(round(T/dt)); t = (np.arange(n)+0.5)*dt
    s = np.ones(n)
    r = t < rise; s[r] = np.sin(0.5*np.pi*t[r]/rise)**2
    r = t > T-rise; s[r] = np.sin(0.5*np.pi*(T-t[r])/rise)**2
    return t, s

def _spline_basis(n, K):
    u = (np.arange(n)+0.5)/n; kn = np.linspace(0, 1, K); B = np.zeros((n, K))
    for i in range(K):
        e = np.zeros(K); e[i] = 1; B[:, i] = CubicSpline(kn, e, bc_type='natural')(u)
    return B

def pulses(T, x, dt):
    """Sampled control waveforms at step mid-points t=(i+1/2)dt.
    Returns dict with t (ns) and, in rad/ns: fg (coupler g(t)), fI, fQ (drive quadratures), fD (target detuning)."""
    x = np.asarray(x)
    t, sg = envelope(T, dt, RC)
    m = (t >= RC) & (t <= T-RC); tt = t[m]-RC; Td = T-2*RC
    h = np.sin(np.pi*tt/Td)**2
    s = np.ones(tt.size)
    r = tt < RD; s[r] = np.sin(0.5*np.pi*tt[r]/RD)**2
    r = tt > Td-RD; s[r] = np.sin(0.5*np.pi*(Td-tt[r])/RD)**2
    B = _spline_basis(tt.size, K)
    cI, cQ, cM, cD = x[2:2+K], x[2+K:2+2*K], x[2+2*K:2+3*K], x[2+3*K:2+4*K]
    n = len(t); sI = np.zeros(n); sQ = np.zeros(n); mod = np.zeros(n); det = np.zeros(n)
    sI[m] = s*(B@cI); sQ[m] = s*(B@cQ); mod[m] = 0.5*h*(B@cM); det[m] = 40.0*h*(B@cD)
    return dict(t=t, fg=TWO_PI*x[0]*1e-3*sg*(1+mod), fI=TWO_PI*20e-3*sI, fQ=TWO_PI*20e-3*sQ, fD=TWO_PI*1e-3*det)

def drive_freq(gate, x, p=P0):
    return cond_freq(p, gate, x[0]*1e-3) + x[1]*1e-3

def load_params(path=None):
    path = path or os.path.join(HERE, 'gate_params.json')
    return json.load(open(path))

# ---------------------------------------------------------------- fast rotating-frame propagation (numpy)
def propagate(H0, Hks, fs, dt):
    n = len(fs[0]); H = np.broadcast_to(H0, (n, D, D)).astype(complex).copy()
    for Hk, f in zip(Hks, fs): H += f[:, None, None]*Hk[None]
    E, V = np.linalg.eigh(H)
    U = (V*np.exp(-1j*E*dt)[:, None, :]) @ np.conj(np.swapaxes(V, 1, 2))
    while U.shape[0] > 1:
        if U.shape[0] % 2: U = np.concatenate([U, np.eye(D, dtype=complex)[None]], 0)
        U = U[1::2] @ U[0::2]
    return U[0]

def unitary_rwa(gate, T, x, p=P0, dt=0.05, err=None):
    """Rotating-frame (RWA drive) propagator in the interaction picture of the idle Hamiltonian.
    err: optional dict(dfc, dft [MHz], gsc [fraction], goff [MHz], amp [fraction], dwd [MHz drive re-tune])."""
    err = err or {}
    pu = pulses(T, x, dt)
    wd = drive_freq(gate, x, p) + err.get('dwd', 0)*1e-3
    pe = dict(p); pe['wc'] += err.get('dfc', 0)*1e-3; pe['wt'] += err.get('dft', 0)*1e-3
    H0 = h0_rot(pe, wd)
    fg = pu['fg']*(1+err.get('gsc', 0)) + TWO_PI*err.get('goff', 0)*1e-3
    amp = 1+err.get('amp', 0)
    U = propagate(H0, [CPL_RWA, HXt, HYt, NT.astype(complex)], [fg, amp*pu['fI'], amp*pu['fQ'], pu['fD']], dt)
    E0 = np.real(np.diag(h0_rot(p, wd)))                 # nominal frame (calibrated reference)
    return np.diag(np.exp(1j*E0*T)) @ U

# ---------------------------------------------------------------- metrics
def ideal_perm(gate):
    """ideal permutation sigma on the 9 logical states"""
    c0, j, k = GATES[gate]; sig = {}
    for (c, t) in KEYS9:
        t2 = t
        if c == c0: t2 = k if t == j else (j if t == k else t)
        sig[(c, t)] = (c, t2)
    return sig

def phase_vector(ph):
    """virtual-Z correction e^{i(a_c + b_t)} on the 9 logical states; ph=(a1,a2,b1,b2)"""
    return np.array([np.exp(1j*([0, ph[0], ph[1]][c] + [0, ph[2], ph[3]][t])) for (c, t) in KEYS9])

def ideal_unitary9(gate, ph=None):
    sig = ideal_perm(gate); U = np.zeros((9, 9), complex)
    for i, kk in enumerate(KEYS9): U[KEYS9.index(sig[kk]), i] = 1
    if ph is not None: U = np.diag(np.conj(phase_vector(ph))) @ U
    return U

def fpro(L, gate, ph):
    """process fidelity |Tr(U_ideal^† D L)|^2/81 with virtual-Z vector D (L = 9x9 logical block)"""
    Uc = np.diag(phase_vector(ph)) @ L
    return abs(np.trace(ideal_unitary9(gate).conj().T @ Uc))**2/81

def best_phases(L, gate, ph0=None):
    f = lambda ph: -fpro(L, gate, ph)
    starts = ([np.asarray(ph0)] if ph0 is not None else []) + [np.zeros(4)] + \
             [np.random.RandomState(s).uniform(-np.pi, np.pi, 4) for s in range(4)]
    best = None
    for s0 in starts:
        r = minimize(f, s0, method='BFGS')
        if best is None or r.fun < best.fun: best = r
    return best.x, -best.fun

def populations(L, gate):
    sig = ideal_perm(gate)
    per = {kk: float(abs(L[KEYS9.index(sig[kk]), i])**2) for i, kk in enumerate(KEYS9)}
    leak = {kk: float(1-np.sum(abs(L[:, i])**2)) for i, kk in enumerate(KEYS9)}
    return per, leak

def avg_gate_fidelity(L, gate, ph):
    """average fidelity over Haar-random pure states of the 9-dim logical space (Nielsen formula with leakage)"""
    M = np.diag(phase_vector(ph)) @ L
    Ui = ideal_unitary9(gate); d = 9
    A = Ui.conj().T @ M
    return float((np.real(np.trace(M.conj().T @ M)) + abs(np.trace(A))**2)/(d*(d+1)))

def state_fidelity(L, gate, ph, psi):
    psi = np.asarray(psi, complex); psi /= np.linalg.norm(psi)
    out = np.diag(phase_vector(ph)) @ L @ psi
    return float(abs(np.vdot(ideal_unitary9(gate) @ psi, out))**2)

def ket9(c, t):
    """product state of control c (3-vector) and target t (3-vector) in the 9-dim logical ordering"""
    return np.kron(np.asarray(c, complex), np.asarray(t, complex))
