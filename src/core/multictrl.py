"""
multictrl.py — C^k-NOT and C^k-Ry(theta) from native two-qutrit pulses (schedule-level simulator)
=================================================================================================

Line of n = k+1 qutrits: controls q0..q_{k-1}, target q_k (qubit data in {|0>,|1>}).
fold   : C1X12(q0->q1), C2X12(q_{j-1}->q_j), j=2..k-1           (k-1 two-qutrit pulses)
          after the fold q_{k-1} is in |2>  <=>  all k controls were |1>
C^k-NOT : flip  = C2X01(q_{k-1}->q_k)                             (1 pulse)       -> 2k-1 pulses
C^k-Ry  : flip  = Ry(theta/2)[q_k] . C2X01 . Ry(-theta/2)[q_k] . C2X01        -> 2k two-qutrit pulses
          (+ 2 single-qubit pulses on the target; the first one runs in parallel with the fold)
          Identity: X Ry(-a) X = Ry(a), so the target sees Ry(theta) iff the flip fires, else I.
unfold : mirror of the fold                                        (k-1 pulses)

The two-qutrit pulses do not depend on theta: only the single-qubit angle changes, so no
re-calibration of entangling pulses is needed when theta changes (e.g. amplitude encoding).

Open-system model as circuit.py: Lindblad channel per two-qutrit gate (noise.gate_superop), 3-level
T1/T2 idle channel for every waiting qutrit, T_FLUX idle between gates whose qutrit pair changes,
single-qubit Ry = ideal rotation of the target's {0,1} subspace + idle decay for T_1Q (assumption).
"""
import itertools
import numpy as np
from scipy.optimize import minimize
from common import *
from noise import gate_superop, logical_channel, single_idle, rwa_phases
import circuit as cc

T_FLUX = cc.T_FLUX
T_1Q = 25.0                # ns, assumed single-qubit pulse length on the target (0-1 transition)
P = cc.P

def Ry3(a):
    c, s = np.cos(a/2), np.sin(a/2)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], complex)

def schedule(k, kind, theta=None):
    fold = [('2q', 'C1X12', 0, 1)] + [('2q', 'C2X12', j-1, j) for j in range(2, k)]
    if kind == 'cnot':
        mid = [('2q', 'C2X01', k-1, k)]
    else:
        mid = [('1q', Ry3(theta/2), k, 0.0), ('2q', 'C2X01', k-1, k), ('1q', Ry3(-theta/2), k, T_1Q), ('2q', 'C2X01', k-1, k)]
    if kind != 'cnot':      # first Ry runs in parallel with the fold (target idles there): put it first, zero time
        return [mid[0]] + fold + mid[1:] + fold[::-1]
    return fold + mid + fold[::-1]

def counts(k, kind):
    S = schedule(k, kind, 0.0)
    n2 = sum(1 for o in S if o[0] == '2q'); n1 = sum(1 for o in S if o[0] == '1q')
    T = 0.0; prev = None
    for o in S:
        if o[0] == '2q':
            if prev is not None and prev[0] == '2q' and (prev[2], prev[3]) != (o[2], o[3]): T += T_FLUX
            T += P[o[1]]['T_ns']
        else:
            T += o[3]
        prev = o if o[0] == '2q' or o[3] > 0 else prev
    return dict(n_two_qutrit=n2, n_single=n1, total_pulses=n2+n1, depth=n2 + (1 if n1 > 1 else 0), T_total_ns=T)

# ------------------------------------------------------------------ channels
_CACHE = {}
IDEAL_GATES = set()        # what-if analysis: gates listed here are replaced by a perfect unitary + idle decay
def _ideal_pair(name, coh):
    c0, j, kk = GATES[name]; U = np.zeros((9, 9))
    for a, (c, t) in enumerate(KEYS9):
        t2 = (kk if t == j else (j if t == kk else t)) if c == c0 else t
        U[KEYS9.index((c, t2)), a] = 1
    S4 = np.einsum('ca,db->cdab', U, U).astype(complex)
    if coh is None: return S4
    I1 = single_idle(P[name]['T_ns'], coh)                         # decay of both qutrits during the gate window
    # apply idle after the gate on control (c) and target (t):  rho'_{(ct),(c't')} = sum I1[c,c',k,k'] I1[t,t',l,l'] rho_{(kl),(k'l')}
    E = np.einsum('aAkK,bBlL->abABklKL', I1, I1).reshape(9, 9, 9, 9)        # (out row, out col, in row, in col) on the pair
    return np.einsum('xyuv,uvab->xyab', E, S4)
def pair_channel(name, coh):
    if name in IDEAL_GATES: return _ideal_pair(name, coh)
    key = (name, None if coh is None else tuple(sorted(coh.items())))
    if key not in _CACHE: _CACHE[key] = logical_channel(gate_superop(name, P[name]['T_ns'], P[name]['x'], coh), cc.phases(name))
    return _CACHE[key]

def unitary_super(U):
    return np.einsum('ik,jl->ijkl', U, U.conj())

def run(rho, k, kind, theta, coh):
    n = k+1; S = schedule(k, kind, theta); pend = np.zeros(n); prev = None
    def flush(r, q):
        if pend[q] > 0 and coh is not None: r = cc._single_rho(r, single_idle(pend[q], coh), q, n)
        pend[q] = 0; return r
    for o in S:
        if o[0] == '2q':
            _, name, c, t = o
            if prev is not None and prev[0] == '2q' and (prev[2], prev[3]) != (c, t): pend += T_FLUX
            rho = flush(rho, c); rho = flush(rho, t)
            rho = cc._pair_rho(rho, pair_channel(name, coh), c, t, n)
            for q in range(n):
                if q not in (c, t): pend[q] += P[name]['T_ns']
            prev = o
        else:
            _, U, q, dur = o
            rho = flush(rho, q); rho = cc._single_rho(rho, unitary_super(U), q, n)
            pend += dur
            if dur > 0: prev = None
    for q in range(n): rho = flush(rho, q)
    return rho

# ------------------------------------------------------------------ ideal action on the qubit subspace
def bits(n): return list(itertools.product((0, 1), repeat=n))
def i3(b): return int(np.ravel_multi_index(b, (3,)*len(b)))
def ideal_vec(psi, k, kind, theta):
    n = k+1; B = bits(n); out = np.array(psi, complex).copy()
    i0 = B.index((1,)*k + (0,)); i1 = B.index((1,)*k + (1,))
    a, b = psi[i0], psi[i1]
    if kind == 'cnot': out[i0], out[i1] = b, a
    else:
        c, s = np.cos(theta/2), np.sin(theta/2); out[i0], out[i1] = c*a - s*b, s*a + c*b
    return out

def out_block(psi, k, kind, theta, coh):
    n = k+1; v = np.zeros(3**n, complex)
    for i, b in enumerate(bits(n)): v[i3(b)] = psi[i]
    r = run(np.outer(v, v.conj()).reshape((3,)*(2*n)), k, kind, theta, coh).reshape(3**n, 3**n)
    I = [i3(b) for b in bits(n)]
    return r[np.ix_(I, I)]

def fid(block, psi, k, kind, theta, phi):
    n = k+1; w = ideal_vec(psi, k, kind, theta)
    w = np.exp(1j*np.array([sum(phi[q]*b[q] for q in range(n)) for b in bits(n)]))*w
    return float(np.real(w.conj() @ block @ w))

def rand_states(n, Q, seed):
    rng = np.random.RandomState(seed); out = []
    for _ in range(Q):
        z = rng.randn(2**n)+1j*rng.randn(2**n); out.append(z/np.linalg.norm(z))
    return out

def fit_phases(k, kind, theta, Q=8):
    n = k+1; sts = rand_states(n, Q, 7); blocks = [out_block(s, k, kind, theta, None) for s in sts]
    f = lambda phi: -np.mean([fid(b, s, k, kind, theta, phi) for b, s in zip(blocks, sts)])
    return min((minimize(f, s0, method='BFGS') for s0 in [np.zeros(n)] + [np.random.RandomState(i).uniform(-1, 1, n) for i in range(2)]), key=lambda r: r.fun).x

def metrics(k, kind, theta, coh, phi, Q=8, nbasis=None, seed=0):
    n = k+1; B = bits(n)
    idx = list(range(2**n)) if (nbasis is None or nbasis >= 2**n) else sorted(set(np.random.RandomState(5).choice(2**n, nbasis, replace=False).tolist() + [B.index((1,)*n), B.index((1,)*k+(0,))]))
    bf = []
    for i in idx:
        e = np.zeros(2**n); e[i] = 1; bf.append(fid(out_block(e, k, kind, theta, coh), e, k, kind, theta, phi))
    rs = [fid(out_block(s, k, kind, theta, coh), s, k, kind, theta, phi) for s in rand_states(n, Q, 100+seed)]
    return dict(basis_mean=float(np.mean(bf)), basis_min=float(np.min(bf)), n_basis=len(idx),
                random_mean=float(np.mean(rs)), random_min=float(np.min(rs)), Q=Q)

def check_ideal(k, kind, theta, Q=6):
    """ideal (perfect) gates as permutation/rotation unitaries on n qutrits: verifies the decomposition exactly"""
    n = k+1; dim = 3**n
    def perm_gate(name, c, t):
        c0, j, kk = GATES[name]; U = np.zeros((dim, dim))
        for a in itertools.product(range(3), repeat=n):
            b = list(a)
            if a[c] == c0: b[t] = kk if a[t] == j else (j if a[t] == kk else a[t])
            U[i3(tuple(b)), i3(a)] = 1
        return U
    def one(U3, q):
        return np.kron(np.kron(np.eye(3**q), U3), np.eye(3**(n-q-1)))
    err = 0.0
    for s in rand_states(n, Q, 3):
        v = np.zeros(dim, complex)
        for i, b in enumerate(bits(n)): v[i3(b)] = s[i]
        for o in schedule(k, kind, theta):
            v = (perm_gate(o[1], o[2], o[3]) if o[0] == '2q' else one(o[1], o[2])) @ v
        w = ideal_vec(s, k, kind, theta); ref = np.zeros(dim, complex)
        for i, b in enumerate(bits(n)): ref[i3(b)] = w[i]
        err = max(err, 1 - abs(np.vdot(ref, v))**2)
    return err
