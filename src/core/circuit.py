"""
circuit.py — multi-qutrit circuit simulator for the C^k-NOT construction (Toffoli = k = 2)
=========================================================================================

C^k-NOT (fold / flip / unfold; k controls q0..q_{k-1}, target q_k, nearest-neighbour line, NO ancilla):
    fold   : C^1X_12(q0->q1), then C^2X_12(q_{j-1}->q_j) for j = 2..k-1     (k-1 gates)
    flip   : C^2X_01(q_{k-1}->q_k)                                          (1 gate)
    unfold : mirror of the fold                                              (k-1 gates)
  -> 2k-1 two-qutrit gates, depth 2k-1, zero single-qutrit pulses (virtual Z only).
After the fold, q_j sits in |2> exactly when q0..q_j were all |1>; the flip fires only for all-ones.

Open-system simulation: each two-qutrit gate is the Lindblad channel of noise.gate_superop restricted to the
9 logical states (its leakage to |3> is lost probability); every qutrit not taking part in a gate idles with
its own 3-level T1/T2 channel (|2> decays faster); between consecutive gates all qutrits idle for the flux
move T_FLUX of the qutrit that changes role. Quasi-static errors are sampled per qutrit (frequency offset)
and per gate (amplitude) / per device (coupler strength); a mirrored gate re-uses its fold partner's channel.
"""
import itertools
import numpy as np
from common import *
from noise import gate_superop, logical_channel, single_idle, rwa_phases

T_FLUX = 10.0
P = load_params()
PH = {}
def phases(g):
    if g not in PH: PH[g] = rwa_phases(g, P[g]['T_ns'], P[g]['x'])
    return PH[g]

def gates(k):
    fold = [('C1X12', 0, 1)] + [('C2X12', j-1, j) for j in range(2, k)]
    return fold + [('C2X01', k-1, k)] + fold[::-1]

def durations(k):
    G = gates(k); Tg = sum(P[g][ 'T_ns'] for g, _, _ in G)
    return dict(n_two_qutrit=len(G), n_single=0, depth=len(G), T_gates_ns=Tg, T_flux_ns=T_FLUX*(len(G)-1), T_total_ns=Tg+T_FLUX*(len(G)-1))

def check_logic(k):
    act = {'C1X12': (1, 1, 2), 'C2X01': (2, 0, 1), 'C2X12': (2, 1, 2)}
    for bits in itertools.product((0, 1), repeat=k+1):
        s = list(bits)
        for name, c, t in gates(k):
            c0, j, kk = act[name]
            if s[c] == c0: s[t] = kk if s[t] == j else (j if s[t] == kk else s[t])
        if tuple(s) != bits[:k] + (bits[k] ^ int(all(bits[:k])),): return False
    return True

_CACHE = {}
def channels(k, coh, errs=None):
    """one 9x9x9x9 logical channel per distinct gate of fold+flip"""
    out = []
    for i, (name, c, t) in enumerate(gates(k)[:k]):
        if errs is None:
            key = (name, None if coh is None else tuple(sorted(coh.items())))
            if key not in _CACHE: _CACHE[key] = logical_channel(gate_superop(name, P[name]['T_ns'], P[name]['x'], coh), phases(name))
            out.append(_CACHE[key])
        else:
            out.append(logical_channel(gate_superop(name, P[name]['T_ns'], P[name]['x'], coh, errs[i]), phases(name)))
    return out

def sample_errors(k, rng, s):
    """s = dict(f=flux sigma MHz, g=coupler fraction, a=amplitude fraction)"""
    n = k+1; dq = [rng.randn()*s['f'] for q in range(n)]; gs = rng.randn()*s['g']
    return [dict(dfc=dq[c], dft=dq[t], gsc=gs, amp=rng.randn()*s['a']) for (name, c, t) in gates(k)[:k]]

# ---------------- tensor application (rho has 2n axes: rows q0..q_{n-1}, then columns)
def _pair_rho(rho, S4, c, t, n):
    N2 = 2*n; ol = list(range(N2)); ol[c] = N2; ol[t] = N2+1; ol[n+c] = N2+2; ol[n+t] = N2+3
    return np.einsum(S4.reshape((3,)*8), [N2, N2+1, N2+2, N2+3, c, t, n+c, n+t], rho, list(range(N2)), ol, optimize=True)
def _single_rho(rho, S1, q, n):
    N2 = 2*n; ol = list(range(N2)); ol[q] = N2; ol[n+q] = N2+1
    return np.einsum(S1, [N2, N2+1, q, n+q], rho, list(range(N2)), ol, optimize=True)
def _pair_pop(p, S4, c, t, n):
    P4 = np.real(np.einsum('aabb->ab', S4)).reshape(3, 3, 3, 3)
    ol = list(range(n)); ol[c] = n; ol[t] = n+1
    return np.einsum(P4, [n, n+1, c, t], p, list(range(n)), ol, optimize=True)
def _single_pop(p, S1, q, n):
    ol = list(range(n)); ol[q] = n
    return np.einsum(np.real(np.einsum('iikk->ik', S1)), [n, q], p, list(range(n)), ol, optimize=True)

def run(state, k, chans, coh, mode='rho'):
    """propagate a density tensor (mode='rho') or a population tensor (mode='pop') through the C^k-NOT schedule"""
    n = k+1; G = gates(k); pend = np.zeros(n)
    pair = _pair_rho if mode == 'rho' else _pair_pop; single = _single_rho if mode == 'rho' else _single_pop
    def flush(st, q):
        if pend[q] > 0 and coh is not None: st = single(st, single_idle(pend[q], coh), q, n)
        pend[q] = 0; return st
    for gi, (name, c, t) in enumerate(G):
        state = flush(state, c); state = flush(state, t)
        state = pair(state, chans[min(gi, len(G)-1-gi)], c, t, n)
        for q in range(n):
            if q not in (c, t): pend[q] += P[name]['T_ns']
        if gi < len(G)-1: pend += T_FLUX
    for q in range(n): state = flush(state, q)
    return state

# ---------------- metrics
def bits(n): return list(itertools.product((0, 1), repeat=n))
def ideal_bits(b): return b[:-1] + (b[-1] ^ int(all(b[:-1])),)
def i3(b): return int(np.ravel_multi_index(b, (3,)*len(b)))

def basis_success(k, chans, coh):
    n = k+1; out = {}
    for b in bits(n):
        p = np.zeros((3,)*n); p[b] = 1
        out[''.join(map(str, b))] = float(run(p, k, chans, coh, 'pop')[ideal_bits(b)])
    return out

def out_block(psi, k, chans, coh):
    n = k+1; v = np.zeros(3**n, complex)
    for i, b in enumerate(bits(n)): v[i3(b)] = psi[i]
    rho = run(np.outer(v, v.conj()).reshape((3,)*(2*n)), k, chans, coh).reshape(3**n, 3**n)
    I = [i3(b) for b in bits(n)]
    return rho[np.ix_(I, I)]

def ideal_out(psi, n, phi):
    out = np.zeros(2**n, complex); B = bits(n)
    for i, b in enumerate(B): out[B.index(ideal_bits(b))] = psi[i]
    return np.exp(1j*np.array([sum(phi[q]*b[q] for q in range(n)) for b in B]))*out

def fidelity(block, psi, n, phi):
    w = ideal_out(psi, n, phi); return float(np.real(w.conj() @ block @ w))

def random_states(n, Q, seed):
    rng = np.random.RandomState(seed); out = []
    for _ in range(Q):
        z = rng.randn(2**n)+1j*rng.randn(2**n); out.append(z/np.linalg.norm(z))
    return out

def named_states(n):
    plus = np.ones(2**n)/np.sqrt(2**n); ghz = np.zeros(2**n); ghz[0] = ghz[-1] = 1/np.sqrt(2)
    cp = np.array([1.0 if b[-1] == 0 else 0.0 for b in bits(n)]); cp /= np.linalg.norm(cp)
    return {'|+...+>': plus, 'GHZ': ghz, 'controls |+>, target |0> (entangling)': cp}

def fit_final_phases(k, chans0, Q=10):
    """3..n free final virtual-Z phases fitted once on the noise-free circuit, then kept fixed"""
    from scipy.optimize import minimize
    n = k+1; sts = random_states(n, Q, 7); blocks = [out_block(s, k, chans0, None) for s in sts]
    f = lambda phi: -np.mean([fidelity(b, s, n, phi) for b, s in zip(blocks, sts)])
    best = min((minimize(f, s0, method='BFGS') for s0 in [np.zeros(n)] + [np.random.RandomState(i).uniform(-1, 1, n) for i in range(3)]), key=lambda r: r.fun)
    return best.x

def superposition_metrics(k, chans, coh, phi, Q=12, seed=0):
    n = k+1; fs = []
    for s in random_states(n, Q, 100+seed): fs.append(fidelity(out_block(s, k, chans, coh), s, n, phi))
    named = {name: fidelity(out_block(v, k, chans, coh), v, n, phi) for name, v in named_states(n).items()}
    return dict(F_random_mean=float(np.mean(fs)), F_random_min=float(np.min(fs)), named=named)

def process_fidelity_k2(chans, coh, phi):
    """exact qubit-subspace process (entanglement) fidelity for the Toffoli (k=2) from 64 operator inputs"""
    n = 3; B = bits(n); tot = 0
    for a, ba in enumerate(B):
        for b, bb in enumerate(B):
            v1 = np.zeros(27); v1[i3(ba)] = 1; v2 = np.zeros(27); v2[i3(bb)] = 1
            O = run(np.outer(v1, v2).astype(complex).reshape((3,)*6), 2, chans, coh).reshape(27, 27)
            oa, ob = ideal_bits(ba), ideal_bits(bb)
            pa = np.exp(1j*sum(phi[q]*oa[q] for q in range(3))); pb = np.exp(1j*sum(phi[q]*ob[q] for q in range(3)))
            tot += np.conj(pa)*pb*O[i3(oa), i3(ob)]
    return float(np.real(tot))/64
