"""
ibm_ry.py — IBM baseline for C^k-Ry(theta) (and helpers shared with C^k-NOT)
===========================================================================
  mcry_circuit(k, th)          : Qiskit multi-controlled RY, no ancilla (default synthesis)
  mcry_ancilla_circuit(k, th)  : AND of the k controls into one result qubit with a linear-cost C^k-X that uses
                                 k-2 clean ancillas (synth_mcx_n_clean_m15), a CRy from the result qubit, and
                                 uncomputation; total 2k ancilla-assisted structure (fair 'ancilla' baseline)
  noisy_eval_unitary(...)      : Aer superoperator of the scheduled circuit (gate + idle relaxation noise,
                                 readout off) -> mean basis-input fidelity and random-state fidelity
"""
import itertools, warnings
import numpy as np
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate
from qiskit.synthesis import synth_mcx_n_clean_m15
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveSuperOp, SaveUnitary
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.passes import RelaxationNoisePass
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime import fake_provider as fp
from ibm_tools import pulse_counts, device_coherence

def mcry_circuit(k, th):
    qc = QuantumCircuit(k+1); qc.append(RYGate(th).control(k), list(range(k+1))); return qc

def mcry_ancilla_circuit(k, th):
    if k < 3:
        return mcry_circuit(k, th)
    m = synth_mcx_n_clean_m15(k)                # qubits: k controls, 1 target, k-2 ancillas
    n = m.num_qubits + 1                        # + the data target
    qc = QuantumCircuit(n)
    res = k                                     # result qubit of the AND
    qc.compose(m, qubits=list(range(m.num_qubits)), inplace=True)
    qc.cry(th, res, n-1)
    qc.compose(m.inverse(), qubits=list(range(m.num_qubits)), inplace=True)
    return qc

def best_transpile_qc(qc, name, nseed=30, mmax=None):
    be = getattr(fp, name)(); tg = be.target; best = None
    for seed in range(nseed):
        t = transpile(qc, backend=be, optimization_level=3, seed_transpiler=seed, scheduling_method='alap')
        act = sorted({t.find_bit(q).index for ins in t.data if ins.operation.name not in ('delay', 'barrier') for q in ins.qubits})
        if mmax is not None and len(act) > mmax: continue
        d = 1e9*t.estimate_duration(tg, unit='s')
        if best is None or d < best[0]: best = (d, seed, t, act)
    return be, best

def all_to_all(qc):
    return pulse_counts(transpile(qc, basis_gates=['cz', 'sx', 'x', 'rz'], optimization_level=3, seed_transpiler=0))

def _ideal(psi, k, th):
    n = k+1; B = list(itertools.product((0, 1), repeat=n)); out = np.array(psi, complex).copy()
    i0 = B.index((1,)*k + (0,)); i1 = B.index((1,)*k + (1,)); a, b = psi[i0], psi[i1]
    c, s = np.cos(th/2), np.sin(th/2); out[i0], out[i1] = c*a - s*b, s*a + c*b
    return out

def noisy_eval_unitary(k, name, th, nseed=30, mmax=6, nrand=30):
    """C^k-Ry on an IBM fake back-end; logical qubit order (b0..bk), target last"""
    be, best = best_transpile_qc(mcry_circuit(k, th), name, nseed, mmax)
    if best is None: return None
    d, seed, t, act = best; m = len(act); n = k+1
    pos = {q: i for i, q in enumerate(act)}
    qu = QuantumCircuit(m)
    for ins in t.data:
        if ins.operation.name in ('delay', 'barrier'): continue
        qu.append(ins.operation, [pos[t.find_bit(q).index] for q in ins.qubits])
    qu.append(SaveUnitary(m, 'U'), list(range(m)))
    U = np.asarray(AerSimulator(method='unitary').run(qu).result().data()['U'])
    # qubit positions: inputs whose controls are not all 1 are mapped to themselves (identity)
    X = [x for x in itertools.product((0, 1), repeat=n) if not all(x[:k])]
    found = None
    for pin in itertools.permutations(range(m), n):
        ins_ = [int(sum(x[i] << pin[i] for i in range(n))) for x in X]
        outs = [int(np.argmax(abs(U[:, a]))) for a in ins_]
        if min(abs(U[o, a]) for o, a in zip(outs, ins_)) < 0.99: continue
        Y = np.array([[(o >> j) & 1 for j in range(m)] for o in outs]); Xa = np.array(X); pout = []
        for i in range(n):
            js = [j for j in range(m) if np.array_equal(Y[:, j], Xa[:, i])]
            if not js: break
            pout.append(js[0])
        if len(pout) == n and len(set(pout)) == n: found = (pin, pout); break
    pin, pout = found
    T1 = [be.qubit_properties(q).t1 for q in range(be.num_qubits)]; T2 = [be.qubit_properties(q).t2 for q in range(be.num_qubits)]
    m1 = np.median([v for v in T1 if v]); m2 = np.median([v for v in T2 if v])
    t1s = [v if v else m1 for v in T1]; t2s = [min(v if v else m2, 2*t1s[i]) for i, v in enumerate(T2)]
    tc = PassManager([RelaxationNoisePass(t1s, t2s, dt=be.dt)]).run(t)
    qs = tc.copy(); qs.append(SaveSuperOp(m, 'S'), act)
    S = np.asarray(AerSimulator(noise_model=NoiseModel.from_backend(be, readout_error=False), method='superop').run(qs).result().data()['S'])
    dM = 2**m; Xall = list(itertools.product((0, 1), repeat=n))
    def evolve(psi):
        v = np.zeros(dM, complex)
        for xi, x in enumerate(Xall): v[int(sum(x[i] << pin[i] for i in range(n)))] = psi[xi]
        o = (S @ np.outer(v, v.conj()).reshape(-1, order='F')).reshape(dM, dM, order='F').reshape((2,)*(2*m))
        rows = [m-1-q for q in range(m)]; cols = [2*m-1-q for q in range(m)]; sub = list(range(2*m))
        for q in range(m):
            if q not in pout: sub[cols[q]] = sub[rows[q]]
        return np.einsum(o, sub, [rows[pout[i]] for i in range(n)] + [cols[pout[i]] for i in range(n)]).reshape(2**n, 2**n)
    # global phase of the transpiled unitary is irrelevant for fidelities with pure states
    bf = []
    for xi in range(2**n):
        e = np.zeros(2**n); e[xi] = 1; w = _ideal(e, k, th); bf.append(float(np.real(w.conj() @ evolve(e) @ w)))
    rng = np.random.RandomState(3); fs = []
    for _ in range(nrand):
        z = rng.randn(2**n)+1j*rng.randn(2**n); z /= np.linalg.norm(z); w = _ideal(z, k, th); fs.append(float(np.real(w.conj() @ evolve(z) @ w)))
    return dict(backend=name, k=k, duration_ns=float(d), active_qubits=m, basis_mean=float(np.mean(bf)), basis_min=float(np.min(bf)),
                random_mean=float(np.mean(fs)), T1_median_us=device_coherence(be)[0], **pulse_counts(t))
