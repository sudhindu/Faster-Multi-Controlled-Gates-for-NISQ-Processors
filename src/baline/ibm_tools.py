"""
ibm_tools.py — IBM baseline (qubit-only) for Toffoli and C^k-NOT
===============================================================

Uses Qiskit fake backends (snapshots of real IBM device calibrations: coupling map = heavy-hex for Eagle/Heron,
square lattice for Nighthawk; native gates, gate durations, T1/T2, gate errors).

  best_transpile : transpile the qubit-only C^k-NOT (Qiskit mcx synthesis, no ancilla) with optimization_level=3,
                   SWAP routing on the real coupling map, ALAP scheduling; keeps the seed with the shortest schedule.
  pulse_counts   : physical pulses = 2-qubit gates (cz / ecr / cx) + single-qubit pulses (sx, x);
                   rz is a virtual (frame) gate and costs no pulse and no time — the same convention as for the qutrits.
  noisy_eval     : Aer simulation of the scheduled circuit with NoiseModel.from_backend (gate depolarising +
                   thermal relaxation) PLUS RelaxationNoisePass (idle T1/T2 decay during the schedule), readout
                   error off; returns basis-input success and superposition fidelities from the process superoperator.
IBM numbers are noise-model simulations of calibration snapshots, not hardware runs (no crosstalk, no drift).
"""
import itertools, warnings
import numpy as np
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveSuperOp, SaveUnitary
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.passes import RelaxationNoisePass
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime import fake_provider as fp

BACKENDS = ['FakeKingston', 'FakeNighthawk', 'FakeFez', 'FakeTorino', 'FakeBrisbane']
TWOQ = ('cz', 'ecr', 'cx', 'rzz')

def mcx_circuit(k):
    qc = QuantumCircuit(k+1); qc.mcx(list(range(k)), k); return qc

def mcx_ancilla_circuit(k):
    """linear-cost C^k-NOT with k-2 clean ancilla qubits (Qiskit synth_mcx_n_clean_m15, V-chain type)"""
    from qiskit.synthesis import synth_mcx_n_clean_m15
    return synth_mcx_n_clean_m15(k)

def best_transpile(k, name, nseed=30, mmax=None, qc=None):
    be = getattr(fp, name)(); tg = be.target; qc = mcx_circuit(k) if qc is None else qc; best = None
    for seed in range(nseed):
        t = transpile(qc, backend=be, optimization_level=3, seed_transpiler=seed, scheduling_method='alap')
        act = sorted({t.find_bit(q).index for ins in t.data if ins.operation.name not in ('delay', 'barrier') for q in ins.qubits})
        if mmax is not None and len(act) > mmax: continue
        d = 1e9*t.estimate_duration(tg, unit='s')
        if best is None or d < best[0]: best = (d, seed, t, act)
    return be, best

def pulse_counts(t):
    ops = t.count_ops()
    n2 = int(sum(v for k, v in ops.items() if k in TWOQ)); n1 = int(sum(v for k, v in ops.items() if k in ('sx', 'x')))
    depth = t.depth(filter_function=lambda ins: ins.operation.name not in ('delay', 'barrier', 'rz'))
    return dict(n2q=n2, n1q_pulses=n1, total_pulses=n2+n1, depth_pulses=int(depth))

def all_to_all_counts(k, ancilla=False):
    """decomposition cost without routing (all-to-all connectivity), Heron-like basis {cz, sx, x, rz}"""
    qc = mcx_ancilla_circuit(k) if (ancilla and k >= 3) else mcx_circuit(k)
    t = transpile(qc, basis_gates=['cz', 'sx', 'x', 'rz'], optimization_level=3, seed_transpiler=0)
    return pulse_counts(t)

def device_coherence(be):
    T1 = [be.qubit_properties(q).t1 for q in range(be.num_qubits)]; T2 = [be.qubit_properties(q).t2 for q in range(be.num_qubits)]
    return float(np.median([v for v in T1 if v])*1e6), float(np.median([v for v in T2 if v])*1e6)

def _ck(bits_): return bits_[:-1] + (bits_[-1] ^ int(all(bits_[:-1])),)

def noisy_eval(k, name, nseed=30, mmax=6, nrand=40):
    be, best = best_transpile(k, name, nseed, mmax)
    if best is None: return None
    d, seed, t, act = best; m = len(act); n = k+1
    pos = {q: i for i, q in enumerate(act)}
    qu = QuantumCircuit(m)
    for ins in t.data:
        if ins.operation.name in ('delay', 'barrier'): continue
        qu.append(ins.operation, [pos[t.find_bit(q).index] for q in ins.qubits])
    qu.append(SaveUnitary(m, 'U'), list(range(m)))
    U = np.asarray(AerSimulator(method='unitary').run(qu).result().data()['U'])
    # locate where each logical qubit enters / leaves (routing permutes qubits)
    X = np.array(list(itertools.product((0, 1), repeat=n))); ideal = np.array([_ck(tuple(x)) for x in X]); found = None
    for pin in itertools.permutations(range(m), n):
        ins_ = [int(sum(int(x[i]) << pin[i] for i in range(n))) for x in X]
        outs = [int(np.argmax(abs(U[:, a]))) for a in ins_]
        if min(abs(U[o, a]) for o, a in zip(outs, ins_)) < 0.99: continue
        Y = np.array([[(o >> j) & 1 for j in range(m)] for o in outs]); pout = []
        for i in range(n):
            js = [j for j in range(m) if np.array_equal(Y[:, j], ideal[:, i])]
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
    dM = 2**m
    def evolve(psi):
        v = np.zeros(dM, complex)
        for xi, x in enumerate(X): v[int(sum(int(x[i]) << pin[i] for i in range(n)))] = psi[xi]
        o = (S @ np.outer(v, v.conj()).reshape(-1, order='F')).reshape(dM, dM, order='F').reshape((2,)*(2*m))
        rows = [m-1-q for q in range(m)]; cols = [2*m-1-q for q in range(m)]
        sub = list(range(2*m))
        for q in range(m):
            if q not in pout: sub[cols[q]] = sub[rows[q]]
        return np.einsum(o, sub, [rows[pout[i]] for i in range(n)] + [cols[pout[i]] for i in range(n)]).reshape(2**n, 2**n)
    Pm = np.zeros((2**n, 2**n))
    for xi in range(2**n): Pm[int(np.where((X == ideal[xi]).all(1))[0][0]), xi] = 1
    succ = []
    for xi in range(2**n):
        e = np.zeros(2**n); e[xi] = 1; o = evolve(e); j = int(np.argmax(Pm[:, xi])); succ.append(float(np.real(o[j, j])))
    rng = np.random.RandomState(3); fs = []
    for _ in range(nrand):
        z = rng.randn(2**n)+1j*rng.randn(2**n); z /= np.linalg.norm(z); o = evolve(z); w = Pm @ z; fs.append(float(np.real(w.conj() @ o @ w)))
    plus = np.ones(2**n)/np.sqrt(2**n); o = evolve(plus); w = Pm @ plus
    T1med, T2med = device_coherence(be)
    return dict(backend=name, k=k, duration_ns=float(d), active_qubits=m, succ=succ, succ_mean=float(np.mean(succ)), succ_min=float(np.min(succ)),
                F_random_mean=float(np.mean(fs)), F_random_min=float(np.min(fs)), F_plus=float(np.real(w.conj() @ o @ w)),
                T1_median_us=T1med, T2_median_us=T2med, **pulse_counts(t))
