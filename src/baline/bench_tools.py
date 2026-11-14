"""
bench_tools.py — cost model and baselines for the benchmark section
===================================================================

Gate-level representation: a list of (kind, controls, target, theta) with kind in
  'mcx'  (multi-controlled X; k = len(controls); k=0 -> X, k=1 -> CNOT, k=2 -> Toffoli)
  'mcz'  (multi-controlled Z, same cost as mcx; H on the target is a single-qubit gate)
  'mcry' / 'mcrx' (multi-controlled rotation)
  '1q'   (single-qubit gate: zero cost, as requested — only two-body pulses are counted)

QUTRIT cost (this work; only two-body pulses counted, single-qubit/virtual gates ignored):
  CNOT / CZ (k=1)        : 1 pulse,      T = 60 ns   (assumed native coupler gate of the same kind)
  Toffoli / CCZ (k=2)    : 3 pulses,     T = 200 ns  (3 x 60 ns + 2 flux moves of 10 ns)
  C^k-X / C^k-Z (k>=3)   : 2k-1 pulses,  T = 360k-520 ns
  C^k-Ry / C^k-Rx (k>=1) : 2k pulses,    T = 360k-460 ns  (the 2 single-qubit rotations are ignored)
  pulse time of a circuit = ASAP critical path, gates on disjoint qubits run in parallel.

QUBIT baselines (IBM-style CZ basis, 68 ns per CZ, single-qubit gates ignored for time):
  qiskit_all2all : Qiskit 2.6.2 transpile, optimization_level=3, basis {cz, sx, x, rz}, no coupling map
  qiskit_routed  : same on FakeKingston (heavy-hex) with SWAP routing
  2q time = (two-qubit depth) x 68 ns
"""
import re, itertools, warnings
import numpy as np
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate, XGate, ZGate, RYGate, RXGate

T_CZ_IBM = 68.0
FAST_C2X12 = False      # projection: C^2X_12 as fast as the other two gates (60 ns) -> T = 140k - 80 ns
def qutrit_gate_cost(kind, k):
    if kind in ('mcx', 'mcz'):
        if k == 0: return 0, 0.0
        if k == 1: return 1, 60.0
        if k == 2: return 3, 200.0
        return 2*k-1, (140.0*k - 80.0) if FAST_C2X12 else (360.0*k - 520.0)
    if kind in ('mcry', 'mcrx'):
        if k == 0: return 0, 0.0
        return 2*k, (140.0*k - 20.0) if FAST_C2X12 else (360.0*k - 460.0)
    return 0, 0.0

def qutrit_cost(gates, n, fast=None):
    global FAST_C2X12
    old = FAST_C2X12
    if fast is not None: FAST_C2X12 = fast
    try:
        return _qutrit_cost(gates, n)
    finally:
        FAST_C2X12 = old

def _qutrit_cost(gates, n):
    """total pulses, ASAP critical-path pulse time (ns), and gate histogram"""
    busy = np.zeros(n); pulses = 0; hist = {}
    for kind, ctr, tgt, th in gates:
        if kind == '1q': continue
        k = len(ctr); p, T = qutrit_gate_cost(kind, k)
        if p == 0: continue
        qs = list(ctr) + [tgt]; start = max(busy[q] for q in qs); end = start + T
        for q in qs: busy[q] = end
        pulses += p; key = '%s%d' % ('C' if kind in ('mcx', 'mcz') else 'R', k); hist[key] = hist.get(key, 0) + 1
    return dict(pulses=int(pulses), time_ns=float(busy.max() if n else 0), hist=hist)

# ------------------------------------------------------------------ Quipper ASCII (.qc "_before" files of the Amy-Maslov-Mosca suite)
_GATE = re.compile(r'QGate\["(\w+)"\](\*?)\((\d+)\)(?: with controls=\[([^\]]*)\])?')
def parse_quipper(path):
    lines = open(path).read().splitlines(); n = lines[0].count('Qbit'); gates = []
    for ln in lines[1:]:
        m = _GATE.match(ln.strip())
        if not m: continue
        name, dag, t = m.group(1), m.group(2) == '*', int(m.group(3))
        ctr = [c.strip() for c in (m.group(4) or '').split(',') if c.strip()]
        if any(c.startswith('-') for c in ctr): raise ValueError('negative controls not supported')
        ctr = [int(c.lstrip('+')) for c in ctr]
        if name == 'not': gates.append(('mcx', ctr, t, None))
        elif name == 'Z' and ctr: gates.append(('mcz', ctr, t, None))
        elif not ctr: gates.append(('1q', [], t, (name, dag)))
        else: raise ValueError('unsupported controlled gate ' + name)
    return n, gates

def to_qiskit(n, gates):
    qc = QuantumCircuit(n)
    for kind, ctr, t, th in gates:
        if kind == '1q':
            name, dag = th if isinstance(th, tuple) else (th, False)
            g = {'H': 'h', 'Z': 'z', 'S': 'sdg' if dag else 's', 'T': 'tdg' if dag else 't', 'X': 'x', 'not': 'x'}[name]
            getattr(qc, g)(t)
        elif kind == 'mcx':
            if len(ctr) == 0: qc.x(t)
            elif len(ctr) == 1: qc.cx(ctr[0], t)
            elif len(ctr) == 2: qc.ccx(ctr[0], ctr[1], t)
            else: qc.append(MCXGate(len(ctr)), ctr + [t])
        elif kind == 'mcz':
            if len(ctr) == 1: qc.cz(ctr[0], t)
            else: qc.h(t); qc.append(MCXGate(len(ctr)), ctr + [t]) if len(ctr) > 2 else qc.ccx(ctr[0], ctr[1], t); qc.h(t)
        elif kind == 'mcry':
            qc.append(RYGate(th).control(len(ctr)), ctr + [t])
        elif kind == 'mcrx':
            qc.append(RXGate(th).control(len(ctr)), ctr + [t])
    return qc

# ------------------------------------------------------------------ Qiskit library circuits -> MCT level
_ONEQ = {'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'rz', 'ry', 'rx', 'p', 'u', 'u1', 'u2', 'u3', 'sx', 'sxdg', 'id', 'barrier', 'measure', 'reset', 'global_phase'}
def to_mct(qc, max_rounds=12):
    """decompose a library circuit until it contains only 1q gates, cx/cz, ccx and multi-controlled X"""
    for _ in range(max_rounds):
        bad = set()
        for ins in qc.data:
            op = ins.operation; nm = op.name
            if nm in _ONEQ or nm in ('cx', 'cz', 'ccx', 'ccz'): continue
            if isinstance(op, MCXGate) or nm.startswith('mcx') or nm in ('c3x', 'c4x'): continue
            bad.add(nm)
        if not bad: break
        qc = qc.decompose(gates_to_decompose=list(bad))
    gates = []
    for ins in qc.data:
        op = ins.operation; qs = [qc.find_bit(q).index for q in ins.qubits]; nm = op.name
        if nm in ('barrier', 'measure', 'reset', 'global_phase', 'id'): continue
        if len(qs) == 1: gates.append(('1q', [], qs[0], nm)); continue
        if nm == 'cz': gates.append(('mcz', qs[:1], qs[1], None)); continue
        if nm == 'ccz': gates.append(('mcz', qs[:2], qs[2], None)); continue
        if nm in ('cx', 'ccx', 'c3x', 'c4x') or isinstance(op, MCXGate) or nm.startswith('mcx'):
            if getattr(op, 'ctrl_state', None) not in (None, 2**(len(qs)-1)-1):        # open controls -> X sandwich (1q, free)
                pass
            gates.append(('mcx', qs[:-1], qs[-1], None)); continue
        raise ValueError('unhandled gate ' + nm)
    return qc.num_qubits, gates

# ------------------------------------------------------------------ qubit baselines
def twoq_stats(t):
    ops = t.count_ops(); n2 = int(sum(v for k, v in ops.items() if k in ('cz', 'cx', 'ecr', 'swap')))
    d2 = t.depth(filter_function=lambda ins: ins.operation.num_qubits == 2)
    return dict(n2q=n2, depth2q=int(d2), time2q_ns=float(d2*T_CZ_IBM))

def qiskit_all2all(qc, seed=0):
    t = transpile(qc, basis_gates=['cz', 'sx', 'x', 'rz'], optimization_level=3, seed_transpiler=seed)
    return twoq_stats(t)

_BE = {}
def qiskit_routed(qc, backend='FakeKingston', seeds=(0, 1, 2)):
    from qiskit_ibm_runtime import fake_provider as fp
    if backend not in _BE: _BE[backend] = getattr(fp, backend)()
    be = _BE[backend]; best = None
    if qc.num_qubits > be.num_qubits: return None
    for s in seeds:
        t = transpile(qc, backend=be, optimization_level=3, seed_transpiler=s)
        r = twoq_stats(t)
        if best is None or r['time2q_ns'] < best['time2q_ns']: best = r
    return best

def quipper_cnot_stats(path):
    """two-qubit statistics of an optimized Clifford+T circuit (Nam et al. outputs as distributed with PyZX)"""
    n, gates = parse_quipper(path)
    qc = QuantumCircuit(n); cnt = 0
    for kind, ctr, t, th in gates:
        if kind == 'mcx' and len(ctr) == 1: qc.cx(ctr[0], t); cnt += 1
        elif kind == 'mcz' and len(ctr) == 1: qc.cz(ctr[0], t); cnt += 1
        elif kind in ('mcx', 'mcz') and len(ctr) > 1: raise ValueError('multi-controlled gate in optimized file')
    d2 = qc.depth(filter_function=lambda ins: ins.operation.num_qubits == 2)
    return dict(n2q=cnt, depth2q=int(d2), time2q_ns=float(d2*T_CZ_IBM))
