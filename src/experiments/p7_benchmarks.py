"""
Part 7 — Benchmark circuits: two-body pulse count and pulse time, qutrit decomposition vs qubit baselines
========================================================================================================
Categories
  A. Grover search / amplitude amplification (n = 3..12 qubits): per iteration an oracle C^{n-1}Z and the
     diffusion C^{n-1}Z; r = floor(pi/4 sqrt(2^n)) iterations.
  B. Reversible arithmetic (Shor building blocks): Cuccaro/CDKM and VBE ripple-carry adders (n = 2..32 bits,
     Qiskit library), plus the arithmetic circuits of the Amy-Maslov-Mosca suite (adders, modular adder /
     multiplier / reduction, GF(2^m) multipliers, comparator).
  C. Boolean oracles / reversible logic: random 3-SAT phase oracles (n = 4..14 variables, m = round(4.26 n)
     clauses) and the Toffoli/MCT circuits of the same suite (tof_k, barenco_tof_k, mod5_4).
Qutrit cost: bench_tools.qutrit_gate_cost (two-body pulses only). Qubit baselines: Qiskit all-to-all and routed
(FakeKingston), best MCX synthesis among Qiskit's synthesis functions, and for the suite the optimized circuits of
Nam et al. (``*_after_heavy``). 'MCT-aware' = a C^kX built from Toffolis and ancillas is recognized and replaced by
one native C^kNOT (what the tool flow of Section 5 does).
Run: python3 p7_benchmarks.py grover arith sat amm plot     (~15 min)
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import os, sys, json, csv, glob, math, time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate, CDKMRippleCarryAdder, VBERippleCarryAdder
from qiskit import synthesis as syn
from bench_tools import *
from common import HERE

from paths import rpath, fpath, RESULTS as OUT, FIGURES as FIG, DATA, nproc
AMM = os.path.join(DATA, 'benchmarks_amm')
def jd(o, n): json.dump(o, open(rpath(n), 'w'), indent=1)
def jl(n):
    f = rpath(n); return json.load(open(f)) if os.path.exists(f) else {}
def wcsv(rows, name):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(rpath(name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()})

# ------------------------------------------------------------------ A. Grover
def mcz_synth(k, method):
    """circuit of a C^kZ (H . C^kX . H on the target) with a given Qiskit MCX synthesis; returns (circuit, n_anc)"""
    f = {'noaux_v24': lambda: syn.synth_mcx_noaux_v24(k), 'noaux_hp24': lambda: syn.synth_mcx_noaux_hp24(k),
         'gray': lambda: syn.synth_mcx_gray_code(k), '1clean_kg24': lambda: syn.synth_mcx_1_clean_kg24(k),
         '2clean_kg24': lambda: syn.synth_mcx_2_clean_kg24(k), 'nclean_m15': lambda: syn.synth_mcx_n_clean_m15(k)}[method]
    c = f(); return c, c.num_qubits - (k+1)

def grover_iteration(n, method=None):
    """one Grover iteration on n qubits (marked state |1..1>); method=None -> Qiskit default MCX"""
    k = n-1
    if method is None:
        qc = QuantumCircuit(n); body = lambda: qc.append(MCXGate(k), list(range(n))); anc = 0
    else:
        g, anc = mcz_synth(k, method); qc = QuantumCircuit(n + anc)
        body = lambda: qc.compose(g, qubits=list(range(n + anc)), inplace=True)
    qc.h(n-1); body(); qc.h(n-1)                             # oracle: C^{n-1}Z
    for q in range(n): qc.h(q); qc.x(q)                       # diffusion
    qc.h(n-1); body(); qc.h(n-1)
    for q in range(n): qc.x(q); qc.h(q)
    return qc, anc

def grover():
    rows = []
    for n in range(3, 13):
        r = int(math.floor(math.pi/4*math.sqrt(2**n)))
        qc, _ = grover_iteration(n); nq, g = to_mct(qc); ours = qutrit_cost(g, nq)
        base = qiskit_all2all(qc)
        best = None
        for m in ('noaux_v24', 'noaux_hp24', '1clean_kg24', '2clean_kg24', 'nclean_m15'):
            try:
                qb, anc = grover_iteration(n, m); s = qiskit_all2all(qb); s['method'] = m; s['ancillas'] = anc
                if best is None or s['n2q'] < best['n2q']: best = s
            except Exception as e:
                pass
        routed = qiskit_routed(qc) if n <= 10 else None
        row = dict(n=n, iterations=r, qutrit_pulses_iter=ours['pulses'], qutrit_time_iter_ns=ours['time_ns'],
                   qiskit_n2q_iter=base['n2q'], qiskit_time_iter_ns=base['time2q_ns'],
                   best_n2q_iter=best['n2q'], best_time_iter_ns=best['time2q_ns'], best_method=best['method'], best_ancillas=best['ancillas'],
                   routed_n2q_iter=routed['n2q'] if routed else '', routed_time_iter_ns=routed['time2q_ns'] if routed else '')
        for key in ('qutrit_pulses', 'qutrit_time', 'qiskit_n2q', 'qiskit_time', 'best_n2q', 'best_time'):
            src = {'qutrit_pulses': 'qutrit_pulses_iter', 'qutrit_time': 'qutrit_time_iter_ns', 'qiskit_n2q': 'qiskit_n2q_iter',
                   'qiskit_time': 'qiskit_time_iter_ns', 'best_n2q': 'best_n2q_iter', 'best_time': 'best_time_iter_ns'}[key]
            row[key + '_total'] = row[src]*r
        rows.append(row); print('Grover n=%d r=%d: qutrit %d pulses %.2f us/iter | qiskit %d CZ %.2f us | best %d CZ %.2f us (%s, %d anc) | routed %s' % (
            n, r, ours['pulses'], ours['time_ns']/1e3, base['n2q'], base['time2q_ns']/1e3, best['n2q'], best['time2q_ns']/1e3, best['method'], best['ancillas'], routed and routed['n2q']), flush=True)
        jd(rows, 'part7_grover.json'); wcsv(rows, 'part7_grover.csv')

# ------------------------------------------------------------------ B. arithmetic (Qiskit library adders)
def arith():
    rows = []
    for n in (2, 4, 8, 12, 16, 24, 32):
        for name, ctor in (('CDKM (Cuccaro)', lambda n: CDKMRippleCarryAdder(n, 'half')), ('VBE', lambda n: VBERippleCarryAdder(n, 'half'))):
            qc = ctor(n); nq, g = to_mct(qc); ours = qutrit_cost(g, nq); base = qiskit_all2all(qc)
            routed = qiskit_routed(qc, seeds=(0,)) if nq <= 70 else None
            rows.append(dict(adder=name, bits=n, qubits=nq, toffoli=ours['hist'].get('C2', 0), cnot=ours['hist'].get('C1', 0),
                             qutrit_pulses=ours['pulses'], qutrit_time_ns=ours['time_ns'], qiskit_n2q=base['n2q'], qiskit_time_ns=base['time2q_ns'],
                             routed_n2q=routed['n2q'] if routed else '', routed_time_ns=routed['time2q_ns'] if routed else ''))
            print(rows[-1], flush=True)
        jd(rows, 'part7_arith.json'); wcsv(rows, 'part7_arith.csv')

# ------------------------------------------------------------------ C. 3-SAT oracles
def sat_oracle(n, m, seed):
    rng = np.random.RandomState(seed); clauses = []
    for _ in range(m):
        v = rng.choice(n, 3, replace=False); s = rng.randint(0, 2, 3); clauses.append(list(zip(v, s)))
    N = n + m; qc = QuantumCircuit(N)
    def clause(i, c):
        for v, pos in c:
            if pos: qc.x(v)
        qc.append(MCXGate(3), [int(c[0][0]), int(c[1][0]), int(c[2][0]), n+i]); qc.x(n+i)        # OR via De Morgan
        for v, pos in c:
            if pos: qc.x(v)
    for i, c in enumerate(clauses): clause(i, c)
    qc.h(N-1); qc.append(MCXGate(m-1), list(range(n, N))); qc.h(N-1)                              # phase if all clauses true
    for i, c in reversed(list(enumerate(clauses))): clause(i, c)
    return qc

def sat():
    rows = []
    for n in (4, 6, 8, 10, 12, 14):
        m = int(round(4.26*n)); qc = sat_oracle(n, m, seed=n)
        nq, g = to_mct(qc); ours = qutrit_cost(g, nq); base = qiskit_all2all(qc)
        routed = qiskit_routed(qc, seeds=(0,)) if nq <= 40 else None
        rows.append(dict(variables=n, clauses=m, qubits=nq, qutrit_pulses=ours['pulses'], qutrit_time_ns=ours['time_ns'], hist=ours['hist'],
                         qiskit_n2q=base['n2q'], qiskit_time_ns=base['time2q_ns'], routed_n2q=routed['n2q'] if routed else '', routed_time_ns=routed['time2q_ns'] if routed else ''))
        print(rows[-1], flush=True)
        jd(rows, 'part7_sat.json'); wcsv(rows, 'part7_sat.csv')

# ------------------------------------------------------------------ AMM suite (arithmetic + Toffoli), with Nam et al. optimized outputs
def mct_aware(name, n, gates):
    """tof_k / barenco_tof_k implement one C^kX with Toffolis and ancillas: replace by one native C^kNOT"""
    import re
    m = re.match(r'(barenco_)?tof_(\d+)$', name)
    if not m: return None
    k = int(m.group(2)); p, T = qutrit_gate_cost('mcx', k); return dict(pulses=p, time_ns=T, k=k)

def amm():
    rows = []
    for f in sorted(glob.glob(os.path.join(AMM, '*_before'))):
        name = os.path.basename(f)[:-7].replace('%5E', '^')
        n, g = parse_quipper(f); ours = qutrit_cost(g, n); qc = to_qiskit(n, g)
        base = qiskit_all2all(qc); nam = quipper_cnot_stats(f.replace('_before', '_after_heavy'))
        routed = qiskit_routed(qc, seeds=(0,))
        aware = mct_aware(name, n, g)
        rows.append(dict(benchmark=name, qubits=n, toffoli=ours['hist'].get('C2', 0), cnot=ours['hist'].get('C1', 0),
                         qutrit_pulses=ours['pulses'], qutrit_time_ns=ours['time_ns'],
                         mct_aware_pulses=aware['pulses'] if aware else '', mct_aware_time_ns=aware['time_ns'] if aware else '',
                         qiskit_n2q=base['n2q'], qiskit_time_ns=base['time2q_ns'], nam_n2q=nam['n2q'], nam_time_ns=nam['time2q_ns'],
                         routed_n2q=routed['n2q'] if routed else '', routed_time_ns=routed['time2q_ns'] if routed else ''))
        print(rows[-1], flush=True)
        jd(rows, 'part7_amm.json'); wcsv(rows, 'part7_amm.csv')

if __name__ == '__main__':
    a = sys.argv[1:]; t0 = time.time()
    for s in ('grover', 'arith', 'sat', 'amm'):
        if s in a: globals()[s]()
    print('done [%.0fs]' % (time.time()-t0))

# ------------------------------------------------------------------ projection (C^2X_12 at 60 ns) and report figure
def projection():
    G = jl('part7_grover.json')
    for r in G:
        qc, _ = grover_iteration(r['n']); nq, g = to_mct(qc); c = qutrit_cost(g, nq, fast=True)
        r['qutrit_time_iter_ns_fast'] = c['time_ns']; r['qutrit_time_total_fast'] = c['time_ns']*r['iterations']
    jd(G, 'part7_grover.json'); wcsv(G, 'part7_grover.csv')
    S = jl('part7_sat.json')
    for r in S:
        qc = sat_oracle(r['variables'], r['clauses'], seed=r['variables']); nq, g = to_mct(qc); r['qutrit_time_ns_fast'] = qutrit_cost(g, nq, fast=True)['time_ns']
    jd(S, 'part7_sat.json'); wcsv(S, 'part7_sat.csv')
    A = jl('part7_amm.json')
    for r in A:
        f = os.path.join(AMM, r['benchmark'].replace('^', '%5E') + '_before'); n, g = parse_quipper(f)
        r['qutrit_time_ns_fast'] = qutrit_cost(g, n, fast=True)['time_ns']
        if r['mct_aware_pulses'] != '': r['mct_aware_time_ns_fast'] = 140.0*int(r['benchmark'].split('_')[-1]) - 80.0
    jd(A, 'part7_amm.json'); wcsv(A, 'part7_amm.csv')
    print('projection columns added')

def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    INK2, GRID = "#52514e", "#e6e5e1"; Gc, Bc, Oc, Vc, Yc = "#1baf7a", "#2a78d6", "#eb6834", "#8a6bd1", "#c9a227"
    plt.rcParams.update({"font.size": 7.2, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "legend.fontsize": 6})
    def clean(ax): ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    G = jl('part7_grover.json'); S = jl('part7_sat.json'); A = jl('part7_amm.json'); R = jl('part7_arith.json')
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    n = [r['n'] for r in G]
    ax = axs[0]
    ax.plot(n, [r['qutrit_pulses_iter'] for r in G], 'o-', ms=2.5, color=Gc, label='qutrit (this work)')
    ax.plot(n, [r['qiskit_n2q_iter'] for r in G], 's--', ms=2.5, color=Bc, label='Qiskit, no ancilla')
    ax.plot(n, [r['best_n2q_iter'] for r in G], 'd:', ms=2.5, color=Vc, label='best MCX synthesis (+anc.)')
    nr = [r['n'] for r in G if r['routed_n2q_iter'] != '']; ax.plot(nr, [r['routed_n2q_iter'] for r in G if r['routed_n2q_iter'] != ''], 'x-', ms=3, lw=0.8, color=Yc, label='Qiskit, heavy-hex routed')
    ax.set_yscale('log'); ax.set_xticks(range(3, 13, 3)); ax.set_xlabel('qubits $n$'); ax.set_ylabel('two-body pulses / iteration'); ax.set_title('(a) Grover iteration', loc='left'); clean(ax)
    ax = axs[1]
    ax.plot(n, [r['qutrit_time_iter_ns']/1e3 for r in G], 'o-', ms=2.5, color=Gc)
    ax.plot(n, [r['qutrit_time_iter_ns_fast']/1e3 for r in G], 'o:', ms=2.5, color=Gc, alpha=0.6, label=r'qutrit, $C^2X_{12}$ at 60 ns (proj.)')
    ax.plot(n, [r['qiskit_time_iter_ns']/1e3 for r in G], 's--', ms=2.5, color=Bc)
    ax.plot(n, [r['best_time_iter_ns']/1e3 for r in G], 'd:', ms=2.5, color=Vc)
    ax.set_xticks(range(3, 13, 3)); ax.set_xlabel('qubits $n$'); ax.set_ylabel(r'two-body time / iteration ($\mu$s)'); ax.set_title('(b) Grover iteration', loc='left'); clean(ax)
    ax = axs[2]
    SEL = ['rc_adder_6', 'vbe_adder_3', 'adder_8', 'qcla_adder_10', 'qcla_com_7', 'mod_red_21', 'mod_mult_55', 'mod_adder_1024',
           'gf2^8_mult', 'gf2^16_mult', 'mod5_4', 'tof_5', 'tof_10', 'barenco_tof_10']
    rows = [r for name in SEL for r in A if r['benchmark'] == name]
    lab = [r['benchmark'].replace('barenco_', 'b_') for r in rows]; y = np.arange(len(rows))
    ax.barh(y-0.2, [r['nam_n2q']/(r['mct_aware_pulses'] or r['qutrit_pulses']) for r in rows], 0.4, color=Gc, label='pulse ratio (Nam et al. / ours)')
    ax.barh(y+0.2, [r['nam_time_ns']/(r['mct_aware_time_ns'] or r['qutrit_time_ns']) for r in rows], 0.4, color=Oc, label='time ratio')
    ax.axvline(1, color=INK2, lw=0.7); ax.set_yticks(y); ax.set_yticklabels([l + ('*' if r['mct_aware_pulses'] != '' else '') for l, r in zip(lab, rows)], fontsize=5.6); ax.invert_yaxis(); ax.set_xlabel('advantage over optimized qubit circuit (x)')
    ax.set_title('(c) vs. Nam et al. (* MCT-aware)', loc='left'); clean(ax); ax.legend(frameon=False, loc='upper right', fontsize=5.5)
    h, l = axs[0].get_legend_handles_labels(); h2, l2 = axs[1].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, frameon=False, loc='lower center', ncol=5, fontsize=5.8, columnspacing=0.9, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.5); fig.savefig(fpath('dac_benchmarks.png'), dpi=300); plt.close(fig)
    print('figure ok')

if __name__ == '__main__':
    if 'projection' in sys.argv[1:]: projection()
    if 'report' in sys.argv[1:]: report()
