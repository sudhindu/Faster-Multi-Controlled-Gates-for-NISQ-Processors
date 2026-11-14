"""
Part 3 — Ideal Toffoli from the three-gate construction, and time/pulse comparison
=================================================================================

Circuit (qubit inputs a,b,c on qutrits q0,q1,q2; q1 is the "middle" qutrit):
    C^1X_12(q0 -> q1)   : if a=1, q1: |1> <-> |2>      (q1 = 2  <=>  a=b=1)
    C^2X_01(q1 -> q2)   : if q1=2, flip target c
    C^1X_12(q0 -> q1)   : undo
q1 changes role (target of gate 1, control of gate 2): it is flux-moved between 5.20 and 5.00 GHz,
assumed 10 ns per move (2 moves). The gates themselves come from Part 1 (lab frame, 4 levels).

Outputs: success probability of the 8 computational inputs, process fidelity of the 8-dim (qubit-subspace)
Toffoli with 3 free final virtual-Z phases, total pulse time, pulse count; comparison with IBM devices
(transpiled schedules) and Galda et al. / Nikolaeva et al.
Run: python3 p3_ideal_toffoli.py      (needs results/part1_logical_blocks.npz from Part 1)
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import os, json, itertools, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize
from common import *

from paths import rpath, fpath, RESULTS as OUT, FIGURES as FIG, DATA, nproc
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a6bd1", "#c9a227"]
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
T_FLUX = 10.0                     # ns per flux move of the middle qutrit (assumption)
T_1Q_ASSUMED = 25.0               # ns, single-qutrit pulse used only for the same-platform estimates of other schemes

def lift(L9, ctrl, tgt, nq=3):
    """embed a 9x9 two-qutrit logical block acting on (ctrl,tgt) into nq qutrits"""
    dim = 3**nq; M = np.zeros((dim, dim), complex)
    for a in itertools.product(range(3), repeat=nq):
        for b in itertools.product(range(3), repeat=nq):
            if all(a[q] == b[q] for q in range(nq) if q not in (ctrl, tgt)):
                M[np.ravel_multi_index(b, (3,)*nq), np.ravel_multi_index(a, (3,)*nq)] = \
                    L9[KEYS9.index((b[ctrl], b[tgt])), KEYS9.index((a[ctrl], a[tgt]))]
    return M

BITS3 = list(itertools.product((0, 1), repeat=3)); IDX8 = [np.ravel_multi_index(b, (3, 3, 3)) for b in BITS3]
def toff(b): return (b[0], b[1], b[2] ^ (b[0] & b[1]))

def toffoli_metrics(V):
    Vq = V[np.ix_(IDX8, IDX8)]
    Tm = np.zeros((8, 8))
    for i, b in enumerate(BITS3): Tm[BITS3.index(toff(b)), i] = 1
    succ = [float(abs(Vq[BITS3.index(toff(b)), i])**2) for i, b in enumerate(BITS3)]
    def F(phi):
        d = np.exp(1j*np.array([sum(phi[q]*b[q] for q in range(3)) for b in BITS3]))
        return abs(np.trace(Tm.T @ (d[:, None]*Vq)))**2/64
    best = max((minimize(lambda p: -F(p), s0, method='BFGS') for s0 in [np.zeros(3)] + [np.random.RandomState(s).uniform(-3, 3, 3) for s in range(4)]), key=lambda r: -r.fun)
    Fe = -best.fun; ret = float(np.mean([np.sum(abs(V[:, j])**2) for j in IDX8]))
    return dict(succ={''.join(map(str, b)): s for b, s in zip(BITS3, succ)}, succ_mean=float(np.mean(succ)), succ_min=float(min(succ)),
                F_pro=float(Fe), F_avg=float((8*Fe+ret)/9), leak=1-ret, final_virtual_Z=best.x.tolist())

if __name__ == '__main__':
    B = np.load(rpath('part1_logical_blocks.npz')); P = load_params()
    Lc = {g: np.diag(phase_vector(B[g+'_phases'])) @ B[g] for g in ('C1X12', 'C2X01')}
    V = lift(Lc['C1X12'], 0, 1) @ lift(Lc['C2X01'], 1, 2) @ lift(Lc['C1X12'], 0, 1)
    m = toffoli_metrics(V)
    T_gates = 2*P['C1X12']['T_ns'] + P['C2X01']['T_ns']
    ours = dict(T_gates_ns=T_gates, T_total_ns=T_gates + 2*T_FLUX, n_two_body=3, n_single=0, total_pulses=3, depth=3, **m)
    print('qutrit Toffoli: gates %d ns, total %d ns, success mean %.4f (min %.4f), F_pro %.4f' % (T_gates, ours['T_total_ns'], m['succ_mean'], m['succ_min'], m['F_pro']))
    # ---------------- IBM (ideal schedules: duration and pulse counts only)
    from ibm_tools import BACKENDS, best_transpile, pulse_counts, all_to_all_counts, device_coherence
    ibm = {}
    for name in BACKENDS:
        be, (d, seed, t, act) = best_transpile(2, name, nseed=30)
        ibm[name] = dict(duration_ns=d, **pulse_counts(t), T1_median_us=device_coherence(be)[0])
        print(name, ibm[name])
    a2a = all_to_all_counts(2)
    # ---------------- other qutrit schemes (reported time / pulse count from their circuits; check against the papers)
    others = {
        'Galda et al.': dict(T_total_ns=1590, total_pulses=12, n_two_body=4, n_single=8, source='reported time; pulse count from their circuit'),
        'Nikolaeva et al.': dict(T_total_ns=None, total_pulses=8, n_two_body=4, n_single=4, source='pulse count from their circuit'),
        'Galda-type on this platform (estimate)': dict(T_total_ns=4*60+8*T_1Q_ASSUMED, total_pulses=12, n_two_body=4, n_single=8,
                                                       source='4 two-qutrit x 60 ns + 8 single-qutrit x %.0f ns (assumed)' % T_1Q_ASSUMED),
        'Nikolaeva-type on this platform (estimate)': dict(T_total_ns=4*60+4*T_1Q_ASSUMED, total_pulses=8, n_two_body=4, n_single=4,
                                                           source='4 two-qutrit x 60 ns + 4 single-qutrit x %.0f ns (assumed)' % T_1Q_ASSUMED)}
    json.dump(dict(ours=ours, ibm=ibm, ibm_all_to_all=a2a, others=others), open(rpath('part3_toffoli_ideal.json'), 'w'), indent=1)
    # ---------------- figures
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ks = list(m['succ']); ax.bar(range(8), [m['succ'][k] for k in ks], color=C[0], width=0.6)
    for i, k in enumerate(ks): ax.text(i, m['succ'][k]+0.0004, '%.4f' % m['succ'][k], ha='center', fontsize=7.5)
    ax.set_xticks(range(8)); ax.set_xticklabels(['|%s>' % k for k in ks]); ax.set_ylim(0.97, 1.0); ax.set_ylabel('success probability')
    ax.set_title('Ideal qutrit Toffoli (lab-frame gates), %d ns:  mean %.4f,  F_pro %.4f' % (ours['T_total_ns'], m['succ_mean'], m['F_pro']), loc='left', fontsize=10)
    ax.grid(axis='y', color=GRID); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    fig.tight_layout(); fig.savefig(fpath('p3_toffoli_8_inputs.png'), dpi=200); plt.close(fig)

    rows = [('This work\n(coupler, qutrit)', ours['T_total_ns'], 3, C[2])] + \
           [('IBM %s' % n.replace('Fake', ''), ibm[n]['duration_ns'], ibm[n]['total_pulses'], C[0]) for n in BACKENDS] + \
           [('Galda et al.\n(reported)', 1590, 12, C[1]), ('Galda-type here\n(estimate)', others['Galda-type on this platform (estimate)']['T_total_ns'], 12, C[1]),
            ('Nikolaeva-type\nhere (estimate)', others['Nikolaeva-type on this platform (estimate)']['T_total_ns'], 8, C[3])]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    y = np.arange(len(rows))
    axs[0].barh(y, [r[1] for r in rows], color=[r[3] for r in rows]); axs[0].set_xscale('log')
    for yy, r in zip(y, rows): axs[0].text(r[1]*1.05, yy, '%.0f ns' % r[1], va='center', fontsize=8)
    axs[0].set_yticks(y); axs[0].set_yticklabels([r[0] for r in rows], fontsize=8); axs[0].invert_yaxis(); axs[0].set_xlabel('Toffoli duration (ns, log)')
    axs[1].barh(y, [r[2] for r in rows], color=[r[3] for r in rows])
    for yy, r in zip(y, rows): axs[1].text(r[2]+0.3, yy, '%d' % r[2], va='center', fontsize=8)
    axs[1].set_yticks(y); axs[1].set_yticklabels([]); axs[1].invert_yaxis(); axs[1].set_xlabel('physical pulses (two-body + single-body; virtual Z free)')
    for a in axs: a.grid(axis='x', color=GRID); [a.spines[s].set_visible(False) for s in ('top', 'right')]
    fig.suptitle('Toffoli: total pulse time and pulse count (ideal schedules)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(fpath('p3_toffoli_time_pulses.png'), dpi=200); plt.close(fig)
