"""
Part 5 — C^k-NOT with 2k-1 two-qutrit gates versus IBM qubit-only decomposition
==============================================================================

Sections (command-line words):
  counts            : logic check (all 2^(k+1) inputs, k = 2..10), pulse count, depth and total pulse time of the
                      qutrit circuit; IBM decomposition WITHOUT routing (all-to-all, decomposition cost only).
  ibmdur            : IBM transpiled on the real coupling maps (heavy-hex: Kingston, Fez, Torino; square: Nighthawk)
                      with SWAP routing: duration, 2-qubit gates, single-qubit pulses, pulse depth (k = 2..8).
  qutrit K S        : noisy qutrit C^k-NOT for k in K (comma list) and scenario S in {measured, ibmlike}.
                      Basis inputs for all k; superposition (random states, named states) for k <= 5.
  ibmnoisy K B      : IBM noisy C^k-NOT (Aer superoperator, <= 6 active qubits) for k in K on backends B.
  plot              : figures.
Connectivity: the qutrit circuit only needs a LINE of k+1 neighbouring qutrits; heavy-hex and square lattices
contain such paths, so no SWAP is needed. If the qutrits were not on a path, each SWAP would cost the same
3 two-body gates as on IBM — we therefore compare without SWAPs (all-to-all, decomposition only) AND with the
routed IBM circuits.
Gokhale et al. (log-depth qutrit C^k-NOT): ≈6k two-qutrit + ≈7k single-qutrit gates (asymptotic count as an asymptotic estimate — to be checked against the paper); no duration is estimated because their depth,
not their count, sets the time.
Run e.g.: python3 p5_cknot.py counts ibmdur ; python3 p5_cknot.py qutrit 2,3,4,5 measured ;
          python3 p5_cknot.py ibmnoisy 2,3,4 FakeKingston,FakeFez ; python3 p5_cknot.py plot
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import os, sys, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import HERE
import circuit as cc
from noise import MEASURED, IBM_LIKE, IMPROVED

from paths import rpath, fpath, RESULTS as OUT, FIGURES as FIG, DATA, nproc
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a6bd1", "#c9a227", "#9c9a94"]
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
HERON = ['FakeKingston', 'FakeNighthawk', 'FakeFez', 'FakeTorino']
SCN = {'ideal': None, 'measured': MEASURED, 'improved': IMPROVED, 'ibmlike': IBM_LIKE}
def jl(name):
    f = rpath(name); return json.load(open(f)) if os.path.exists(f) else {}
def jd(obj, name): json.dump(obj, open(rpath(name), 'w'), indent=1)
def clean(ax):
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ('top', 'right')]

def counts():
    from ibm_tools import all_to_all_counts
    out = {}
    for k in range(2, 11):
        d = cc.durations(k); a2a = all_to_all_counts(k)
        out[k] = dict(logic_ok=cc.check_logic(k), qutrit=dict(total_pulses=d['n_two_qutrit'], **d),
                      ibm_all_to_all=a2a, ibm_all_to_all_ancilla=all_to_all_counts(k, ancilla=True), gokhale_estimate=dict(n_two_qutrit=6*k, n_single=7*k, total_pulses=13*k))
        print('k=%d logic %s | qutrit: %d pulses, %d ns | IBM all-to-all: %d CZ + %d 1q = %d pulses, depth %d' % (
            k, out[k]['logic_ok'], d['n_two_qutrit'], d['T_total_ns'], a2a['n2q'], a2a['n1q_pulses'], a2a['total_pulses'], a2a['depth_pulses']), flush=True)
    jd(out, 'part5_counts.json')

def ibmdur():
    from ibm_tools import best_transpile, pulse_counts, mcx_ancilla_circuit
    out = jl('part5_ibm_durations.json')
    for name in ('FakeKingston', 'FakeFez'):
        for k in range(3, 9):
            be, (d, seed, t, act) = best_transpile(k, name, nseed=20, qc=mcx_ancilla_circuit(k))
            out.setdefault(name+'_ancilla', {})[str(k)] = dict(duration_ns=d, active_qubits=len(act), **pulse_counts(t))
            print(name, 'ancilla', k, out[name+'_ancilla'][str(k)], flush=True)
        jd(out, 'part5_ibm_durations.json')
    for name in HERON:
        for k in range(2, 9):
            be, (d, seed, t, act) = best_transpile(k, name, nseed=30)
            out.setdefault(name, {})[str(k)] = dict(duration_ns=d, active_qubits=len(act), **pulse_counts(t))
            print(name, k, out[name][str(k)], flush=True)
        jd(out, 'part5_ibm_durations.json')

def qutrit(ks, scen, Q=8):
    out = jl('part5_qutrit_noisy.json'); coh = SCN[scen]
    for k in ks:
        t0 = time.time(); ch = cc.channels(k, coh)
        succ = cc.basis_success(k, ch, coh)
        r = dict(duration=cc.durations(k), succ_mean=float(np.mean(list(succ.values()))), succ_min=float(min(succ.values())))
        if k <= 5:
            phi = cc.fit_final_phases(k, cc.channels(k, None))
            r.update(cc.superposition_metrics(k, ch, coh, phi, Q=Q))
        out.setdefault(scen, {})[str(k)] = r
        print('k=%d %s: basis %.4f (min %.4f)%s [%.0fs]' % (k, scen, r['succ_mean'], r['succ_min'],
              ('  superposition %.4f' % r['F_random_mean']) if 'F_random_mean' in r else '', time.time()-t0), flush=True)
        jd(out, 'part5_qutrit_noisy.json')

def ibmnoisy(ks, names):
    from ibm_tools import noisy_eval
    out = jl('part5_ibm_noisy.json')
    for name in names:
        for k in ks:
            r = noisy_eval(k, name, nrand=40)
            out.setdefault(name, {})[str(k)] = r
            print(name, k, None if r is None else 'dur %.0f ns  basis %.4f  superposition %.4f' % (r['duration_ns'], r['succ_mean'], r['F_random_mean']), flush=True)
            jd(out, 'part5_ibm_noisy.json')

def plot():
    Cn = jl('part5_counts.json'); Dd = jl('part5_ibm_durations.json'); Qn = jl('part5_qutrit_noisy.json'); In = jl('part5_ibm_noisy.json')
    ks = sorted(int(k) for k in Cn)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    axs[0].plot(ks, [Cn[str(k)]['qutrit']['total_pulses'] for k in ks], 'o-', color=C[2], label='this work: 2k-1 (no ancilla, no 1-qutrit pulses)')
    axs[0].plot(ks, [Cn[str(k)]['ibm_all_to_all']['total_pulses'] for k in ks], 's-', color=C[0], label='IBM decomposition, all-to-all (CZ + sx/x)')
    kd = sorted(int(k) for k in Dd['FakeKingston'])
    axs[0].plot(kd, [Dd['FakeKingston'][str(k)]['total_pulses'] for k in kd], 's--', color=C[0], alpha=0.6, label='IBM Kingston, heavy-hex routed')
    axs[0].plot(ks, [Cn[str(k)]['ibm_all_to_all_ancilla']['total_pulses'] for k in ks], 'D-', color=C[3], label='IBM with k-2 ancilla qubits, all-to-all')
    axs[0].plot(ks, [Cn[str(k)]['gokhale_estimate']['total_pulses'] for k in ks], '^-', color=C[1], label='Gokhale et al. (≈13k, est.)')
    axs[0].set_yscale('log'); axs[0].set_xlabel('number of controls k'); axs[0].set_ylabel('physical pulses'); axs[0].legend(frameon=False, fontsize=7.5)
    axs[1].plot(ks, [Cn[str(k)]['qutrit']['T_total_ns'] for k in ks], 'o-', color=C[2], label='this work (incl. 10 ns flux moves)')
    for i, n in enumerate(HERON):
        if n in Dd:
            kk = sorted(int(k) for k in Dd[n]); axs[1].plot(kk, [Dd[n][str(k)]['duration_ns'] for k in kk], 's-', color=[C[0], C[3], C[5], C[4]][i], lw=1, label='IBM %s (routed)' % n.replace('Fake', ''))
    if 'FakeKingston_ancilla' in Dd:
        kk = sorted(int(k) for k in Dd['FakeKingston_ancilla']); axs[1].plot(kk, [Dd['FakeKingston_ancilla'][str(k)]['duration_ns'] for k in kk], 'D--', color=C[3], lw=1, label='IBM Kingston, k-2 ancillas (routed)')
    axs[1].set_yscale('log'); axs[1].set_xlabel('number of controls k'); axs[1].set_ylabel('total pulse time (ns)'); axs[1].legend(frameon=False, fontsize=7.5)
    # coherence budget: duration / T1
    T1_ibm = {'FakeKingston': 283, 'FakeNighthawk': 273, 'FakeFez': 145, 'FakeTorino': 185}
    axs[2].plot(ks, [Cn[str(k)]['qutrit']['T_total_ns']/17e3 for k in ks], 'o-', color=C[2], label='this work / T1(2->1) = 17 us (reference device)')
    axs[2].plot(ks, [Cn[str(k)]['qutrit']['T_total_ns']/27e3 for k in ks], 'o--', color=C[2], alpha=0.6, label='this work / T1(1->0) = 27 us')
    for i, n in enumerate(HERON):
        if n in Dd:
            kk = sorted(int(k) for k in Dd[n]); axs[2].plot(kk, [Dd[n][str(k)]['duration_ns']/(T1_ibm[n]*1e3) for k in kk], 's-', lw=1, color=[C[0], C[3], C[5], C[4]][i], label='IBM %s / median T1' % n.replace('Fake', ''))
    axs[2].set_yscale('log'); axs[2].set_xlabel('number of controls k'); axs[2].set_ylabel('circuit time / T1 (coherence budget used)'); axs[2].legend(frameon=False, fontsize=7)
    for a in axs: clean(a)
    fig.suptitle('C^k-NOT: pulse count, pulse time and fraction of the coherence time used', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(fpath('p5_cknot_pulses_time_budget.png'), dpi=200); plt.close(fig)
    # noisy
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, met, lab in ((axs[0], 'succ_mean', 'basis-input success'), (axs[1], 'F_random_mean', 'superposition fidelity (random states)')):
        for scen, col, ls in (('ideal', INK2, ':'), ('measured', C[2], '-'), ('improved', C[3], '-'), ('ibmlike', C[1], '-')):
            if scen in Qn:
                kk = sorted(int(k) for k in Qn[scen] if met in Qn[scen][k])
                ax.plot(kk, [Qn[scen][str(k)][met] for k in kk], 'o'+ls, color=col, label={'ideal': 'qutrit, closed system', 'measured': 'qutrit, reference device (T1 27/17 us)', 'improved': 'qutrit, T1 100/60 us', 'ibmlike': 'qutrit, IBM-like coherence'}[scen])
        for i, n in enumerate(In):
            kk = sorted(int(k) for k in In[n] if In[n][k]); ax.plot(kk, [In[n][str(k)][met] for k in kk], 's--', lw=1, color=[C[0], C[5], C[4], '#555'][i % 4], label='IBM %s' % n.replace('Fake', ''))
        ax.set_xlabel('number of controls k'); ax.set_ylabel(lab); clean(ax); ax.set_xticks(range(2, 9 if met == 'succ_mean' else 6))
    axs[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle('Noisy C^k-NOT: qutrit (decoherence only) versus IBM noise models', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(fpath('p5_cknot_noisy_vs_ibm.png'), dpi=200); plt.close(fig)

if __name__ == '__main__':
    a = sys.argv[1:]; t0 = time.time()
    if 'counts' in a: counts()
    if 'ibmdur' in a: ibmdur()
    if 'qutrit' in a: i = a.index('qutrit'); qutrit([int(v) for v in a[i+1].split(',')], a[i+2])
    if 'ibmnoisy' in a: i = a.index('ibmnoisy'); ibmnoisy([int(v) for v in a[i+1].split(',')], a[i+2].split(','))
    if 'plot' in a: plot()
    print('done [%.0fs]' % (time.time()-t0))
