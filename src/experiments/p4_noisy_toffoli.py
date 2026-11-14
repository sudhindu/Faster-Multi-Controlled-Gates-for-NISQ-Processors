"""
Part 4 — Noisy Toffoli (qutrit, Lindblad) versus IBM devices (Aer noise models)
==============================================================================

Qutrit scenarios (density-matrix simulation of the full 3-qutrit circuit, see circuit.py):
  ideal (closed system), measured device (T1 27/17 µs, Tφ 40 µs), same but |2> with the |1> lifetime
  (isolates the cost of the fast |2> decay), improved (100/60/150 µs), IBM-like coherence (200/100/200 µs),
  measured + residual calibrated errors (Monte-Carlo), measured + uncalibrated stress errors (Monte-Carlo).
Metrics: 8 basis-input success probabilities, exact qubit-subspace process fidelity F_pro, average fidelity
  F_avg = (8 F_pro + retained)/9, fidelity on named superposition inputs.
T1 sweep: F_avg and basis success versus qutrit T1 (|2> lifetime scaled by 17/27) -> plotted together with the
  IBM devices at THEIR median T1, showing at which coherence the short qutrit circuit matches IBM.
IBM: ibm_tools.noisy_eval (transpiled, scheduled, gate + idle relaxation noise, readout off).

Run:  python3 p4_noisy_toffoli.py [scen] [sweep] [ibm] [plot]      (default: all; ~10 min)
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import os, sys, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multiprocessing import Pool
from common import HERE
import circuit as cc
from noise import coherence, MEASURED, IMPROVED, IBM_LIKE

from paths import rpath, fpath, RESULTS as OUT, FIGURES as FIG, DATA, nproc
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a6bd1", "#c9a227", "#9c9a94"]
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
K = 2
MC = {'residual': dict(f=0.05, g=0.002, a=0.002), 'stress': dict(f=0.30, g=0.02, a=0.01)}
SCEN = {'ideal (closed system)': (None, None), 'measured T1 27/17 us, Tphi 40 us': (MEASURED, None),
        'measured, but |2> as long-lived as |1>': (coherence(27, 27, 40), None),
        'improved 100/60/150 us': (IMPROVED, None), 'IBM-like coherence 200/100/200 us': (IBM_LIKE, None),
        'measured + residual errors (flux 0.05 MHz, coupler 0.2 %)': (MEASURED, 'residual'),
        'measured + stress errors (flux 0.3 MHz, coupler 2 %)': (MEASURED, 'stress')}
PHI = None
def init():
    global PHI
    PHI = cc.fit_final_phases(K, cc.channels(K, None))

def evaluate(coh, errs, seed=0):
    ch = cc.channels(K, coh, errs)
    succ = cc.basis_success(K, ch, coh)
    Fp = cc.process_fidelity_k2(ch, coh, PHI)
    sm = cc.superposition_metrics(K, ch, coh, PHI, Q=40, seed=seed)
    ret = 1.0  # retained trace averaged over basis inputs
    return dict(succ=succ, succ_mean=float(np.mean(list(succ.values()))), succ_min=float(min(succ.values())), F_pro=Fp, **sm)

def _job(a):
    name, seed = a; coh, lvl = SCEN[name]
    errs = None if lvl is None else cc.sample_errors(K, np.random.RandomState(500+seed), MC[lvl])
    return name, evaluate(coh, errs, seed)

def scenarios(Kmc=8):
    jobs = [(n, s) for n, (c, l) in SCEN.items() for s in (range(Kmc) if l else [0])]
    res = {}
    with Pool(nproc(), initializer=init) as pool:
        for name, m in pool.imap_unordered(_job, jobs): res.setdefault(name, []).append(m)
    out = {}
    for name, v in res.items():
        out[name] = dict(K=len(v), succ={k: float(np.mean([m['succ'][k] for m in v])) for k in v[0]['succ']},
                         succ_mean=float(np.mean([m['succ_mean'] for m in v])), succ_std=float(np.std([m['succ_mean'] for m in v])),
                         F_pro=float(np.mean([m['F_pro'] for m in v])), F_pro_std=float(np.std([m['F_pro'] for m in v])),
                         F_random_mean=float(np.mean([m['F_random_mean'] for m in v])),
                         named={k: float(np.mean([m['named'][k] for m in v])) for k in v[0]['named']})
        print('%-58s basis %.4f  F_pro %.4f±%.4f  F_random %.4f' % (name, out[name]['succ_mean'], out[name]['F_pro'], out[name]['F_pro_std'], out[name]['F_random_mean']), flush=True)
    json.dump(dict(duration=cc.durations(K), scenarios=out), open(rpath('part4_qutrit_scenarios.json'), 'w'), indent=1)

T1S = [10, 15, 20, 27, 40, 60, 100, 150, 200, 300]
def _sweep_job(T1):
    coh = coherence(T1, T1*17/27, T1*40/27); m = evaluate(coh, None)
    return T1, m
def sweep():
    out = {}
    with Pool(nproc(), initializer=init) as pool:
        for T1, m in pool.imap_unordered(_sweep_job, T1S):
            out['%g' % T1] = m; print('T1 %g: basis %.4f F_pro %.4f F_random %.4f' % (T1, m['succ_mean'], m['F_pro'], m['F_random_mean']), flush=True)
    json.dump(out, open(rpath('part4_T1_sweep.json'), 'w'), indent=1)

def ibm():
    from ibm_tools import BACKENDS, noisy_eval
    out = {}
    for name in BACKENDS:
        out[name] = noisy_eval(2, name, nrand=200); r = out[name]
        print('%-14s %5.0f ns  T1 %.0f us  basis %.4f  F_random %.4f  |+++> %.4f' % (name, r['duration_ns'], r['T1_median_us'], r['succ_mean'], r['F_random_mean'], r['F_plus']), flush=True)
    json.dump(out, open(rpath('part4_ibm_noisy.json'), 'w'), indent=1)

def plot():
    S = json.load(open(rpath('part4_qutrit_scenarios.json')))['scenarios']
    I = json.load(open(rpath('part4_ibm_noisy.json'))); W = json.load(open(rpath('part4_T1_sweep.json')))
    Tq = cc.durations(K)['T_total_ns']
    # (a) fidelity versus T1: qutrit curve vs IBM devices at their own T1
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    T1 = sorted(map(float, W)); key = lambda t: '%g' % t
    for ax, metric, lab in ((axs[0], 'succ_mean', 'basis-input success (8 inputs)'), (axs[1], 'F_random_mean', 'superposition fidelity (random 3-qubit states)')):
        ax.plot(T1, [W[key(t)][metric] for t in T1], 'o-', color=C[2], lw=1.8, label='qutrit Toffoli, %d ns (|2> T1 = 0.63 T1)' % Tq)
        for i, (n, r) in enumerate(I.items()):
            ax.scatter([r['T1_median_us']], [r['succ_mean'] if metric == 'succ_mean' else r['F_random_mean']], s=46, color=C[0], marker='s', zorder=3)
            ax.annotate('%s\n%.0f ns' % (n.replace('Fake', 'IBM '), r['duration_ns']), (r['T1_median_us'], r['succ_mean'] if metric == 'succ_mean' else r['F_random_mean']),
                        textcoords='offset points', xytext=(6, -14 if i % 2 else 4), fontsize=7, color=INK2)
        ax.axvline(27, color=INK2, ls='--', lw=0.8); ax.text(27.5, ax.get_ylim()[0] if False else 0.94, ' reference device', fontsize=7.5, color=INK2)
        ax.set_xscale('log'); ax.set_xlabel(r'$T_1$ ($\mu$s)  [qutrit: $T_1(1\to0)$;  IBM: median qubit $T_1$]'); ax.set_ylabel(lab)
        ax.set_ylim(0.9, 1.0); ax.grid(color=GRID); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    axs[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Toffoli under decoherence: short qutrit pulses versus longer qubit schedules', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(fpath('p4_fidelity_vs_T1_qutrit_vs_ibm.png'), dpi=200); plt.close(fig)
    # (b) per-input success
    fig, ax = plt.subplots(figsize=(11, 3.9)); ins = list(next(iter(S.values()))['succ'])
    series = [('qutrit, measured T1 27/17 us', S['measured T1 27/17 us, Tphi 40 us']['succ'], C[2]),
              ('qutrit, IBM-like coherence', S['IBM-like coherence 200/100/200 us']['succ'], C[3]),
              ('IBM Kingston', dict(zip(ins, I['FakeKingston']['succ'])), C[0]), ('IBM Fez', dict(zip(ins, I['FakeFez']['succ'])), C[5])]
    w = 0.2
    for j, (lab, d, col) in enumerate(series): ax.bar(np.arange(8)+(j-1.5)*w, [d[k] for k in ins], w, color=col, label=lab)
    ax.set_xticks(range(8)); ax.set_xticklabels(['|%s>' % k for k in ins]); ax.set_ylim(0.93, 1.0); ax.set_ylabel('success probability')
    ax.grid(axis='y', color=GRID); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    fig.legend(frameon=False, loc='lower center', ncol=4, fontsize=8.5)
    fig.suptitle('Noisy Toffoli: success probability for the 8 computational inputs', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93)); fig.savefig(fpath('p4_toffoli_per_input_noisy.png'), dpi=200); plt.close(fig)
    # (c) scenarios bar chart
    fig, ax = plt.subplots(figsize=(11, 4.2)); names = list(S); x = np.arange(len(names))
    ax.bar(x-0.2, [S[n]['succ_mean'] for n in names], 0.38, color=C[0], label='basis inputs (success)')
    ax.bar(x+0.2, [S[n]['F_pro'] for n in names], 0.38, yerr=[S[n]['F_pro_std'] for n in names], color=C[1], label=r'superposition ($F_{pro}$)', error_kw=dict(ecolor=INK2))
    ax.set_xticks(x); ax.set_xticklabels([n.replace(' (', '\n(').replace(', T', '\nT').replace(' + ', '\n+ ') for n in names], fontsize=7)
    ax.set_ylim(0.5, 1.0); ax.grid(axis='y', color=GRID); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    fig.legend(frameon=False, loc='lower center', ncol=2, fontsize=8.5)
    fig.suptitle('Noisy qutrit Toffoli (%d ns): scenarios' % Tq, x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93)); fig.savefig(fpath('p4_toffoli_scenarios.png'), dpi=200); plt.close(fig)

if __name__ == '__main__':
    what = sys.argv[1:] or ['scen', 'sweep', 'ibm', 'plot']; t0 = time.time()
    if 'scen' in what: scenarios()
    if 'sweep' in what: sweep()
    if 'ibm' in what: ibm()
    if 'plot' in what: plot()
    print('done [%.0fs]' % (time.time()-t0))
