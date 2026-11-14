"""
Part 2 — Noisy (Lindblad) simulation of the three gates
=======================================================

Sections (select with command-line words; default runs all):
  coherence : F_pro, mean population and leakage versus coherence (T1 sweep, |2> lifetime = T1·17/27),
              plus the measured device (27/17/40 µs) and a control run where |2> has the SAME T1 as |1>
              -> isolates the cost of the faster |2> decay.
  perinput  : per-input populations (9 inputs) with measured coherence.
  flux      : closed-system tolerance to quasi-static frequency offsets of target / control (flux noise)
              and to coupler-strength errors, uncalibrated (phases fixed) vs re-calibrated
              (drive frequency + virtual-Z phases re-tuned, as done in routine calibration).
  mc        : Monte-Carlo: measured decoherence + quasi-static flux/coupler/amplitude errors
              (two levels: 'residual after calibration' and 'uncalibrated stress test').
  validate  : cross-check of the numpy Lindblad propagator against QuTiP propagator(H, T, c_ops) (~1-3 min).

Run: python3 p2_noisy_gates.py [coherence perinput flux mc validate]
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import os, sys, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from multiprocessing import Pool
from common import *
from noise import *

from paths import rpath, fpath, RESULTS as OUT, FIGURES as FIG, DATA, nproc
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COL = {'C1X12': "#2a78d6", 'C2X01': "#eb6834", 'C2X12': "#1baf7a"}
NAME = {'C1X12': r'$C^1X_{12}$', 'C2X01': r'$C^2X_{01}$', 'C2X12': r'$C^2X_{12}$'}
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
P = load_params(); PH = {}
def ph(g):
    if g not in PH: PH[g] = rwa_phases(g, P[g]['T_ns'], P[g]['x'])
    return PH[g]
def clean(ax):
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def jload(name):
    f = rpath(name)
    return json.load(open(f)) if os.path.exists(f) else {}
def noisy_metrics(g, coh, err=None):
    S = gate_superop(g, P[g]['T_ns'], P[g]['x'], coh, err)
    return channel_metrics(logical_channel(S, ph(g)), g)

# ------------------------------------------------------------------ coherence sweep
T1S = [10, 20, 27, 50, 100, 200, 300]
def _coh_job(a):
    g, T1, same2 = a
    coh = coherence(T1, T1 if same2 else T1*17/27, T1*40/27)
    return g, T1, same2, noisy_metrics(g, coh)
def coherence_sweep():
    jobs = [(g, T1, False) for g in P for T1 in T1S] + [(g, 27, True) for g in P]
    res = {}
    with Pool(nproc()) as pool:
        for g, T1, same2, m in pool.imap_unordered(_coh_job, jobs):
            res.setdefault(g, {})['%g%s' % (T1, '_same_T1_for_2' if same2 else '')] = m
            print('coherence', g, T1, same2, 'F_pro %.4f pop %.4f leak %.4f' % (m['F_pro'], m['pop_mean'], m['leak']), flush=True)
    json.dump(res, open(rpath('part2_coherence.json'), 'w'), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for g in P:
        F = [res[g]['%g' % t]['F_pro'] for t in T1S]; pp = [res[g]['%g' % t]['pop_mean'] for t in T1S]
        ax[0].plot(T1S, F, 'o-', color=COL[g], label=NAME[g] + ' (%d ns)' % P[g]['T_ns'])
        ax[1].plot(T1S, pp, 'o-', color=COL[g], label=NAME[g])
    for a, yl in zip(ax, (r'process fidelity $F_{pro}$ (superposition)', 'mean basis-input population')):
        a.set_xscale('log'); a.set_xlabel(r'$T_1(1\to0)$ ($\mu$s);  $T_1(2\to1)=0.63\,T_1$,  $T_\phi=1.48\,T_1$'); a.set_ylabel(yl)
        a.axvline(27, color=INK2, ls='--', lw=0.8); a.text(28, a.get_ylim()[0] if False else 0.0, '', fontsize=7)
        clean(a)
    ax[0].text(28, ax[0].get_ylim()[0]+0.002, 'measured\ndevice', fontsize=7.5, color=INK2)
    ax[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Gate fidelity versus coherence (Lindblad, 4 levels, faster decay of |2>)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(fpath('p2_coherence_sweep.png'), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ per-input bars
def per_input():
    res = {}
    for g in P:
        res[g] = dict(measured=noisy_metrics(g, MEASURED), ideal=noisy_metrics(g, None))
        print('perinput', g, 'measured F_pro %.4f pop %.4f' % (res[g]['measured']['F_pro'], res[g]['measured']['pop_mean']), flush=True)
    json.dump(res, open(rpath('part2_per_input.json'), 'w'), indent=1)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    labs = ["|%d,%d>" % k for k in KEYS9]
    for ax, g in zip(axs, P):
        ide = [res[g]['ideal']['per_input'][l] for l in labs]; me = [res[g]['measured']['per_input'][l] for l in labs]
        xx = np.arange(9)
        ax.bar(xx-0.2, ide, 0.38, color=COL[g], alpha=0.45, label='closed system')
        ax.bar(xx+0.2, me, 0.38, color=COL[g], label='T1 27/17 µs, Tφ 40 µs')
        ax.set_xticks(xx); ax.set_xticklabels(labs, rotation=60, fontsize=8); ax.set_ylim(0.9, 1.003)
        ax.set_title('%s  (%d ns)   F_pro %.3f -> %.3f' % (NAME[g], P[g]['T_ns'], res[g]['ideal']['F_pro'], res[g]['measured']['F_pro']), fontsize=9.5, loc='left')
        clean(ax)
    axs[0].set_ylabel('probability of ideal output'); axs[0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.tight_layout(); fig.savefig(fpath('p2_per_input_noisy.png'), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ flux / coupler tolerance (closed system)
def F_closed(g, err, recal):
    T, x = P[g]['T_ns'], P[g]['x']
    def f(dwd):
        e = dict(err); e['dwd'] = dwd
        L = unitary_rwa(g, T, x, dt=0.1, err=e)[np.ix_(LOGICAL, LOGICAL)]
        return fpro(L, g, ph(g)) if not recal else best_phases(L, g, ph(g))[1]
    if not recal: return f(0.0)
    r = minimize_scalar(lambda d: -f(d), bounds=(-6, 6), method='bounded', options=dict(xatol=0.02))
    return max(-r.fun, f(0.0))
def _tol_job(a):
    g, key, v, recal = a
    return g, key, v, recal, F_closed(g, {key: v}, recal)
def tolerance():
    grids = dict(dft=np.linspace(-0.5, 0.5, 11), dfc=np.linspace(-0.5, 0.5, 11), gsc=np.linspace(-0.03, 0.03, 13))
    jobs = [(g, k, float(v), rc) for g in P for k, vs in grids.items() for v in vs for rc in (False, True)]
    res = {}
    with Pool(nproc()) as pool:
        for g, k, v, rc, F in pool.imap_unordered(_tol_job, jobs):
            res.setdefault(g, {}).setdefault(k, {}).setdefault('recal' if rc else 'fixed', {})['%.4f' % v] = F
    json.dump(res, open(rpath('part2_tolerance.json'), 'w'), indent=1)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    ttl = dict(dft='target frequency offset (MHz) — flux noise', dfc='control frequency offset (MHz)', gsc='coupler-strength error (%)')
    for ax, k in zip(axs, ('dft', 'dfc', 'gsc')):
        for g in P:
            for rc, ls in (('fixed', '--'), ('recal', '-')):
                d = res[g][k][rc]; xs = sorted(d, key=float)
                xv = [float(s)*(100 if k == 'gsc' else 1) for s in xs]
                ax.plot(xv, [d[s] for s in xs], ls, color=COL[g], lw=1.5, label='%s %s' % (NAME[g], 'recalibrated' if rc == 'recal' else 'no recalibration'))
        ax.set_xlabel(ttl[k]); clean(ax)
    axs[0].set_ylabel(r'$F_{pro}$ (closed system)'); axs[0].set_ylim(0.3, 1.005)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle('Tolerance to quasi-static flux-tuning and coupler errors', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94)); fig.savefig(fpath('p2_flux_coupler_tolerance.png'), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ Monte Carlo
MC_LEVELS = {'residual (after calibration): flux 0.05 MHz, coupler 0.2 %, amp 0.2 %': dict(f=0.05, g=0.002, a=0.002),
             'stress (uncalibrated): flux 0.3 MHz, coupler 2 %, amp 1 %': dict(f=0.30, g=0.02, a=0.01)}
def _mc_job(a):
    g, lvl, seed = a; s = MC_LEVELS[lvl]; rng = np.random.RandomState(seed)
    err = dict(dft=rng.randn()*s['f'], dfc=rng.randn()*s['f'], gsc=rng.randn()*s['g'], amp=rng.randn()*s['a'], goff=0.0)
    return g, lvl, noisy_metrics(g, MEASURED, err)
def montecarlo(Kmc=8):
    res = {}
    jobs = [(g, l, s) for g in P for l in MC_LEVELS for s in range(Kmc)]
    with Pool(nproc()) as pool:
        for g, l, m in pool.imap_unordered(_mc_job, jobs): res.setdefault(g, {}).setdefault(l, []).append(m)
    summ = {g: {l: dict(F_pro=float(np.mean([m['F_pro'] for m in v])), F_pro_std=float(np.std([m['F_pro'] for m in v])),
                        pop_mean=float(np.mean([m['pop_mean'] for m in v])), leak=float(np.mean([m['leak'] for m in v])), K=len(v))
                for l, v in d.items()} for g, d in res.items()}
    json.dump(summ, open(rpath('part2_montecarlo.json'), 'w'), indent=1)
    for g in summ:
        for l, v in summ[g].items(): print('MC', g, l, 'F_pro %.4f±%.4f pop %.4f' % (v['F_pro'], v['F_pro_std'], v['pop_mean']))

def validate():
    import qutip as qt
    g = 'C1X12'; T, x = P[g]['T_ns'], P[g]['x']; dt = 0.05
    pu = pulses(T, x, dt); wd = drive_freq(g, x)
    H = qt.QobjEvo([Q(h0_rot(P0, wd)), [Q(CPL_RWA), pu['fg']], [Q(HXt), pu['fI']], [Q(HYt), pu['fQ']], [Q(NT), pu['fD']]], tlist=pu['t'])
    t0 = time.time()
    Sq = qt.propagator(H, T, c_ops=[Q(c) for c in collapse_ops(MEASURED)], options=dict(nsteps=200000, atol=1e-9, rtol=1e-8, max_step=0.05))
    E0 = np.real(np.diag(h0_rot(P0, wd))); phv = np.exp(1j*E0*T)
    Sq = np.kron(phv.conj(), phv)[:, None]*Sq.full()
    mq = channel_metrics(logical_channel(Sq, ph(g)), g); mn = channel_metrics(logical_channel(gate_superop(g, T, x, MEASURED, dt=dt), ph(g)), g)
    out = dict(qutip=dict(F_pro=mq['F_pro'], pop=mq['pop_mean']), numpy=dict(F_pro=mn['F_pro'], pop=mn['pop_mean']), seconds=time.time()-t0)
    json.dump(out, open(rpath('part2_validation.json'), 'w'), indent=1); print('validate', out)

if __name__ == '__main__':
    what = sys.argv[1:] or ['coherence', 'perinput', 'flux', 'mc']
    t0 = time.time()
    if 'perinput' in what: per_input()
    if 'coherence' in what: coherence_sweep()
    if 'flux' in what: tolerance()
    if 'mc' in what: montecarlo()
    if 'validate' in what: validate()
    print('done [%.0fs]' % (time.time()-t0))
