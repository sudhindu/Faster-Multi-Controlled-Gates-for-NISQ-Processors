"""
make_dac_figs.py — compact composite figures for the 6-page DAC paper (run after Parts 1-4)
  figures/dac_gates.png      : row 1 control waveforms, row 2 selectivity dynamics, for the 3 gates (full width)
  figures/dac_gate_noise.png : (a) F_pro vs T1, (b) F_pro vs coupler-strength error (column width)
  figures/dac_toffoli_T1.png : noisy Toffoli vs T1, qutrit vs IBM devices (column width)
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import os, json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

from paths import rpath, fpath, RESULTS as R, FIGURES as F
SURF, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
C3 = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
                     "legend.fontsize": 6.5, "axes.titlesize": 8})
NAME = {'C1X12': r'$C^1X_{12}$', 'C2X01': r'$C^2X_{01}$', 'C2X12': r'$C^2X_{12}$'}
def clean(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

P = load_params()
# ---------------------------------------------------------------- Fig: gates
fig, axs = plt.subplots(2, 3, figsize=(7.0, 2.55))
for col, g in enumerate(P):
    T, x = P[g]['T_ns'], P[g]['x']; pu = pulses(T, x, 0.1); t = pu['t']
    ax = axs[0, col]
    ax.plot(t, pu['fg']/TWO_PI*1e3, color=C3[0], label=r'coupler $g(t)$')
    ax.plot(t, pu['fI']/TWO_PI*1e3, color=C3[1], label=r'drive $\Omega_I(t)$')
    ax.plot(t, pu['fQ']/TWO_PI*1e3, color='#8a6bd1', label=r'drive $\Omega_Q(t)$')
    ax.plot(t, pu['fD']/TWO_PI*1e3, color=C3[2], label=r'target flux $\delta_t(t)$')
    ax.set_title('%s, %d ns' % (NAME[g], T), loc='left'); ax.set_xlim(0, T); clean(ax)
    if col == 0: ax.set_ylabel('MHz')
    c0, j, k = GATES[g]
    ax = axs[1, col]
    for c in range(3):
        rows = list(csv.reader(open(rpath('part1_per_input', '%s_in_%d%d.csv' % (g, c, j)))))
        hdr = rows[0]; data = np.array(rows[1:], float); col_k = hdr.index('P|%d,%d>' % (c, k))
        ax.plot(data[:, 0], data[:, col_k], color=['#3d3d3d', '#c9a227', '#d4477a'][c], lw=1.5 if c == c0 else 1.0, ls='-' if c == c0 else '--',
                label='control $|%d\\rangle$' % c)
    ax.set_xlim(0, T); ax.set_ylim(-0.03, 1.05); ax.set_xlabel('time (ns)'); clean(ax)
    ax.text(0.03, 0.55, r'$P(|c,%d\rangle)$, start $|c,%d\rangle$' % (k, j), transform=ax.transAxes, fontsize=6.5, color=INK2)
    if col == 0: ax.set_ylabel('population')
h, l = axs[0, 0].get_legend_handles_labels(); h2, l2 = axs[1, 0].get_legend_handles_labels()
fig.legend(h + h2, l + l2, frameon=False, loc='lower center', ncol=7, fontsize=6.3, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.8)
fig.savefig(fpath('dac_gates.png'), dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig: gate noise
coh = json.load(open(rpath('part2_coherence.json'))); tol = json.load(open(rpath('part2_tolerance.json')))
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75))
T1S = [10, 20, 27, 50, 100, 200, 300]
for i, g in enumerate(P):
    axs[0].plot(T1S, [coh[g]['%g' % t]['F_pro'] for t in T1S], 'o-', ms=2.5, color=C3[i], label=NAME[g])
    axs[0].plot([27], [coh[g]['27_same_T1_for_2']['F_pro']], 'x', ms=4, color=C3[i])
    d = tol[g]['gsc']
    for rc, ls in (('fixed', '--'), ('recal', '-')):
        xs = sorted(d[rc], key=float); axs[1].plot([100*float(s) for s in xs], [d[rc][s] for s in xs], ls, color=C3[i], lw=1.0)
axs[0].set_xscale('log'); axs[0].axvline(27, color=INK2, ls=':', lw=0.7)
axs[0].set_xlabel(r'$T_1(1\!\to\!0)$ ($\mu$s)'); axs[0].set_ylabel(r'$F_{\mathrm{pro}}$'); axs[0].set_ylim(0.92, 1.0)
axs[0].legend(frameon=False, loc='lower right', handlelength=1.2); axs[0].set_title('(a)', loc='left')
axs[1].set_xlabel('coupler error (%)'); axs[1].set_ylim(0.0, 1.02); axs[1].set_title('(b)', loc='left')
for a in axs: clean(a)
fig.tight_layout(w_pad=0.6); fig.savefig(fpath('dac_gate_noise.png'), dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig: Toffoli vs T1
W = json.load(open(rpath('part4_T1_sweep.json'))); I = json.load(open(rpath('part4_ibm_noisy.json')))
fig, ax = plt.subplots(figsize=(3.45, 2.25))
T1 = sorted(map(float, W))
ax.plot(T1, [W['%g' % t]['succ_mean'] for t in T1], '--', color=C3[2], lw=1.1, label='qutrit, basis inputs')
ax.plot(T1, [W['%g' % t]['F_random_mean'] for t in T1], '-', color=C3[2], lw=1.4, label='qutrit, superposition')
for n, r in I.items():
    if n == 'FakeBrisbane': continue
    ax.scatter([r['T1_median_us']], [r['succ_mean']], s=14, facecolors='none', edgecolors=C3[0], lw=0.9)
    ax.scatter([r['T1_median_us']], [r['F_random_mean']], s=14, color=C3[0])
    off = {'FakeKingston': ((2, 5), 'right', 'succ_mean'), 'FakeNighthawk': ((-6, -10), 'right', 'F_random_mean'),
           'FakeFez': ((-5, -2), 'right', 'succ_mean'), 'FakeTorino': ((6, -2), 'left', 'F_random_mean')}[n]
    if n == 'FakeNighthawk': continue
    ax.annotate('Kingston, Nighthawk' if n == 'FakeKingston' else n.replace('Fake', ''), (r['T1_median_us'], r[off[2]]), textcoords='offset points', xytext=off[0], fontsize=5.8, color=INK2, ha=off[1])
ax.scatter([], [], s=14, facecolors='none', edgecolors=C3[0], label='IBM, basis'); ax.scatter([], [], s=14, color=C3[0], label='IBM, superposition')
ax.axvline(27, color=INK2, ls=':', lw=0.7); ax.text(28, 0.991, 'reference device', fontsize=6, color=INK2)
ax.set_xscale('log'); ax.set_xlabel(r'$T_1$ ($\mu$s)  [qutrit $T_1(1\!\to\!0)$; IBM median]'); ax.set_ylabel('Toffoli fidelity')
ax.set_ylim(0.93, 0.995); ax.set_xlim(8, 400); clean(ax); fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=2, handlelength=1.4, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(fpath('dac_toffoli_T1.png'), dpi=300); plt.close(fig)
print('ok')

# ---------------------------------------------------------------- Fig: gate noise, full width (per-input, T1 sweep, flux, coupler)
pin = json.load(open(rpath('part2_per_input.json')))
fig = plt.figure(figsize=(7.0, 1.95)); gs = fig.add_gridspec(1, 4, width_ratios=[2.3, 1, 1, 1], wspace=0.45)
ax = fig.add_subplot(gs[0]); labs = ["|%d,%d>" % k for k in KEYS9]
for i, g in enumerate(P):
    xs = np.arange(9) + i*10
    ax.bar(xs, [pin[g]['measured']['per_input'][l] for l in labs], 0.75, color=C3[i], label=NAME[g])
    ax.scatter(xs, [pin[g]['ideal']['per_input'][l] for l in labs], marker='_', s=30, color=INK, lw=1.0, zorder=3)
ax.set_xticks([i*10 + j for i in range(3) for j in range(9)])
ax.set_xticklabels(['%d%d' % k for _ in range(3) for k in KEYS9], fontsize=4.8, rotation=90)
ax.set_ylim(0.95, 1.002); ax.set_ylabel('ideal-output population'); ax.set_title('(a) per input $|c,t\\rangle$; bars: $T_1$=27/17 $\\mu$s, ticks: closed', loc='left', fontsize=6.8)
import matplotlib.transforms as mtr
for i, g in enumerate(P): ax.text(i*10+4, -0.30, NAME[g], transform=mtr.blended_transform_factory(ax.transData, ax.transAxes), ha='center', fontsize=7, color=C3[i])
clean(ax)
ax = fig.add_subplot(gs[1])
for i, g in enumerate(P):
    ax.plot(T1S, [coh[g]['%g' % t]['F_pro'] for t in T1S], 'o-', ms=2.2, color=C3[i])
    ax.plot([27], [coh[g]['27_same_T1_for_2']['F_pro']], 'x', ms=4, color=INK)
ax.set_xscale('log'); ax.axvline(27, color=INK2, ls=':', lw=0.7); ax.set_ylim(0.92, 1.0)
ax.set_xlabel(r'$T_1(1\!\to\!0)$ ($\mu$s)'); ax.set_ylabel(r'$F_{\mathrm{pro}}$'); ax.set_title('(b) coherence', loc='left', fontsize=6.8); clean(ax)
for j, (key, xl, sc, ttl) in enumerate((('dft', 'target offset (MHz)', 1, '(c) flux offset'), ('gsc', 'coupler error (%)', 100, '(d) coupler'))):
    ax = fig.add_subplot(gs[2+j])
    for i, g in enumerate(P):
        d = tol[g][key]
        for rc, ls in (('fixed', '--'), ('recal', '-')):
            xs = sorted(d[rc], key=float); ax.plot([sc*float(s) for s in xs], [d[rc][s] for s in xs], ls, color=C3[i], lw=1.0)
    ax.set_xlabel(xl); ax.set_title(ttl, loc='left', fontsize=6.8); clean(ax)
    ax.set_ylim((0.8, 1.003) if key == 'dft' else (0.0, 1.02))
fig.text(0.995, 0.01, 'dashed: fixed calibration, solid: re-calibrated', ha='right', fontsize=5.8, color=INK2)
fig.subplots_adjust(left=0.075, right=0.995, bottom=0.30, top=0.88)
fig.savefig(fpath('dac_gate_noise_full.png'), dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig: Toffoli per input (ideal + noisy, qutrit vs IBM)
t3 = json.load(open(rpath('part3_toffoli_ideal.json'))); s4 = json.load(open(rpath('part4_qutrit_scenarios.json')))['scenarios']
ins = list(t3['ours']['succ'])
series = [('qutrit, closed system', [s4['ideal (closed system)']['succ'][k] for k in ins], '#9c9a94'),
          ('qutrit, $T_1$=27/17 $\\mu$s', [s4['measured T1 27/17 us, Tphi 40 us']['succ'][k] for k in ins], C3[2]),
          ('qutrit, IBM-like $T_1$', [s4['IBM-like coherence 200/100/200 us']['succ'][k] for k in ins], C3[1]),
          ('IBM Kingston', I['FakeKingston']['succ'], C3[0]), ('IBM Fez', I['FakeFez']['succ'], '#8a6bd1')]
fig, ax = plt.subplots(figsize=(3.45, 1.9)); w = 0.16
for j, (lab, v, col) in enumerate(series): ax.bar(np.arange(8)+(j-2)*w, v, w, color=col, label=lab)
ax.set_xticks(range(8)); ax.set_xticklabels(['|%s>' % k for k in ins], fontsize=6); ax.set_ylim(0.95, 1.0); ax.set_ylabel('success probability'); clean(ax)
fig.legend(frameon=False, loc='lower center', ncol=3, fontsize=5.6, handlelength=1, columnspacing=0.8)
fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig(fpath('dac_toffoli_inputs.png'), dpi=300); plt.close(fig)
print('ok2')
