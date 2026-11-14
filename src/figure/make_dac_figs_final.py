"""
make_dac_figs_final.py — space-saving composite for the final 6+1-page DAC version (run after Parts 3-4)
  figures/dac_toffoli.png : (a) noisy Toffoli vs T1, qutrit vs IBM devices; (b) success per basis input  (column width)
Same data and styling as make_dac_figs.py (dac_toffoli_T1.png + dac_toffoli_inputs.png merged into one figure).
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

from paths import rpath, fpath, RESULTS as R, FIGURES as F
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"; C3 = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.2, "legend.fontsize": 6, "axes.titlesize": 7.5})
def clean(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

W = json.load(open(rpath('part4_T1_sweep.json'))); I = json.load(open(rpath('part4_ibm_noisy.json')))
t3 = json.load(open(rpath('part3_toffoli_ideal.json'))); s4 = json.load(open(rpath('part4_qutrit_scenarios.json')))['scenarios']
fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.45, 3.05), gridspec_kw=dict(height_ratios=[1.15, 1]))
# (a) fidelity vs T1
T1 = sorted(map(float, W))
ax.plot(T1, [W['%g' % t]['succ_mean'] for t in T1], '--', color=C3[2], lw=1.1, label='qutrit, basis')
ax.plot(T1, [W['%g' % t]['F_random_mean'] for t in T1], '-', color=C3[2], lw=1.4, label='qutrit, superposition')
for n, r in I.items():
    if n == 'FakeBrisbane': continue
    ax.scatter([r['T1_median_us']], [r['succ_mean']], s=13, facecolors='none', edgecolors=C3[0], lw=0.9)
    ax.scatter([r['T1_median_us']], [r['F_random_mean']], s=13, color=C3[0])
    off = {'FakeKingston': ((2, 5), 'right', 'succ_mean'), 'FakeFez': ((-5, -2), 'right', 'succ_mean'),
           'FakeTorino': ((6, -2), 'left', 'F_random_mean')}.get(n)
    if off: ax.annotate('Kingston, Nighthawk' if n == 'FakeKingston' else n.replace('Fake', ''), (r['T1_median_us'], r[off[2]]),
                        textcoords='offset points', xytext=off[0], fontsize=5.6, color=INK2, ha=off[1])
ax.scatter([], [], s=13, facecolors='none', edgecolors=C3[0], label='IBM, basis'); ax.scatter([], [], s=13, color=C3[0], label='IBM, superposition')
ax.axvline(27, color=INK2, ls=':', lw=0.7); ax.text(28, 0.989, 'reference device', fontsize=5.8, color=INK2)
ax.set_xscale('log'); ax.set_xlabel(r'$T_1$ ($\mu$s)  [qutrit $T_1(1\!\to\!0)$; IBM median]', labelpad=1); ax.set_ylabel('Toffoli fidelity')
ax.set_ylim(0.93, 0.995); ax.set_xlim(8, 400); clean(ax)
ax.legend(frameon=False, loc='lower right', ncol=2, handlelength=1.3, columnspacing=0.8, fontsize=5.6)
ax.set_title('(a) noisy Toffoli vs. $T_1$', loc='left')
# (b) per input
ins = list(t3['ours']['succ'])
series = [('qutrit, closed', [s4['ideal (closed system)']['succ'][k] for k in ins], '#9c9a94'),
          ('qutrit, $T_1$=27/17 $\\mu$s', [s4['measured T1 27/17 us, Tphi 40 us']['succ'][k] for k in ins], C3[2]),
          ('qutrit, IBM-like $T_1$', [s4['IBM-like coherence 200/100/200 us']['succ'][k] for k in ins], C3[1]),
          ('Kingston', I['FakeKingston']['succ'], C3[0]), ('Fez', I['FakeFez']['succ'], '#8a6bd1')]
w = 0.16
for j, (lab, v, col) in enumerate(series): bx.bar(np.arange(8)+(j-2)*w, v, w, color=col, label=lab)
bx.set_xticks(range(8)); bx.set_xticklabels(['|%s>' % k for k in ins], fontsize=6); bx.set_ylim(0.95, 1.0); bx.set_ylabel('success'); clean(bx)
bx.set_title('(b) success per basis input', loc='left')
bx.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=5, fontsize=5.4, handlelength=0.9, columnspacing=0.6, handletextpad=0.3)
fig.tight_layout(h_pad=0.4); fig.savefig(fpath('dac_toffoli.png'), dpi=400); plt.close(fig)
print('ok')
