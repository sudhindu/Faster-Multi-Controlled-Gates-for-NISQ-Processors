"""p1_branch_separation.py - how strongly the coupler makes the target frequency depend on the control level.

For each gate C^{c0}X_{jk} and its optimized coupling g_on, computes the dressed target transition j->k for
control levels c = 0, 1, 2 (static Hamiltonian, coupler on, no drive). The difference between the active
branch c0 and the nearest other branch sets how short a selective drive can be. df/dg (MHz per MHz) is the sensitivity of the
active line to the coupling strength, i.e. why the coupler amplitude must be calibrated.
Writes results/part1_ideal_gates/part1_branch_separation.json.   Run: python p1_branch_separation.py  (~2 s)
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import json
import numpy as np
from common import GATES, P0, TWO_PI, dressed_order, h0_rot, load_params
from paths import rpath

P = load_params(); out = {}
for gate, (c0, j, k) in GATES.items():
    g_on = P[gate]['x'][0]/1e3                                    # MHz -> GHz
    E, o = dressed_order(h0_rot(P0, 0.0, g_on))
    f = {c: float((E[o[(c, k)]] - E[o[(c, j)]])/TWO_PI) for c in range(3)}
    sep = {c: abs(f[c0] - f[c])*1e3 for c in range(3) if c != c0}
    d = 1e-4                                                     # sensitivity of the active line to the coupling
    def fa(gg):
        E2, o2 = dressed_order(h0_rot(P0, 0.0, gg)); return (E2[o2[(c0, k)]] - E2[o2[(c0, j)]])/TWO_PI
    slope = float((fa(g_on + d) - fa(g_on - d))/(2*d))
    out[gate] = dict(g_on_MHz=g_on*1e3, active_control_level=c0, transition='%d->%d' % (j, k),
                     branch_freq_GHz=f, separation_MHz=sep, nearest_separation_MHz=min(sep.values()),
                     largest_separation_MHz=max(sep.values()), dfreq_dg_MHz_per_MHz=slope)
    print(gate, 'g_on %.1f MHz' % (g_on*1e3), {c: '%.4f GHz' % v for c, v in f.items()}, 'nearest %.1f MHz' % min(sep.values()), 'df/dg %.2f' % slope)
json.dump(out, open(rpath('part1_branch_separation.json'), 'w'), indent=1)
