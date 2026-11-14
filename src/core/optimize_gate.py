"""
optimize_gate.py — phase-aware pulse optimisation (how gate_params.json was produced)
====================================================================================

Objective: J = 1 - F_pro, with F_pro = |Tr(U_ideal^† D L)|^2 / 81 the process fidelity on the 9 logical states,
D the 4 virtual-Z phases (optimised together with the pulse). Because F_pro needs the correct PHASES as well as
the correct populations, the resulting gates are valid for superposition inputs (not only for basis inputs).
Controls: g_on, drive detuning, and 6-knot splines for drive I, drive Q, coupler modulation, target flux detuning
(see common.pulses). Fast rotating-frame propagation (dt = 0.25 ns), L-BFGS-B with box bounds, random restarts.
Verify any result in the lab frame with p1_ideal_gates.py.

Usage: python3 optimize_gate.py C2X12 170 [n_starts=2] [seed=0]   -> prints F_pro and writes opt_<gate>_<T>.json
"""
import sys, json
import numpy as np
from scipy.optimize import minimize
from common import *

def bounds():
    return [(20, 80), (-40, 40)] + [(-6, 6)]*(2*K) + [(-1, 1)]*(2*K) + [(-2*np.pi, 2*np.pi)]*4

def cost(x, gate, T, dt=0.25):
    L = unitary_rwa(gate, T, x[:2+4*K], dt=dt)[np.ix_(LOGICAL, LOGICAL)]
    return 1 - fpro(L, gate, x[2+4*K:])

def initial(gate, T, rng, g_on=50.0):
    c0, j, k = GATES[gate]; me = np.sqrt(2) if (j, k) == (1, 2) else 1.0
    A = 1000*0.5/(me*max(T-26.0, 1.0))*1.15                      # pi-pulse area guess (MHz)
    x = np.zeros(2+4*K+4); x[0] = g_on; x[1] = rng.uniform(-3, 3)
    x[2:2+K] = A/20*rng.uniform(0.8, 1.2); x[2+K:2+2*K] = 0.1*rng.randn(K)
    x[2+2*K:2+4*K] = 0.05*rng.randn(2*K); x[2+4*K:] = rng.uniform(-np.pi, np.pi, 4)
    return x

if __name__ == '__main__':
    gate, T = sys.argv[1], float(sys.argv[2]); ns = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    rng = np.random.RandomState(int(sys.argv[4]) if len(sys.argv) > 4 else 0); best = None
    for s in range(ns):
        r = minimize(cost, initial(gate, T, rng), args=(gate, T), method='L-BFGS-B', bounds=bounds(), options=dict(maxiter=300))
        print('start %d: F_pro %.4f' % (s, 1-r.fun), flush=True)
        if best is None or r.fun < best.fun: best = r
    L = unitary_rwa(gate, T, best.x[:2+4*K], dt=0.05)[np.ix_(LOGICAL, LOGICAL)]
    print('best F_pro (dt=0.05): %.4f' % fpro(L, gate, best.x[2+4*K:]))
    json.dump(dict(T_ns=T, x=best.x.tolist()), open('opt_%s_%d.json' % (gate, T), 'w'), indent=1)
