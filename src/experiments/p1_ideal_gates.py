"""
Part 1 — Ideal (closed-system) simulation of C^1X_12, C^2X_01, C^2X_12 in the FULL LAB FRAME
==========================================================================================

What it does
  1. builds the time-dependent lab-frame Hamiltonian of common.py (no rotating-wave approximation:
     the counter-rotating coupling terms and the real microwave carrier are kept);
  2. integrates the Schrödinger equation with QuTiP `sesolve` for each of the 9 logical inputs |c,t>;
  3. writes, per gate and per input, a CSV with the populations of all 9 logical states and of the
     leakage subspace versus time (results/part1_per_input/<gate>_in_<c><t>.csv);
  4. computes, with free virtual-Z phases (frame updates, zero duration):
       - population fidelity per input and mean (basis / "classical" inputs),
       - leakage per input,
       - process fidelity F_pro (requires correct phases -> valid for superposition inputs),
       - average gate fidelity over Haar-random superposition states (closed formula) and the
         sampled distribution of fidelities over 500 random 9-dim superposition states,
       - fidelities for named superposition inputs;
  5. plots the control pulses (coupler g(t), drive I/Q, target flux detuning, lab-frame drive) and the
     population dynamics of the 9 inputs.

Run:   python3 p1_ideal_gates.py            (≈ 1 min)       optional: SIM_N=5 python3 p1_ideal_gates.py
Outputs: results/part1_metrics.json, results/part1_summary.csv, results/part1_logical_blocks.npz,
         results/part1_per_input/*.csv, figures/p1_pulses_<gate>.png, figures/p1_dynamics_<gate>.png
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import os, json, csv, time
import numpy as np
import qutip as qt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

SFX = '' if N == 4 else '_N%d' % N          # 5-level check writes to separate files
from paths import rpath, fpath, RESULTS as OUT, FIGURES as FIG, DATA, nproc
PER = os.path.dirname(rpath('part1_per_input' + SFX, 'x'))
for d in (OUT, FIG, PER): os.makedirs(d, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#8a6bd1"]
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
NAME = {'C1X12': r'$C^1X_{12}$', 'C2X01': r'$C^2X_{01}$', 'C2X12': r'$C^2X_{12}$'}

def lab_hamiltonian(gate, T, x, dt=0.01):
    """QuTiP QobjEvo of the lab-frame Hamiltonian, coefficients sampled on a dt grid"""
    pu = pulses(T, x, dt); t = pu['t']; wd = drive_freq(gate, x)
    X = Q(AT + AT.conj().T)
    drive = pu['fI']*np.cos(TWO_PI*wd*t) - pu['fQ']*np.sin(TWO_PI*wd*t)   # rad/ns, multiplies (a_t + a_t†)
    H = qt.QobjEvo([Q(h0_lab(P0)), [Q((AC+AC.conj().T)@(AT+AT.conj().T)), pu['fg']], [X, drive], [Q(NT), pu['fD']]], tlist=t)
    return H, pu, wd

def run_gate(gate, T, x, npts=241):
    H, pu, wd = lab_hamiltonian(gate, T, x)
    E0 = np.real(np.diag(h0_lab(P0)))
    tl = np.linspace(0, T, npts)
    opts = dict(nsteps=400000, atol=1e-9, rtol=1e-8, max_step=0.02)
    Wcols = np.zeros((D, 9), complex); traces = {}
    for col, (c, tq) in enumerate(KEYS9):
        psi0 = qt.basis(D, idx(c, tq)); psi0.dims = [[N, N], [1, 1]]
        r = qt.sesolve(H, psi0, tl, options=opts)
        amps = np.array([s.full().ravel() for s in r.states])            # (npts, D)
        pops = abs(amps)**2
        traces[(c, tq)] = pops
        Wcols[:, col] = np.exp(1j*E0*T)*amps[-1]                          # interaction picture of idle H
    return Wcols, traces, tl, pu, wd

def superposition_tests(L, gate, ph):
    s2 = 1/np.sqrt(2); s3 = 1/np.sqrt(3)
    tests = {
        'control |+>01, target |+>01': ket9([s2, s2, 0], [s2, s2, 0]),
        'control |+>12, target |+>12': ket9([0, s2, s2], [0, s2, s2]),
        'control (|0>+|1>+|2>)/√3, target (|0>+|1>+|2>)/√3': ket9([s3, s3, s3], [s3, s3, s3]),
        'control |+>01, target (|1>+i|2>)/√2': ket9([s2, s2, 0], [0, s2, 1j*s2]),
        'control (|1>+|2>)/√2, target (|0>+i|1>)/√2': ket9([0, s2, s2], [s2, 1j*s2, 0]),
    }
    out = {k: state_fidelity(L, gate, ph, v) for k, v in tests.items()}
    rng = np.random.RandomState(11); fs = []
    for _ in range(500):
        z = rng.randn(9)+1j*rng.randn(9); fs.append(state_fidelity(L, gate, ph, z))
    return out, dict(mean=float(np.mean(fs)), std=float(np.std(fs)), min=float(np.min(fs)), max=float(np.max(fs)))

def plot_pulses(gate, T, pu, wd):
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 7.4), sharex=True)
    t = pu['t']
    ax[0].plot(t, pu['fg']/TWO_PI*1e3, color=COL[0]); ax[0].set_ylabel("coupler g(t)\n(MHz)")
    ax[1].plot(t, pu['fI']/TWO_PI*1e3, color=COL[1], label="I (in-phase)")
    ax[1].plot(t, pu['fQ']/TWO_PI*1e3, color=COL[3], label="Q (quadrature)"); ax[1].set_ylabel("drive envelope\n(MHz)")
    ax[1].legend(frameon=False, fontsize=8, loc='upper right')
    ax[2].plot(t, pu['fD']/TWO_PI*1e3, color=COL[2]); ax[2].set_ylabel("target flux\ndetuning (MHz)")
    lab = pu['fI']*np.cos(TWO_PI*wd*t) - pu['fQ']*np.sin(TWO_PI*wd*t)
    ax[3].plot(t, lab/TWO_PI*1e3, color=INK2, lw=0.4); ax[3].set_ylabel("lab-frame drive\n(MHz)"); ax[3].set_xlabel("time (ns)")
    for a in ax:
        a.grid(color=GRID, lw=0.7); a.set_axisbelow(True)
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle("%s control pulses, %d ns (drive carrier %.4f GHz)" % (NAME[gate], T, wd), x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(fpath('p1_pulses_%s%s.png' % (gate, SFX)), dpi=200); plt.close(fig)

def plot_dynamics(gate, T, tl, traces):
    fig, axs = plt.subplots(3, 3, figsize=(9.6, 7.2), sharex=True, sharey=True)
    for (c, tq), ax in zip(KEYS9, axs.ravel()):
        pops = traces[(c, tq)]
        for tt in range(3):
            ax.plot(tl, pops[:, idx(c, tt)], color=COL[tt], lw=1.4, label="target |%d>" % tt)
        leak = 1 - sum(pops[:, idx(cc, tt)] for cc in range(3) for tt in range(3))
        ax.plot(tl, leak, color=INK2, lw=1, ls='--', label="leakage")
        ax.set_title("input |%d,%d>" % (c, tq), fontsize=9, loc='left')
        ax.grid(color=GRID, lw=0.7); ax.set_ylim(-0.03, 1.05)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    for a in axs[-1]: a.set_xlabel("time (ns)")
    for a in axs[:, 0]: a.set_ylabel("population (control fixed)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='lower center', ncol=4, fontsize=8.5)
    fig.suptitle("%s population dynamics, lab frame (no RWA), all 9 inputs |control,target>" % NAME[gate], x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(fpath('p1_dynamics_%s%s.png' % (gate, SFX)), dpi=200); plt.close(fig)

if __name__ == '__main__':
    P = load_params(); metrics = {}; blocks = {}; t0 = time.time()
    for gate, v in P.items():
        T, x = v['T_ns'], v['x']
        W, traces, tl, pu, wd = run_gate(gate, T, x)
        L = W[LOGICAL, :]
        ph, F = best_phases(L, gate, x[2+4*K:])
        per, leak = populations(L, gate)
        sup, rnd = superposition_tests(L, gate, ph)
        metrics[gate] = dict(duration_ns=T, drive_GHz=wd, g_on_MHz=x[0], N_levels=N,
                             pop_mean=float(np.mean(list(per.values()))), pop_min=float(min(per.values())),
                             leak_mean=float(np.mean(list(leak.values()))), F_pro=float(F),
                             F_avg=avg_gate_fidelity(L, gate, ph), virtual_Z=ph.tolist(),
                             per_input_population={"|%d,%d>" % k: per[k] for k in KEYS9},
                             per_input_leakage={"|%d,%d>" % k: leak[k] for k in KEYS9},
                             superposition_inputs=sup, random_states=rnd)
        blocks[gate] = L; blocks[gate+'_phases'] = ph
        for (c, tq) in KEYS9:
            with open(os.path.join(PER, '%s_in_%d%d.csv' % (gate, c, tq)), 'w', newline='') as f:
                w = csv.writer(f); pops = traces[(c, tq)]
                w.writerow(['t_ns'] + ['P|%d,%d>' % k for k in KEYS9] + ['P_leak'])
                for i, tt in enumerate(tl):
                    row = [pops[i, idx(*k)] for k in KEYS9]
                    w.writerow(['%.3f' % tt] + ['%.6f' % r for r in row] + ['%.6f' % (1-sum(row))])
        plot_pulses(gate, T, pu, wd); plot_dynamics(gate, T, tl, traces)
        m = metrics[gate]
        print("%s  T=%d ns  pop %.4f (min %.4f)  leak %.4f  F_pro %.4f  F_avg %.4f  random-state F %.4f±%.4f  [%.0fs]" % (
            gate, T, m['pop_mean'], m['pop_min'], m['leak_mean'], m['F_pro'], m['F_avg'], rnd['mean'], rnd['std'], time.time()-t0), flush=True)
    sfx = SFX
    json.dump(metrics, open(rpath('part1_metrics%s.json' % sfx), 'w'), indent=1)
    np.savez(rpath('part1_logical_blocks%s.npz' % sfx), **blocks)
    with open(rpath('part1_summary%s.csv' % sfx), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gate', 'duration_ns', 'input', 'population_ideal_output', 'leakage'])
        for g, m in metrics.items():
            for kk in m['per_input_population']:
                w.writerow([g, m['duration_ns'], kk, '%.5f' % m['per_input_population'][kk], '%.5f' % m['per_input_leakage'][kk]])
            w.writerow([g, m['duration_ns'], 'MEAN', '%.5f' % m['pop_mean'], '%.5f' % m['leak_mean']])
            w.writerow([g, m['duration_ns'], 'F_pro (superposition, process)', '%.5f' % m['F_pro'], ''])
            w.writerow([g, m['duration_ns'], 'F_avg (Haar superpositions)', '%.5f' % m['F_avg'], ''])
