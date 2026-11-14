"""
Part 6 — Multi-controlled unitaries: C^k-NOT (2k-1 pulses) and C^k-Ry(theta) (2k two-qutrit pulses)
====================================================================================================
Sections (command-line words):
  check            : exact verification of both decompositions with ideal gates (random superposition inputs)
  counts           : pulse counts, depth and pulse time of the qutrit circuits (k = 2..10) and IBM decompositions
                     without routing (no ancilla and k-2 clean ancillas)                      -> part6_counts.csv
  ibmdur           : IBM C^k-Ry transpiled on real coupling maps (k = 2..8), no-ancilla and ancilla versions
  qutrit K S       : noisy qutrit C^k-Ry for k in K, scenario S in {ideal, measured, improved, ibmlike}
  theta            : sensitivity of the noisy C^3-Ry result to theta
  ibmnoisy K B     : noisy IBM C^k-Ry (Aer superoperator, <= 6 active qubits)
  plot             : figures/dac_multictrl.png and CSV tables
C^k-NOT noisy data come from p5_cknot.py (same model). theta = 0.9 rad unless stated.
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import os, sys, json, time, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import HERE
import multictrl as mc
from noise import MEASURED, IMPROVED, IBM_LIKE

from paths import rpath, fpath, RESULTS as OUT, FIGURES as FIG, DATA, nproc
TH = 0.9
SCN = {'ideal': None, 'measured': MEASURED, 'improved': IMPROVED, 'ibmlike': IBM_LIKE}
HERON = ['FakeKingston', 'FakeNighthawk', 'FakeFez', 'FakeTorino']
def jl(n):
    f = rpath(n); return json.load(open(f)) if os.path.exists(f) else {}
def jd(o, n): json.dump(o, open(rpath(n), 'w'), indent=1)

def check():
    out = {}
    for k in range(2, 7):
        out[k] = dict(cnot=mc.check_ideal(k, 'cnot', 0.0), ry=max(mc.check_ideal(k, 'ry', th) for th in (0.3, TH, 2.5)))
        print('k=%d  max infidelity of ideal decomposition: C^kNOT %.1e  C^kRy %.1e' % (k, out[k]['cnot'], out[k]['ry']), flush=True)
    # the ancilla-assisted IBM baseline is also checked (ancillas start and end in |0>)
    from ibm_ry import mcry_ancilla_circuit
    from qiskit.quantum_info import Statevector
    for k in (3, 4):
        qc = mcry_ancilla_circuit(k, TH); n = qc.num_qubits; data = list(range(k)) + [n-1]
        rng = np.random.RandomState(1); err = 0
        for _ in range(4):
            z = rng.randn(2**(k+1))+1j*rng.randn(2**(k+1)); z /= np.linalg.norm(z)
            full = np.zeros(2**n, complex)
            for xi in range(2**(k+1)):
                idx = sum(((xi >> (k-i)) & 1) << data[i] for i in range(k+1)); full[idx] = z[xi]
            o = Statevector(full).evolve(qc).data
            want = mc.ideal_vec(z, k, 'ry', TH); ref = np.zeros(2**n, complex)
            for xi in range(2**(k+1)):
                idx = sum(((xi >> (k-i)) & 1) << data[i] for i in range(k+1)); ref[idx] = want[xi]
            err = max(err, 1-abs(np.vdot(ref, o))**2)
        out['ancilla_baseline_k%d' % k] = err; print('ancilla baseline k=%d infidelity %.1e' % (k, err))
    jd(out, 'part6_check.json')

def counts():
    from ibm_tools import mcx_circuit, mcx_ancilla_circuit, pulse_counts, all_to_all_counts
    from ibm_ry import mcry_circuit, mcry_ancilla_circuit, all_to_all
    rows = []
    for k in range(2, 11):
        qn, qr = mc.counts(k, 'cnot'), mc.counts(k, 'ry')
        a_n, a_na = all_to_all_counts(k), all_to_all_counts(k, ancilla=True)
        a_r, a_ra = all_to_all(mcry_circuit(k, TH)), all_to_all(mcry_ancilla_circuit(k, TH))
        rows.append(dict(k=k, qutrit_cnot_pulses=qn['total_pulses'], qutrit_cnot_time_ns=qn['T_total_ns'], qutrit_cnot_depth=qn['depth'],
                         qutrit_ry_2q=qr['n_two_qutrit'], qutrit_ry_1q=qr['n_single'], qutrit_ry_time_ns=qr['T_total_ns'], qutrit_ry_depth=qr['depth'],
                         ibm_a2a_cnot_pulses=a_n['total_pulses'], ibm_a2a_cnot_2q=a_n['n2q'], ibm_a2a_cnot_anc_pulses=a_na['total_pulses'], ibm_a2a_cnot_anc_2q=a_na['n2q'],
                         ibm_a2a_ry_pulses=a_r['total_pulses'], ibm_a2a_ry_2q=a_r['n2q'], ibm_a2a_ry_anc_pulses=a_ra['total_pulses'], ibm_a2a_ry_anc_2q=a_ra['n2q']))
        print(rows[-1], flush=True)
    with open(rpath('part6_counts.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    jd(rows, 'part6_counts.json')

def ibmdur():
    from ibm_ry import mcry_circuit, mcry_ancilla_circuit, best_transpile_qc
    from ibm_tools import pulse_counts
    out = jl('part6_ibm_ry_durations.json')
    for name in HERON:
        for k in range(2, 9):
            be, (d, s, t, act) = best_transpile_qc(mcry_circuit(k, TH), name, nseed=20)
            out.setdefault(name, {})[str(k)] = dict(duration_ns=d, active_qubits=len(act), **pulse_counts(t))
            print(name, k, out[name][str(k)], flush=True)
        jd(out, 'part6_ibm_ry_durations.json')
    for name in ('FakeKingston',):
        for k in range(3, 9):
            be, (d, s, t, act) = best_transpile_qc(mcry_ancilla_circuit(k, TH), name, nseed=20)
            out.setdefault(name + '_ancilla', {})[str(k)] = dict(duration_ns=d, active_qubits=len(act), **pulse_counts(t))
            print(name, 'ancilla', k, out[name + '_ancilla'][str(k)], flush=True)
        jd(out, 'part6_ibm_ry_durations.json')

def qutrit(ks, scen):
    out = jl('part6_qutrit_ry_noisy.json'); coh = SCN[scen]
    for k in ks:
        t0 = time.time(); phi = mc.fit_phases(k, 'ry', TH)
        r = mc.metrics(k, 'ry', TH, coh, phi, Q=8, nbasis=None if k <= 4 else 16)
        r['duration'] = mc.counts(k, 'ry'); out.setdefault(scen, {})[str(k)] = r
        print('C^%d-Ry %s: basis %.4f (min %.4f, %d inputs)  random %.4f  [%.0fs]' % (k, scen, r['basis_mean'], r['basis_min'], r['n_basis'], r['random_mean'], time.time()-t0), flush=True)
        jd(out, 'part6_qutrit_ry_noisy.json')

def theta_scan():
    out = {}
    for th in (0.3, 0.9, 1.6, 2.5, np.pi):
        phi = mc.fit_phases(3, 'ry', th); r = mc.metrics(3, 'ry', th, MEASURED, phi, Q=6)
        out['%.3f' % th] = r; print('theta %.2f: basis %.4f random %.4f' % (th, r['basis_mean'], r['random_mean']), flush=True)
    jd(out, 'part6_theta_scan.json')

def ibmnoisy(ks, names):
    from ibm_ry import noisy_eval_unitary
    out = jl('part6_ibm_ry_noisy.json')
    for name in names:
        for k in ks:
            r = noisy_eval_unitary(k, name, TH); out.setdefault(name, {})[str(k)] = r
            print(name, k, None if r is None else 'dur %.0f ns basis %.4f random %.4f' % (r['duration_ns'], r['basis_mean'], r['random_mean']), flush=True)
            jd(out, 'part6_ibm_ry_noisy.json')

def plot():
    SURF, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"font.size": 7.2, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                         "legend.fontsize": 6, "figure.facecolor": SURF, "axes.facecolor": SURF})
    G, B, O, V = "#1baf7a", "#2a78d6", "#eb6834", "#8a6bd1"
    def clean(ax):
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    C = {r['k']: r for r in jl('part6_counts.json')}; ks = sorted(C)
    Dn = jl('part5_ibm_durations.json'); Dr = jl('part6_ibm_ry_durations.json')
    Qn = jl('part5_qutrit_noisy.json'); Qr = jl('part6_qutrit_ry_noisy.json'); In = jl('part5_ibm_noisy.json'); Ir = jl('part6_ibm_ry_noisy.json')
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.05))
    ax = axs[0]
    ax.plot(ks, [C[k]['qutrit_cnot_pulses'] for k in ks], 'o-', ms=2.5, color=G, label=r'qutrit $C^k$NOT ($2k{-}1$)')
    ax.plot(ks, [C[k]['qutrit_ry_2q'] + C[k]['qutrit_ry_1q'] for k in ks], 's-', ms=2.5, color=O, label=r'qutrit $C^kR_y$ ($2k$+2)')
    kk = sorted(int(x) for x in Dn['FakeKingston']); ax.plot(kk, [Dn['FakeKingston'][str(k)]['total_pulses'] for k in kk], 'o--', ms=2.5, color=B, label='IBM $C^k$NOT (routed)')
    kk = sorted(int(x) for x in Dr['FakeKingston']); ax.plot(kk, [Dr['FakeKingston'][str(k)]['total_pulses'] for k in kk], 's--', ms=2.5, color=V, label='IBM $C^kR_y$ (routed)')
    kk = sorted(int(x) for x in Dr['FakeKingston_ancilla']); ax.plot(kk, [Dr['FakeKingston_ancilla'][str(k)]['total_pulses'] for k in kk], ':', color=V, label='IBM $C^kR_y$, $k{-}2$ ancillas (routed)')
    ax.set_yscale('log'); ax.set_xlabel('controls $k$'); ax.set_ylabel('physical pulses'); ax.set_title('(a)', loc='left'); clean(ax)
    ax = axs[1]
    ax.plot(ks, [C[k]['qutrit_cnot_time_ns']/1e3 for k in ks], 'o-', ms=2.5, color=G)
    ax.plot(ks, [C[k]['qutrit_ry_time_ns']/1e3 for k in ks], 's-', ms=2.5, color=O)
    kk = sorted(int(x) for x in Dn['FakeKingston']); ax.plot(kk, [Dn['FakeKingston'][str(k)]['duration_ns']/1e3 for k in kk], 'o--', ms=2.5, color=B)
    kk = sorted(int(x) for x in Dr['FakeKingston']); ax.plot(kk, [Dr['FakeKingston'][str(k)]['duration_ns']/1e3 for k in kk], 's--', ms=2.5, color=V)
    if 'FakeKingston_ancilla' in Dr:
        kk = sorted(int(x) for x in Dr['FakeKingston_ancilla']); ax.plot(kk, [Dr['FakeKingston_ancilla'][str(k)]['duration_ns']/1e3 for k in kk], ':', color=V)
    ax.set_yscale('log'); ax.set_xlabel('controls $k$'); ax.set_ylabel(r'pulse time ($\mu$s)'); ax.set_title('(b) IBM: Kingston', loc='left'); clean(ax)
    ax = axs[2]
    def series(D, scen, key):
        kk = sorted(int(x) for x in D.get(scen, {}) if key in D[scen][x]); return kk, [D[scen][str(k)][key] for k in kk]
    k1, v1 = series(Qn, 'measured', 'F_random_mean'); ax.plot(k1, v1, 'o-', ms=2.5, color=G)
    k1, v1 = series(Qr, 'measured', 'random_mean'); ax.plot(k1, v1, 's-', ms=2.5, color=O)
    k1, v1 = series(Qn, 'ibmlike', 'F_random_mean'); ax.plot(k1, v1, 'o:', ms=2.5, color=G, label=r'qutrit, IBM-like $T_1$ (dotted)')
    k1, v1 = series(Qr, 'ibmlike', 'random_mean'); ax.plot(k1, v1, 's:', ms=2.5, color=O)
    if 'FakeKingston' in In:
        kk = sorted(int(x) for x in In['FakeKingston'] if In['FakeKingston'][x]); ax.plot(kk, [In['FakeKingston'][str(k)]['F_random_mean'] for k in kk], 'o--', ms=2.5, color=B)
    if 'FakeKingston' in Ir:
        kk = sorted(int(x) for x in Ir['FakeKingston'] if Ir['FakeKingston'][x]); ax.plot(kk, [Ir['FakeKingston'][str(k)]['random_mean'] for k in kk], 's--', ms=2.5, color=V)
    ax.set_xlabel('controls $k$'); ax.set_ylabel('superposition fidelity'); ax.set_title(r'(c) solid: qutrit $T_1$=27/17 $\mu$s', loc='left'); ax.set_xticks(range(2, 6)); clean(ax)
    h, l = axs[0].get_legend_handles_labels(); h2, l2 = axs[2].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, frameon=False, loc='lower center', ncol=3, fontsize=6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.6); fig.savefig(fpath('dac_multictrl.png'), dpi=300); plt.close(fig)
    # summary CSV
    rows = []
    for k in range(2, 9):
        r = dict(k=k, qutrit_cnot_time_ns=C[k]['qutrit_cnot_time_ns'], qutrit_cnot_pulses=C[k]['qutrit_cnot_pulses'],
                 qutrit_ry_time_ns=C[k]['qutrit_ry_time_ns'], qutrit_ry_pulses=C[k]['qutrit_ry_2q'] + C[k]['qutrit_ry_1q'])
        for lab, D, kk in (('ibm_cnot', Dn, 'FakeKingston'), ('ibm_ry', Dr, 'FakeKingston'), ('ibm_ry_anc', Dr, 'FakeKingston_ancilla')):
            e = D.get(kk, {}).get(str(k)); r[lab + '_time_ns'] = e['duration_ns'] if e else ''; r[lab + '_pulses'] = e['total_pulses'] if e else ''
        for lab, D, scen, key in (('qutrit_cnot_measured', Qn, 'measured', 'succ_mean'), ('qutrit_cnot_measured_sup', Qn, 'measured', 'F_random_mean'),
                                  ('qutrit_ry_measured_basis', Qr, 'measured', 'basis_mean'), ('qutrit_ry_measured_sup', Qr, 'measured', 'random_mean'),
                                  ('qutrit_ry_ibmlike_sup', Qr, 'ibmlike', 'random_mean')):
            e = D.get(scen, {}).get(str(k)); r[lab] = e.get(key, '') if e else ''
        for lab, D, key in (('ibm_cnot_kingston_basis', In, 'succ_mean'), ('ibm_cnot_kingston_sup', In, 'F_random_mean'),
                            ('ibm_ry_kingston_basis', Ir, 'basis_mean'), ('ibm_ry_kingston_sup', Ir, 'random_mean')):
            e = D.get('FakeKingston', {}).get(str(k)); r[lab] = e[key] if e else ''
        rows.append(r)
    with open(rpath('part6_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('plot + csv ok')

if __name__ == '__main__':
    a = sys.argv[1:]; t0 = time.time()
    if 'check' in a: check()
    if 'counts' in a: counts()
    if 'ibmdur' in a: ibmdur()
    if 'qutrit' in a: i = a.index('qutrit'); qutrit([int(v) for v in a[i+1].split(',')], a[i+2])
    if 'theta' in a: theta_scan()
    if 'ibmnoisy' in a: i = a.index('ibmnoisy'); ibmnoisy([int(v) for v in a[i+1].split(',')], a[i+2].split(','))
    if 'plot' in a: plot()
    print('done [%.0fs]' % (time.time()-t0))

def budget():
    """error budget at measured coherence: real C2X12 vs perfect C2X12 (+ its decay); all gates perfect (+ decay)"""
    from noise import MEASURED
    out = {}
    for label, ideal in (('real gates', set()), ('perfect C2X12', {'C2X12'}), ('all gates perfect', {'C1X12', 'C2X12', 'C2X01'})):
        mc.IDEAL_GATES.clear(); mc.IDEAL_GATES.update(ideal); mc._CACHE.clear()
        for kind in ('cnot', 'ry'):
            for k in (3, 4, 5):
                phi = mc.fit_phases(k, kind, TH, Q=6)
                r = mc.metrics(k, kind, TH, MEASURED, phi, Q=6, nbasis=16)
                out.setdefault(label, {}).setdefault(kind, {})[str(k)] = r
                print('%-18s %-4s k=%d basis %.4f random %.4f' % (label, kind, k, r['basis_mean'], r['random_mean']), flush=True)
                jd(out, 'part6_error_budget.json')
    mc.IDEAL_GATES.clear(); mc._CACHE.clear()

if __name__ == '__main__' and 'budget' in sys.argv[1:]:
    budget()

def whatif60():
    """hypothetical: C2X12 as fast as the other gates (60 ns) and perfect (+ decay) — sets the improvement headroom"""
    from noise import MEASURED, IBM_LIKE
    out = jl('part6_error_budget.json'); T0 = mc.P['C2X12']['T_ns']
    mc.IDEAL_GATES.clear(); mc.IDEAL_GATES.add('C2X12'); mc.P['C2X12']['T_ns'] = 60; mc._CACHE.clear()
    for scen, coh in (('measured', MEASURED), ('ibmlike', IBM_LIKE)):
        for kind in ('cnot', 'ry'):
            for k in (3, 4, 5):
                phi = mc.fit_phases(k, kind, TH, Q=6); r = mc.metrics(k, kind, TH, coh, phi, Q=6, nbasis=16)
                r['T_total_ns'] = mc.counts(k, kind)['T_total_ns']
                out.setdefault('C2X12 at 60 ns, perfect (' + scen + ')', {}).setdefault(kind, {})[str(k)] = r
                print('60ns-C2X12 %s %-4s k=%d T=%.0f basis %.4f random %.4f' % (scen, kind, k, r['T_total_ns'], r['basis_mean'], r['random_mean']), flush=True)
    mc.P['C2X12']['T_ns'] = T0; mc.IDEAL_GATES.clear(); mc._CACHE.clear()
    jd(out, 'part6_error_budget.json')

if __name__ == '__main__' and 'whatif60' in sys.argv[1:]:
    whatif60()
