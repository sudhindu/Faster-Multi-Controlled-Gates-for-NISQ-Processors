"""make_appendix_tables.py - writes the LaTeX tables of the appendix (docs/appendix/generated/*.tex)
directly from the result files, so that every number in the appendix tables is traceable to results/.

Run after the experiments (or with the shipped results):   python src/figures/make_appendix_tables.py
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', _d) for _d in ('core', 'baselines')]
import os, json, csv
from paths import rpath, ROOT

OUT = os.path.join(ROOT, 'docs', 'appendix', 'generated'); os.makedirs(OUT, exist_ok=True)
NM = {'C1X12': r'$C^1X_{12}$', 'C2X01': r'$C^2X_{01}$', 'C2X12': r'$C^2X_{12}$'}
def j(n):
    f = rpath(n); return json.load(open(f)) if os.path.exists(f) else None
def c(n):
    return list(csv.DictReader(open(rpath(n))))
def w(name, lines):
    open(os.path.join(OUT, name), 'w').write('\n'.join(lines) + '\n'); print('wrote', name)
def tab(colspec, head, rows, size=r'\footnotesize'):
    L = [r'{%s' % size, r'\begin{tabular}{%s}' % colspec, r'\toprule', head + r' \\', r'\midrule']
    L += [r + r' \\' for r in rows]
    L += [r'\bottomrule', r'\end{tabular}}']
    return L
def ket(s):
    return r'$|%s\rangle$' % s.strip('|<>')
def f3(x): return '--' if x in (None, '') else '%.3f' % float(x)
def f4(x): return '--' if x in (None, '') else '%.4f' % float(x)
def fi(x): return '--' if x in (None, '') else '%d' % round(float(x))

# ---------------------------------------------------------------- Part 1
m1, m5 = j('part1_metrics.json'), j('part1_metrics_N5.json')
rows = []
for g, m in m1.items():
    rows.append('%s & %d & %.1f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f$\\pm$%.4f & %.4f' % (
        NM[g], m['duration_ns'], m['g_on_MHz'], m['pop_mean'], m['pop_min'], m['leak_mean'], m['F_pro'], m['F_avg'],
        m['random_states']['mean'], m['random_states']['std'], m5[g]['F_pro']))
w('tab_p1_gates.tex', tab('lrrrrrrrrr', r'gate & $T$ (ns) & $g_{\rm on}/2\pi$ (MHz) & mean pop. & worst pop. & leakage & $F_{\rm pro}$ & $F_{\rm avg}$ & random states & $F_{\rm pro}$ (5 lev.)', rows, r'\scriptsize'))
keys = list(next(iter(m1.values()))['per_input_population'])
rows = ['%s & ' % ket(k) + ' & '.join('%.4f' % m1[g]['per_input_population'][k] for g in m1) + ' & ' +
        ' & '.join('%.4f' % m1[g]['per_input_leakage'][k] for g in m1) for k in keys]
w('tab_p1_perinput.tex', tab('lrrrrrr', r'input $|c,t\rangle$ & ' + ' & '.join(NM[g] + ' pop.' for g in m1) + ' & ' + ' & '.join(NM[g] + ' leak.' for g in m1), rows, r'\scriptsize'))
keys = list(next(iter(m1.values()))['superposition_inputs'])
def sup_label(s):
    import re
    s = re.sub(r'\|\+>(01|12)', r'$|{+}\\rangle_{\1}$', s)
    s = s.replace('/√3', r'$/\sqrt{3}$').replace('/√2', r'$/\sqrt{2}$')
    s = re.sub(r'\|(\d)>', r'$|\1\\rangle$', s)
    return s.replace('$$', '')
rows = ['%s & ' % sup_label(k) + ' & '.join('%.4f' % m1[g]['superposition_inputs'][k] for g in m1) for k in keys]
w('tab_p1_superpos.tex', tab('p{8.2cm}rrr', 'input state (control, target) & ' + ' & '.join(NM[g] for g in m1), rows, r'\scriptsize'))

bs = j('part1_branch_separation.json')
rows = ['%s & %d & %s & %.1f & %s & %.1f & %.1f & %.2f' % (NM[g], v['active_control_level'], '$%s$' % v['transition'].replace('->', r'{\to}'), v['g_on_MHz'],
        ' / '.join('%.4f' % v['branch_freq_GHz'][str(cc)] for cc in range(3)), v['nearest_separation_MHz'], v['largest_separation_MHz'], v['dfreq_dg_MHz_per_MHz']) for g, v in bs.items()]
w('tab_p1_branch.tex', tab('lrrrrrrr', r'gate & $c_0$ & $j{\to}k$ & $g_{\rm on}/2\pi$ (MHz) & line for control $|0\rangle/|1\rangle/|2\rangle$ (GHz) & nearest (MHz) & largest (MHz) & $\partial f/\partial g$', rows, r'\scriptsize'))
H = json.load(open(rpath('part0_design_history.json')))
rows = ['%s & %d & population only & %.4f & %.3f' % (NM[e['gate']], e['T_ns'], e['lab_frame']['pop'], e['lab_frame']['Fpro']) for e in H['population_only']]
rows += ['%s & %d & phase-aware & %.4f & %.3f' % (NM[e['gate']], e['T_ns'], e['lab_frame']['pop'], e['lab_frame']['Fpro']) for e in H['phase_aware_duration_scan']]
w('tab_p0_history.tex', tab('lrlrr', r'gate & $T$ (ns) & objective & mean population & $F_{\rm pro}$', rows))
S = H['toffoli_one_at_a_time_sensitivity']['results']
w('tab_p0_sens.tex', tab('lrr', r'control error switched on (Toffoli, reference $T_1$) & $F_{\rm pro}$ & basis success',
  ['%s & %.4f & %.4f' % (k.replace('%', r'\%'), v['Fpro'], v['succ']) for k, v in S.items()]))

# ---------------------------------------------------------------- Part 2
c2, pi2, tol, mc, val = j('part2_coherence.json'), j('part2_per_input.json'), j('part2_tolerance.json'), j('part2_montecarlo.json'), j('part2_validation.json')
rows = []
for g in c2:
    rows.append('%s & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f' % (NM[g], pi2[g]['ideal']['F_pro'], c2[g]['10']['F_pro'], c2[g]['20']['F_pro'],
                c2[g]['27']['F_pro'], c2[g]['27_same_T1_for_2']['F_pro'], c2[g]['50']['F_pro'], c2[g]['100']['F_pro'], c2[g]['200']['F_pro']))
w('tab_p2_coherence.tex', tab('lrrrrrrrr', r'gate & closed & $T_1{=}10$ & 20 & \textbf{27} (ref.) & 27, $|2\rangle$ as $|1\rangle$ & 50 & 100 & 200\,$\mu$s', rows, r'\scriptsize'))
rows = []
for g in tol:
    f = lambda k, v: '%.3f / %.3f' % (tol[g][k]['fixed'][v], tol[g][k]['recal'][v])
    rows.append('%s & %s & %s & %s & %s' % (NM[g], f('dft', '0.3000'), f('dfc', '0.3000'), f('gsc', '0.0100'), f('gsc', '0.0200')))
w('tab_p2_tolerance.tex', tab('lrrrr', r'gate & target offset 0.3\,MHz & control offset 0.3\,MHz & coupler $+1\%$ & coupler $+2\%$', rows))
rows = []
for g in mc:
    for l, v in mc[g].items():
        lab = 'residual (calibrated)' if l.startswith('residual') else 'stress (uncalibrated)'
        rows.append('%s & %s & %.4f$\\pm$%.4f & %.4f' % (NM[g], lab, v['F_pro'], v['F_pro_std'], v['pop_mean']))
w('tab_p2_mc.tex', tab('llrr', r'gate & error level & $F_{\rm pro}$ (mean$\pm$sd) & mean population', rows))
w('val_p2.tex', ['%.6f' % val['numpy']['F_pro'] + ' (numpy propagator) and %.6f (QuTiP \\texttt{propagator})' % val['qutip']['F_pro']])

# ---------------------------------------------------------------- Part 3
t3 = j('part3_toffoli_ideal.json'); o = t3['ours']
w('tab_p3_inputs.tex', tab('l' + 'r'*8, 'input & ' + ' & '.join(ket(k) for k in o['succ']), ['success & ' + ' & '.join('%.4f' % v for v in o['succ'].values())], r'\scriptsize'))
rows = [r'\textbf{this work} (3 native gates) & %d & 3 & 0 & 3 & simulation (Sec.~\ref{app:toffoli})' % o['T_total_ns']]
for n, r in t3['ibm'].items():
    rows.append('IBM %s (routed) & %.0f & %d & %d & %d & Qiskit 2.6.2, fake back-end' % (n.replace('Fake', ''), r['duration_ns'], r['n2q'], r['n1q_pulses'], r['total_pulses']))
a = t3['ibm_all_to_all']; rows.append('IBM decomposition, all-to-all & -- & %d & %d & %d & Qiskit 2.6.2' % (a['n2q'], a['n1q_pulses'], a['total_pulses']))
for n, r in t3['others'].items():
    rows.append('%s & %s & %d & %d & %d & %s' % (n, '%.0f' % r['T_total_ns'] if r['T_total_ns'] else '--', r['n_two_body'], r['n_single'], r['total_pulses'], r['source'].replace('x 60', r'$\times$60').replace('x 25', r'$\times$25')))
w('tab_p3_compare.tex', tab('p{4.4cm}rrrrp{4.6cm}', r'implementation & time (ns) & 2-body & 1-body & total & source', rows, r'\scriptsize'))

# ---------------------------------------------------------------- Part 4
s4, i4 = j('part4_qutrit_scenarios.json'), j('part4_ibm_noisy.json')
rows = []
for n, v in s4['scenarios'].items():
    rows.append('qutrit: %s & %d & %.4f & %.4f & %.4f & %.4f$\\pm$%.4f' % (n.replace('Tphi', r'$T_\phi$').replace('T1', r'$T_1$').replace('us', r'$\mu$s').replace('%', r'\%').replace('|2>', r'$|2\rangle$').replace('|1>', r'$|1\rangle$'),
                s4['duration']['T_total_ns'], v['succ_mean'], min(v['succ'].values()), v['F_random_mean'], v['F_pro'], v['F_pro_std']))
for n, r in i4.items():
    rows.append(r'IBM %s ($T_1$ median %.0f\,$\mu$s) & %.0f & %.4f & %.4f & %.4f & --' % (n.replace('Fake', ''), r['T1_median_us'], r['duration_ns'], r['succ_mean'], r['succ_min'], r['F_random_mean']))
w('tab_p4_scenarios.tex', tab('p{6.6cm}rrrrr', r'case & time (ns) & basis mean & basis worst & superpos. & $F_{\rm pro}$', rows, r'\scriptsize'))
W = j('part4_T1_sweep.json')
rows = ['%g & %.4f & %.4f' % (float(t), W[t]['succ_mean'], W[t]['F_random_mean']) for t in sorted(W, key=float)]
w('tab_p4_sweep.tex', tab('rrr', r'$T_1(1{\to}0)$ ($\mu$s) & basis success & superposition fidelity', rows))

# ---------------------------------------------------------------- Part 5 / 6
c5, d5, q5, i5 = j('part5_counts.json'), j('part5_ibm_durations.json'), j('part5_qutrit_noisy.json'), j('part5_ibm_noisy.json')
rows = []
for k in sorted(c5, key=int):
    x = c5[k]; kr = d5['FakeKingston'].get(k); ka = d5.get('FakeKingston_ancilla', {}).get(k)
    rows.append('%s & %d & %d & %d & %d & %s & %s & %d' % (k, x['qutrit']['total_pulses'], x['qutrit']['T_total_ns'], x['ibm_all_to_all']['total_pulses'],
                x['ibm_all_to_all_ancilla']['total_pulses'], '%d / %.0f' % (kr['total_pulses'], kr['duration_ns']) if kr else '--',
                '%d / %.0f' % (ka['total_pulses'], ka['duration_ns']) if ka else '--', x['gokhale_estimate']['total_pulses']))
w('tab_p5_counts.tex', tab('rrrrrrrr', r'$k$ & qutrit pulses & qutrit ns & IBM a2a & IBM a2a, anc. & Kingston routed & Kingston anc. routed & Gokhale est.', rows, r'\scriptsize'))
ks = sorted({k for s in q5.values() for k in s}, key=int)
lab = {'ideal': 'qutrit, closed system', 'measured': r'qutrit, reference $T_1$ 27/17\,$\mu$s', 'improved': r'qutrit, 100/60\,$\mu$s', 'ibmlike': r'qutrit, IBM-like 200/100\,$\mu$s'}
rows = []
for s in ('ideal', 'measured', 'improved', 'ibmlike'):
    rows.append(lab[s] + ' & ' + ' & '.join(('%.3f' % q5[s][k]['succ_mean'] + (' / %.3f' % q5[s][k]['F_random_mean'] if 'F_random_mean' in q5[s][k] else '')) if k in q5[s] else '--' for k in ks))
for n, d in i5.items():
    rows.append('IBM %s & ' % n.replace('Fake', '') + ' & '.join(('%.3f / %.3f' % (d[k]['succ_mean'], d[k]['F_random_mean'])) if d.get(k) else '--' for k in ks))
w('tab_p5_noisy.tex', tab('l' + 'r'*len(ks), 'case & ' + ' & '.join('$k{=}%s$' % k for k in ks), rows, r'\scriptsize'))
S6 = c('part6_summary.csv'); rows = []
for r in S6:
    rows.append('%s & %s / %s & %s / %s & %s / %s & %s / %s & %s / %s' % (r['k'], r['qutrit_cnot_pulses'], fi(r['qutrit_cnot_time_ns']), r['qutrit_ry_pulses'], fi(r['qutrit_ry_time_ns']),
                r['ibm_cnot_pulses'], fi(r['ibm_cnot_time_ns']), r['ibm_ry_pulses'], fi(r['ibm_ry_time_ns']), r['ibm_ry_anc_pulses'] or '--', fi(r['ibm_ry_anc_time_ns'])))
w('tab_p6_cost.tex', tab('rrrrrr', r'$k$ & qutrit $C^k$-NOT & qutrit $C^k$-$R_y$ & Kingston $C^k$-NOT & Kingston $C^k$-$R_y$ & Kingston $C^k$-$R_y$, anc.', rows, r'\scriptsize'))
rows = []
for r in S6:
    if not r['qutrit_ry_measured_sup']: continue
    rows.append('%s & %s / %s & %s / %s & %s & %s / %s & %s / %s' % (r['k'], f3(r['qutrit_cnot_measured']), f3(r['qutrit_cnot_measured_sup']), f3(r['qutrit_ry_measured_basis']), f3(r['qutrit_ry_measured_sup']),
                f3(r['qutrit_ry_ibmlike_sup']), f3(r['ibm_cnot_kingston_basis']), f3(r['ibm_cnot_kingston_sup']), f3(r['ibm_ry_kingston_basis']), f3(r['ibm_ry_kingston_sup'])))
w('tab_p6_noisy.tex', tab('rrrrrr', r'$k$ & qutrit NOT (ref.) & qutrit $R_y$ (ref.) & qutrit $R_y$ (IBM-like), sup. & Kingston NOT & Kingston $R_y$', rows, r'\scriptsize'))
EB = j('part6_error_budget.json'); rows = []
names = {'real gates': 'optimized gates (Table~\\ref{tab:a-gates})', 'perfect C2X12': r'perfect $C^2X_{12}$ (170\,ns)', 'all gates perfect': 'all three gates perfect',
         'C2X12 at 60 ns, perfect (measured)': r'perfect 60\,ns $C^2X_{12}$, reference $T_1$', 'C2X12 at 60 ns, perfect (ibmlike)': r'perfect 60\,ns $C^2X_{12}$, IBM-like $T_1$'}
for sc, v in EB.items():
    rows.append(names.get(sc, sc) + ' & ' + ' & '.join('%.3f / %.3f' % (v[g][k]['basis_mean'], v[g][k]['random_mean']) for g in ('cnot', 'ry') for k in ('3', '4', '5')))
w('tab_p6_budget.tex', tab('lrrrrrr', r'scenario & NOT $k{=}3$ & NOT 4 & NOT 5 & $R_y$ 3 & $R_y$ 4 & $R_y$ 5', rows, r'\scriptsize'))
TH = j('part6_theta_scan.json')
w('tab_p6_theta.tex', tab('rrr', r'$\theta$ & basis success & superposition fidelity', ['%s & %.4f & %.4f' % (t, v['basis_mean'], v['random_mean']) for t, v in TH.items()]))
CH = j('part6_check.json')
w('val_p6.tex', ['%.1e' % max(max(v.values()) for k, v in CH.items() if isinstance(v, dict))])

# ---------------------------------------------------------------- Part 7
G = c('part7_grover.csv'); rows = []
for r in G:
    rows.append('%s & %s & %s / %s & %s / %s & %s / %s (%s, %s anc.) & %s & %s / %s' % (r['n'], r['iterations'], r['qutrit_pulses_iter'], fi(r['qutrit_time_iter_ns']),
                r['qiskit_n2q_iter'], fi(r['qiskit_time_iter_ns']), r['best_n2q_iter'], fi(r['best_time_iter_ns']), r['best_method'].replace('_', r'\_'), r['best_ancillas'],
                r['routed_n2q_iter'] or '--', fi(r['qutrit_time_total']), fi(r['best_time_total'])))
w('tab_p7_grover.tex', tab('rrrrrrr', r'$n$ & iter. & qutrit (pulses / ns) & Qiskit no anc. & best synthesis & routed & full search ns: qutrit / best', rows, r'\scriptsize'))
A = c('part7_arith.csv')
rows = ['%s & %s & %s & %s & %s & %s / %s & %s / %s & %s' % (r['adder'], r['bits'], r['qubits'], r['toffoli'], r['cnot'], r['qutrit_pulses'], fi(r['qutrit_time_ns']), r['qiskit_n2q'], fi(r['qiskit_time_ns']), r['routed_n2q'] or '--') for r in A]
w('tab_p7_arith.tex', tab('lrrrrrrr', r'adder & bits & qubits & Toffoli & CNOT & qutrit (pulses / ns) & Qiskit (CZ / ns) & routed CZ', rows, r'\scriptsize'))
SAT = c('part7_sat.csv')
rows = ['%s & %s & %s & %s / %s & %s / %s & %s' % (r['variables'], r['clauses'], r['qubits'], r['qutrit_pulses'], fi(r['qutrit_time_ns']), r['qiskit_n2q'], fi(r['qiskit_time_ns']), r['routed_n2q'] or '--') for r in SAT]
w('tab_p7_sat.tex', tab('rrrrrr', r'variables & clauses & qubits & qutrit (pulses / ns) & Qiskit (CZ / ns) & routed CZ', rows))
AM = c('part7_amm.csv'); rows = []
for r in AM:
    nm = r['benchmark'].replace('_', r'\_').replace('^', r'\^{}')
    mct = ('%s / %s' % (r['mct_aware_pulses'], fi(r['mct_aware_time_ns']))) if r['mct_aware_pulses'] else '--'
    rows.append(r'\texttt{%s} & %s & %s & %s & %s / %s & %s & %s / %s & %s / %s & %s' % (nm, r['qubits'], r['toffoli'], r['cnot'], r['qutrit_pulses'], fi(r['qutrit_time_ns']), mct,
                r['nam_n2q'], fi(r['nam_time_ns']), r['qiskit_n2q'], fi(r['qiskit_time_ns']), r['routed_n2q'] or '--'))
w('tab_p7_amm.tex', tab('lrrrrrrrr', r'circuit & qubits & Toffoli & CNOT & qutrit (pulses / ns) & MCT-aware & Nam et al. (CZ / ns) & Qiskit (CZ / ns) & routed CZ', rows, r'\scriptsize'))
