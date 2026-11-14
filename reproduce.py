"""reproduce.py - one entry point to regenerate the results of the paper (works on Linux, macOS and Windows).

    python reproduce.py figures      # paper figures + tables + appendix tables from the shipped results (~30 s)
    python reproduce.py quick        # fast re-computation: gates, Toffoli cost, C^k counts, exactness checks,
                                     #   benchmarks; then figures (~5 min on a laptop)
    python reproduce.py full         # everything, including all noisy simulations (several hours on 2 cores;
                                     #   set QUTRIT_NPROC to use more cores)
    python reproduce.py part1 ... part7   # a single part (see the PARTS table below)

Results are written to ./results (override with QUTRIT_RESULTS_DIR). Logs go to ./logs.
"""
import os, sys, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, 'src', 'experiments'); FIG = os.path.join(ROOT, 'src', 'figures')
LOG = os.path.join(ROOT, 'logs'); os.makedirs(LOG, exist_ok=True)
PY = sys.executable

PARTS = {
    'part1': [('p1_ideal_gates.py', []), ('p1_ideal_gates.py', [], {'SIM_N': '5'}), ('p1_branch_separation.py', [])],
    'part2': [('p2_noisy_gates.py', ['perinput', 'coherence', 'flux', 'mc', 'validate'])],
    'part3': [('p3_ideal_toffoli.py', [])],
    'part4': [('p4_noisy_toffoli.py', ['scen', 'sweep', 'ibm', 'plot'])],
    'part5': [('p5_cknot.py', ['counts', 'ibmdur']),
              ('p5_cknot.py', ['qutrit', '2,3,4,5,6,7,8', 'ideal']),
              ('p5_cknot.py', ['qutrit', '2,3,4,5,6,7,8', 'measured']),
              ('p5_cknot.py', ['qutrit', '2,3,4,5,6,7,8', 'improved']),
              ('p5_cknot.py', ['qutrit', '2,3,4,5,6,7,8', 'ibmlike']),
              ('p5_cknot.py', ['ibmnoisy', '2,3,4,5', 'FakeKingston,FakeFez,FakeNighthawk']),
              ('p5_cknot.py', ['plot'])],
    'part6': [('p6_multicontrol.py', ['check', 'counts', 'ibmdur']),
              ('p6_multicontrol.py', ['qutrit', '2,3,4,5', 'ideal']),
              ('p6_multicontrol.py', ['qutrit', '2,3,4,5', 'measured']),
              ('p6_multicontrol.py', ['qutrit', '2,3,4,5', 'improved']),
              ('p6_multicontrol.py', ['qutrit', '2,3,4,5', 'ibmlike']),
              ('p6_multicontrol.py', ['theta']),
              ('p6_multicontrol.py', ['ibmnoisy', '2,3,4,5', 'FakeKingston,FakeFez,FakeNighthawk']),
              ('p6_multicontrol.py', ['budget']), ('p6_multicontrol.py', ['whatif60']), ('p6_multicontrol.py', ['plot'])],
    'part7': [('p7_benchmarks.py', ['grover', 'arith', 'sat', 'amm']), ('p7_benchmarks.py', ['projection', 'report'])],
}
QUICK = [('p1_ideal_gates.py', []), ('p1_branch_separation.py', []), ('p3_ideal_toffoli.py', []), ('p5_cknot.py', ['counts']),
         ('p6_multicontrol.py', ['check', 'counts']), ('p7_benchmarks.py', ['grover', 'arith', 'sat', 'amm']),
         ('p7_benchmarks.py', ['projection', 'report'])]
FIGURES = [(os.path.join(FIG, 'make_dac_figs.py'), []), (os.path.join(FIG, 'make_dac_figs_final.py'), []),
           (os.path.join(EXP, 'p6_multicontrol.py'), ['plot']), (os.path.join(FIG, 'make_tables.py'), []),
           (os.path.join(FIG, 'make_appendix_tables.py'), [])]

def run(script, args, env_extra=None):
    path = script if os.path.isabs(script) else os.path.join(EXP, script)
    env = dict(os.environ); env.update(env_extra or {})
    tag = os.path.basename(path).replace('.py', '') + ('_' + '_'.join(a.replace(',', '-') for a in args) if args else '') + \
          ('_N5' if (env_extra or {}).get('SIM_N') == '5' else '')
    t0 = time.time(); print('>>', os.path.basename(path), ' '.join(args), flush=True)
    with open(os.path.join(LOG, tag[:120] + '.log'), 'w') as f:
        r = subprocess.run([PY, path] + args, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    print('   %s in %.0f s' % ('ok' if r.returncode == 0 else 'FAILED (see logs/)', time.time() - t0), flush=True)
    if r.returncode != 0: sys.exit(r.returncode)

def main(what):
    if not what: print(__doc__); return
    for w in what:
        if w == 'figures':
            for s, a in FIGURES: run(s, a)
        elif w == 'quick':
            for s, a in QUICK: run(s, a)
            for s, a in FIGURES: run(s, a)
        elif w == 'full':
            for p in sorted(PARTS):
                for item in PARTS[p]: run(*item)
            for s, a in FIGURES: run(s, a)
        elif w in PARTS:
            for item in PARTS[w]: run(*item)
        else:
            sys.exit('unknown target: %s' % w)

if __name__ == '__main__':
    main(sys.argv[1:])
