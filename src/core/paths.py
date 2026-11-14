"""paths.py - every file location used by the scripts, relative to the repository root.

Nothing is written outside the repository unless you ask for it:
  * QUTRIT_ROOT         override the repository root (default: two folders above this file)
  * QUTRIT_RESULTS_DIR  write/read results somewhere else (default: <root>/results)

Layout (created on demand):
  results/part1_ideal_gates/ ... results/part7_benchmarks/   numbers (json, csv, npz)
  results/figures/paper/                                      figures used in the paper (dac_*.png)
  results/figures/supplementary/                              all other figures (p1_* ... p5_*)
  data/benchmarks_amm/                                        input benchmark circuits
"""
import os, re

ROOT = os.path.abspath(os.environ.get('QUTRIT_ROOT') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RESULTS = os.path.abspath(os.environ.get('QUTRIT_RESULTS_DIR') or os.path.join(ROOT, 'results'))
DATA = os.path.join(ROOT, 'data')
FIGURES = os.path.join(RESULTS, 'figures')

PART_DIRS = {'part0': 'part0_design_history', 'part1': 'part1_ideal_gates', 'part2': 'part2_noisy_gates', 'part3': 'part3_ideal_toffoli',
             'part4': 'part4_noisy_toffoli', 'part5': 'part5_cknot', 'part6': 'part6_multicontrol',
             'part7': 'part7_benchmarks'}

def rpath(*parts):
    """path of a result file; 'partN_...' files go to results/partN_<name>/, others to results/"""
    m = re.match(r'(part\d)', parts[0])
    folder = os.path.join(RESULTS, PART_DIRS[m.group(1)]) if m else RESULTS
    full = os.path.join(folder, *parts)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full

def fpath(name):
    """path of a figure; dac_* (paper) -> results/figures/paper/, everything else -> results/figures/supplementary/"""
    folder = os.path.join(FIGURES, 'paper' if name.startswith('dac_') else 'supplementary')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)

def nproc(default=2):
    """number of worker processes for the multiprocessing parts (env QUTRIT_NPROC)"""
    return int(os.environ.get('QUTRIT_NPROC', default))
