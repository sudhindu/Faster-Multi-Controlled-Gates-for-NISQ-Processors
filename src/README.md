# src/

* `core/` — physics: `common.py` (Hamiltonian, pulses, metrics), `noise.py` (Lindblad), `circuit.py` and
  `multictrl.py` (circuit simulators), `optimize_gate.py`, `gate_params.json`, `paths.py` (all file locations).
* `baselines/` — Qiskit/Aer baselines and the benchmark cost model.
* `experiments/` — `p1_*` … `p7_*`, one script per part; run them from anywhere, e.g. `python src/experiments/p3_ideal_toffoli.py`.
* `figures/` — paper figures, `RESULTS_TABLES.md` and the appendix tables.

Every script finds `core/` and `baselines/` by itself and writes to `results/<part folder>/` and `results/figures/`
(see `core/paths.py`). File names in the script headers (e.g. `results/part1_metrics.json`) refer to these per-part
folders. Set `QUTRIT_RESULTS_DIR` to write elsewhere and `QUTRIT_NPROC` for more worker processes.
