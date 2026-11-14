# Artifact: Faster Multi-Controlled Gates for NISQ Processors — Phase-Aware Qutrit Pulse Design and Pulse-Level Compilation

Anonymous artifact for a DAC 2027 submission. It contains every script, input and result behind
Sections II–III of the paper. It also contains a detailed appendix (`docs/appendix/appendix.pdf`) that explains the
physics, the methods, every assumption and every number, written for readers who are not experimentalists.

The compiler tool of Section IV (`qtflow`) is released as a separate repository (link given in the paper);
it is **not** part of this artifact.

## What the paper claims and where the evidence is

| claim in the paper | evidence (this repository) | appendix |
|---|---|---|
| Three single-window, phase-aware two-qutrit gates $C^1X_{12}$, $C^2X_{01}$, $C^2X_{12}$ reach $F_{\rm pro}$ 0.994 / 0.997 / 0.982 in 60 / 60 / 170 ns (lab frame) | `results/part1_ideal_gates/` | §C–D |
| Gate fidelity under measured $T_1$ and control errors | `results/part2_noisy_gates/` | §E |
| Toffoli = 3 pulses, 200 ns; IBM devices need 23 pulses, ≈ 0.83 µs | `results/part3_ideal_toffoli/` | §F |
| Noisy Toffoli vs. IBM Kingston / Nighthawk / Fez / Torino / Brisbane | `results/part4_noisy_toffoli/` | §F |
| $C^k$-NOT with $2k-1$, $C^k$-$R_y$ with $2k$ pulses, exact, no ancilla; $T_{\rm NOT}=360k-520$ ns | `results/part5_cknot/`, `results/part6_multicontrol/` | §G |
| Noisy $C^k$ gates, error budget, angle independence, coherence tolerance | `results/part5_cknot/`, `results/part6_multicontrol/` | §G |
| Benchmarks: Grover, adders, Amy–Maslov–Mosca suite, 3-SAT oracles | `results/part7_benchmarks/` | §H |

`docs/appendix/PAPER_CROSSREF.md` maps every number of the paper to its file.

## Repository layout

```
.
├── README.md                this file
├── requirements.txt         pinned Python dependencies
├── reproduce.py             one entry point: figures | quick | full | part1..part7
├── LICENSE, NOTICE          Apache-2.0; third-party data and trademarks
├── src/
│   ├── core/                physics and simulators
│   │   ├── common.py            lab-frame Hamiltonian, pulse shapes, fidelity metrics
│   │   ├── noise.py             Lindblad model (level-dependent T1, dephasing), gate superoperators
│   │   ├── circuit.py           multi-qutrit circuit simulator (Toffoli, C^k-NOT)
│   │   ├── multictrl.py         C^k-NOT and C^k-Ry schedule-level simulator
│   │   ├── optimize_gate.py     the phase-aware pulse optimizer (how gate_params.json was made)
│   │   ├── gate_params.json     the optimized pulses
│   │   └── paths.py             all input/output locations (relative to the repository)
│   ├── baselines/           IBM and benchmark baselines (Qiskit / Aer)
│   │   ├── ibm_tools.py         transpilation on fake back-ends, pulse counting, Aer noise
│   │   ├── ibm_ry.py            C^k-Ry baselines (no ancilla, k-2 clean ancillas)
│   │   └── bench_tools.py       cost model, Quipper parser, Qiskit all-to-all / routed baselines
│   ├── experiments/         one script per part of the study (p1 ... p7)
│   └── figures/             paper figures, results tables, appendix tables
├── data/benchmarks_amm/     benchmark circuits (third-party, Apache-2.0, see data/README.md)
├── results/                 all numbers (json/csv/npz) per part + figures (paper/, supplementary/)
└── docs/appendix/           appendix.pdf, its LaTeX source, generated tables, paper cross-reference
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The versions of `qiskit`, `qiskit-aer` and `qiskit-ibm-runtime` are pinned. The IBM device data
(coupling maps, gate errors, $T_1$/$T_2$) come from the calibration snapshots inside `qiskit-ibm-runtime 0.51.0`,
and other versions ship different snapshots.

## Reproduce

All scripts write **inside this repository** (`results/`, `logs/`), never to an absolute path.
To write somewhere else, set `QUTRIT_RESULTS_DIR=/your/folder`.

```bash
python reproduce.py figures   # ~30 s : all paper figures and tables from the shipped results
python reproduce.py quick     # ~5 min: recompute gates, Toffoli, C^k counts, exactness checks, benchmarks
python reproduce.py full      # hours : everything, including all noisy density-matrix simulations
python reproduce.py part4     # one part only (part1 ... part7)
QUTRIT_NPROC=8 python reproduce.py part2   # more worker processes for the parallel parts
```

| part | script | what it computes | time (2 cores) |
|---|---|---|---|
| 1 | `p1_ideal_gates.py` | lab-frame gates, 9 inputs, $F_{\rm pro}$, superpositions, 5-level check | ~10 s each |
| 2 | `p2_noisy_gates.py` | Lindblad gates: $T_1$ sweep, per-input, flux/coupler tolerance, Monte-Carlo, QuTiP validation | ~1 h |
| 3 | `p3_ideal_toffoli.py` | ideal Toffoli, IBM transpilation, pulse/time comparison | ~10 s |
| 4 | `p4_noisy_toffoli.py` | noisy Toffoli scenarios, $T_1$ sweep, IBM Aer runs | ~30 min |
| 5 | `p5_cknot.py` | $C^k$-NOT counts, IBM schedules, noisy qutrit and IBM runs | ~1–2 h |
| 6 | `p6_multicontrol.py` | $C^k$-NOT / $C^k$-$R_y$ exactness, costs, noise, θ scan, error budget | ~1–2 h |
| 7 | `p7_benchmarks.py` | Grover, adders, 3-SAT, suite of 26 circuits | ~1 min |

The "quick" target was checked to reproduce the shipped files of parts 1, 3, 5 (counts), 6 (check, counts) and 7
bit for bit. The noisy parts use fixed random seeds. They should match to the printed precision; floating-point
differences between platforms can change the last digit.

## Conventions

* Time in ns, frequency in GHz (Hamiltonians in rad/ns). $T_1$ in µs.
* "Pulse" = one physical control window. Two-body pulse = one two-qutrit or two-qubit gate. $R_z$ is virtual (free)
  on both platforms.
* "Reference device" = the measured relaxation times $T_1(1\to0)=27$ µs, $T_1(2\to1)=17$ µs of a laboratory
  transmon; $T_\phi=40$ µs is assumed.
* IBM results are **simulations** with Qiskit fake back-ends (calibration snapshots), not hardware runs.

## License

Apache-2.0 (see `LICENSE`). Third-party benchmark circuits: see `NOTICE` and `data/README.md`.
