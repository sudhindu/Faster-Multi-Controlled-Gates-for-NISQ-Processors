# results/

All numbers behind the paper, one folder per part (written by `src/experiments/pN_*.py`), plus figures.

| folder | content | produced by |
|---|---|---|
| `part0_design_history/` | design-stage data quoted in the paper (population-only pulses, duration scan, Toffoli one-at-a-time sensitivity) — see its README | earlier prototype (data only) |
| `part1_ideal_gates/` | gate metrics (4 and 5 levels), per-input time traces (`part1_per_input*/`), logical blocks, branch separation | `p1_ideal_gates.py`, `p1_branch_separation.py` |
| `part2_noisy_gates/` | $T_1$ sweep, per-input noisy populations, flux/coupler tolerance, Monte-Carlo, QuTiP validation | `p2_noisy_gates.py` |
| `part3_ideal_toffoli/` | ideal Toffoli, IBM transpilation, other qutrit schemes | `p3_ideal_toffoli.py` |
| `part4_noisy_toffoli/` | noisy Toffoli scenarios, $T_1$ sweep, IBM Aer results | `p4_noisy_toffoli.py` |
| `part5_cknot/` | $C^k$-NOT counts (k = 2–10), IBM schedules, noisy qutrit and IBM results | `p5_cknot.py` |
| `part6_multicontrol/` | $C^k$-NOT / $C^k$-$R_y$: exactness, costs, noise, θ scan, error budget, summary | `p6_multicontrol.py` |
| `part7_benchmarks/` | Grover, adders, 3-SAT, 26-circuit suite | `p7_benchmarks.py` |
| `figures/paper/` | the figures of the paper (`dac_*.png`) | `src/figures/*`, `p6`, `p7` |
| `figures/supplementary/` | additional figures (`p1_*` … `p5_*`) | `p1` … `p5` |
| `RESULTS_TABLES.md` | human-readable summary tables | `src/figures/make_tables.py` |

Units: time in ns, $T_1$ in µs, frequency in GHz/MHz. "basis" = mean success over computational inputs; "superposition"
/ `F_random_mean` = mean fidelity over Haar-random input states; `F_pro` = process fidelity (see appendix §B.5).
