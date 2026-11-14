# part0_design_history

Design-stage numbers quoted in the paper (Section II-A, "Ideal results"):

* `population_only` — the same gates optimized for **populations only** (mean probability of the correct output over the
  9 basis inputs). Populations reach 0.989–0.995, but the process fidelity F_pro is only 0.47–0.76: the phases are wrong.
  This is the evidence for "Optimizing populations alone gave only F_pro = 0.47–0.76".
* `phase_aware_duration_scan` — phase-aware optimization at several durations (lab frame, 4 levels). It shows, e.g.,
  C2X12 at 100 ns reaching F_pro 0.970, versus 0.982 at the chosen 170 ns.

* `toffoli_one_at_a_time_sensitivity` — Toffoli F_pro at the reference T1 with one control error at a time. A static
  0.3 MHz residual coupling gives F_pro 0.954 against 0.965 with decoherence only, i.e. about 1 % lower. This is the
  evidence for the sentence on idle ZZ in Section II-A.

Provenance: these values come from the design stage, which used an earlier prototype of the same model. That prototype
is not part of the final pipeline, so the values are kept as data only. The chosen pulses (60/60/170 ns) are fully
reproduced by `src/experiments/p1_ideal_gates.py`. Any duration can be re-optimized with `src/core/optimize_gate.py`.
