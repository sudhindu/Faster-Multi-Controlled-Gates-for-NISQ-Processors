# data/

`benchmarks_amm/` — 26 benchmark circuits in Quipper ASCII format, used by Part 7 (Section III of the paper).

* `<name>_before`        input circuit of the Amy–Maslov–Mosca arithmetic and Toffoli suite
* `<name>_after_heavy`   the same circuit after the "heavy" optimization of Nam et al. (strongest published
                         qubit circuits for these inputs; used as the "best qubit" baseline)

The files were copied unchanged from the PyZX repository (Apache-2.0), folder
`circuits/Arithmetic_and_Toffoli`. `%5E` in a file name stands for `^` (e.g. `gf2%5E16_mult` = `gf2^16_mult`).
Please cite the original authors (see `../NOTICE`) when you use them.
