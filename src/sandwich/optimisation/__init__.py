"""Week 7-8: ANN-MOGA optimisation (Sahib & Kovacs 2024, Sec. 4 and 5.2).

problem.py  genes -> ANN inputs -> objectives (Eqs. 20-21) and constraints (Eqs. 22-27)
metrics.py  front size, spread, violation rate, hypervolume, IMDSM knee point (Eq. 19)
run.py      reads moga.yaml, builds NSGA-II, runs it, saves the results
"""