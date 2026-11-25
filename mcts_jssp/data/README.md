# Data

## `orlib/instances/*.txt` and `orlib/bks.json`

The 82 classical job shop instances of the OR-Library used as the test set (Table 3.3 of the thesis):
`ft06, ft10, ft20, la01-la40, abz5-abz9, orb01-orb10, swv01-swv20, yn1-yn4`.

* **Source:** JSPLib, the benchmark collection of ScheduleOpt
  (<https://github.com/ScheduleOpt/benchmarks>, directory `jobshop/`), commit
  `8a6e36ef31263d36975e56fb481e0ba04e188d6f` of 22 October 2026.
* **Licence of the collection:** CC BY-SA 4.0.  The instances themselves come from the OR-Library
  (J. E. Beasley) and the publications named in the thesis.
* **Format of an instance:** the first line holds the numbers of jobs and machines; every following line
  is one job, as pairs `machine processing_time` in the order of its operations (machines numbered from 0).
* **`bks.json`:** for every instance the lower bound, the upper bound (best-known makespan, BKS) and the
  status from `jobshop/solutions/bks.json`.  `closed` means that the BKS is proven optimal.  Three instances
  are open: swv07, swv08 and swv10.

The parser (`mctsjssp.instances.orlib`) verifies on reading that the size agrees with Table 3.3, that every
job visits every machine exactly once and that all processing times are non-negative integers.  It also reads
the collection file `jobshop1.txt` of the OR-Library itself (`parse_orlib_collection`).

To refresh the best-known values from a newer copy of JSPLib:

    git clone --depth 1 https://github.com/ScheduleOpt/benchmarks /tmp/jsplib
    python scripts/update_bks.py /tmp/jsplib

## `published_rule_gaps.csv`

Published mean gaps of the dispatching rules SPT, MWKR, MOR and FIFO on the six OR-Library families, for the
second validation check (Section 3.7).  The values are those of Table 10 of Ho et al., *Residual Scheduling:
A New Reinforcement Learning Approach to Solving Job Shop Scheduling Problem* (arXiv:2309.15517v2; IEEE Access
12, 2024), transcribed from the arXiv version: check them against the paper before citing them.
`python -m mctsjssp validate-env` prints them next to the gaps obtained in this environment.
