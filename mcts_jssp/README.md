# MCTS with fixed and learned rollout distributions for the job shop scheduling problem

Code for the experiments of the thesis: a configurable MCTS agent for the classical JSSP that is run with
four rollout distributions (U, MWKR, LR, AZ) on the 82 OR-Library instances, the self-labelling training of
the learned distributions, the experimental design of Table 3.6 (RQ1-RQ5), and the statistical analysis of
Section 3.6.4.  Everything is driven by configuration files; no experiment needs a change to the code.

- [1. Install and check](#1-install-and-check)
- [2. The pipeline](#2-the-pipeline)
- [3. Where each part of Chapter 3 lives](#3-where-each-part-of-chapter-3-lives)
- [4. Experiments and their configuration files](#4-experiments-and-their-configuration-files)
- [5. Results and analysis](#5-results-and-analysis)
- [6. Decisions the thesis leaves open](#6-decisions-the-thesis-leaves-open)
- [7. Cost of the rollouts](#7-cost-of-the-rollouts)
- [8. Tests](#8-tests)
- [9. Observations from the first runs](#9-observations-from-the-first-runs)

## 1. Install and check

Python 3.10 or newer.

```bash
pip install -e ".[test]"        # numpy, scipy, pandas, pyyaml, matplotlib, torch, torch_geometric, pytest
pytest                          # the test suite, about a minute
./scripts/smoke.sh              # every command of the pipeline with tiny settings, about six minutes
```

The smoke run writes to `smoke_out/`; its numbers mean nothing, it only shows that the pipeline works on the
machine.  Without `pip install -e .` set `PYTHONPATH=src` (the scripts do this themselves).

## 2. The pipeline

One command per step of Figure 3.6.  `./scripts/pipeline.sh` (Linux, Kaggle) or `.\scripts\pipeline.ps1`
(Windows PowerShell; `-Workers 12 -From 6 -Hours 10`) runs them in this order; trained networks and
finished runs are skipped, so every step can be restarted.

| Step | Command | Thesis |
|---|---|---|
| Validate the model and the environment | `python -m mctsjssp validate-env --config configs/base.yaml` | 3.7, checks 1-2 |
| Pilot study: time per iteration, hours of the design | `python -m mctsjssp pilot --config configs/pilot.yaml` | 3.6.3 |
| Train the networks by self-labelling (smaller sizes and teachers first, in the order of `models`) | `python -m mctsjssp train --config configs/base.yaml --all --workers 8` | 3.6.1, Table 3.4 |
| Train six networks by AlphaZero-style self-play for the comparison (pipeline step `self_play`) | `python -m mctsjssp train --config configs/base.yaml --model lr_6x6 ... --set paths.artifacts=artifacts/self_play --set training.method=alphazero` | 3.6.1 |
| Compare the training styles: the same validation of both (pipeline step `compare_training`) | `python -m mctsjssp compare-training --config configs/base.yaml --other artifacts/self_play --workers 8` | 3.6.1, Chapter 4 |
| Train the fast rollout networks (optional) | `python -m mctsjssp train-fast --config configs/base.yaml --all --workers 4` | section 6 below |
| Check the learning pipeline on 6 x 6 | `python -m mctsjssp validate-env --config configs/base.yaml --learning` | 3.7, check 3 |
| Assess the networks offline | `python -m mctsjssp assess --config configs/base.yaml --all` | 3.6.2 |
| Set temperatures, choose the entropy level | `python -m mctsjssp select-entropy --config configs/base.yaml` | 3.2.2, Table 3.5 |
| Run an experiment (resumable) | `python -m mctsjssp run --config configs/experiments/rq1.yaml --workers 8` | 3.6.3, Table 3.6 |
| Fast against large networks (optional) | `python -m mctsjssp run --config configs/experiments/fast.yaml --workers 8` | section 6 below |
| Compare the distributions on fixed states | `python -m mctsjssp fixed-states --config configs/experiments/rq3.yaml` | 3.6.4 |
| Tables, tests and figures | `python -m mctsjssp analyse --config configs/experiments/rq1.yaml` | 3.6.4 |

Any setting can be overridden on the command line, for example `--set runs=1 --set search.c=0.5`.
`run --hours H` stops an experiment after H hours (finished runs are kept); the same command continues it.
Temperatures and time budgets are computed when an experiment first needs them and are then stored.

**On AWS SageMaker**: `python sagemaker_pilot.py` (install, tests, pilot, budget test, `pilot_results.zip`).

**On AWS EC2** (`scripts/aws.sh`).  Ubuntu 22.04/24.04 or Amazon Linux 2023; a compute-optimised instance
without SMT (C7a/C8a: every vCPU is a physical core) gives steady timings for the time budgets.

    ./scripts/aws.sh setup    # packages, .venv, the package
    ./scripts/aws.sh test     # all unit tests (incl. slow) and the smoke run of every command
    ./scripts/aws.sh pilot    # validate, pilot study, budget test  -> read results/pilot/*.csv
    ./scripts/aws.sh study    # tune ... compare in a tmux session (survives a disconnect)
    ./scripts/aws.sh status   # progress;   ./scripts/aws.sh pack  # archive to download

Workers default to the physical cores minus one (`WORKERS=31 ./scripts/aws.sh pilot` to change).  Run the
pilot and the study on the same instance type: the time budgets are measured where they are used.

**One settings file for the whole study.**  `study.yaml` holds the values to decide before the experiments
(`settings`), values for single experiments (`experiment_settings`), the values fixed by Chapter 3 (`fixed`)
and how the pipeline runs (`pipeline`: workers, session hours, steps, experiments).  `run_pipeline.py` passes
them to every command, so base.yaml and the experiment files stay unchanged and every step uses the same values:

    python run_pipeline.py                          # all steps in study.yaml
    python run_pipeline.py --steps budget_test      # suggests training.budget_model_steps
    python run_pipeline.py --steps run,analyse --hours 10
    python run_pipeline.py --dry-run                # show the commands

## 3. Where each part of Chapter 3 lives

```
configs/            base.yaml (all defaults), experiments/rq1..rq5.yaml, pilot.yaml, smoke/ (tiny settings)
data/orlib/         the 82 instances and best-known makespans (JSPLib), see data/README.md
src/mctsjssp/
  instances/        3.5   instance, OR-Library parser and Table 3.3, generator, disturbances
  env/              3.3   dynamics model (model.py), 3.4 environment and observations, 3.7 checker
  distributions/    3.2.2 U, MWKR, LR/AZ rollouts; entropy, KL divergence; bisection of the temperature
  mcts/             3.2.1 UCT search (uct.py), decision loop and budgets (agent.py), PUCT for self-play,
                          recording at three levels (recorder.py)
  networks/         3.6.2 GIN of LR (gin.py), networks of AZ (ffnn.py), single-state inference, checkpoints,
                          fast rollout networks (fast.py)
  training/         3.6.1 self-labelling (self_label.py, used); earlier runs: self-play, ranked reward, replay
                          buffer, trainer, tuning (training.method: alphazero); offline assessment; copying
                          (imitation.py); self-play of the fast networks (fast_trainer.py)
  experiments/      3.6.3 design of Table 3.6 (design.py), runner, time budgets, pilot study,
                          3.7 validation and dispatching rules, fixed states of RQ3
  analysis/         3.6.4 gaps and convergence (metrics.py), tests (stats.py), per-RQ analysis (rq.py), figures
                          (plots.py), training curves (training_report.py), self-labelling against self-play
                          (compare_training.py)
  cli.py            the commands above
tests/              the test suite (section 8)
scripts/            smoke.sh, pipeline.sh, update_bks.py
```

The four configurations share one search class (`mcts/uct.py`) and one decision loop (`mcts/agent.py`); they
differ only in the rollout policy object that is passed in, as Table 3.2 requires.

## 4. Experiments and their configuration files

| File | Instances | Configurations | Budgets | Runs |
|---|---|---|---|---|
| `rq1.yaml` | all 82 | U, MWKR | 25, 50, 100, 200 iterations | 3 280 |
| `rq2.yaml` | all 82 | U, MWKR, LR (15 x 15 network) | as RQ1, plus T50 and T200 | 7 380 |
| `rq3.yaml` | 24 of a training size | U; MWKR, LR, AZ at entropy 0, 0.25, 0.5, 0.75 | 100 iterations, T200 | 3 120 |
| `rq4.yaml` | all 82 | U, MWKR, LR trained on 6 x 6, 10 x 10, 15 x 15 | T200 | 2 050 |
| `rq5.yaml` | 18 (first of each row of Table 3.3) | U, MWKR, LR; AZ on 6 instances; sigma 0.1 and 0.2 | T200 | 600 |
| `sensitivity_c.yaml` | 18 | U and MWKR at c = 0.05, 0.2, 1, 2 | 100 iterations | optional, not in Table 3.6 |

A configuration entry names the rollout distribution, its network and its entropy level:

```yaml
configurations:
  - {name: U}
  - {name: MWKR, entropy: tuned}                       # the level chosen by `select-entropy`
  - {name: LR, model: "lr_{size}", entropy: [0, 0.5]}  # the network of the size of the instance
  - {name: AZ, model: "az_{size}_s{sigma}", instances: rq5_az}
budgets:
  - {iterations: 100}
  - {time: 200}                                        # T200
```

To reduce a design (Section 3.6.3), edit the lists in its file or override them, for example
`--set instances=rq5`.  Results are stored under `results/<experiment>-<hash of the settings>/`, so changed
settings never mix with earlier results.

Networks are listed under `models:` in `configs/base.yaml`: three sizes for LR and AZ, and the networks
retrained with disturbances for RQ5 (`lr_15x15_s010`, `lr_15x15_s020`, `az_<size>_s010`, `az_<size>_s020`).

## 5. Results and analysis

Every run writes one line to `results.jsonl`: instance, family, size class, configuration, entropy level,
temperature, network, budget, seed, makespan, gap, job sequence of the schedule, best makespan found by a
rollout, iterations and time per decision and, if recorded, the behaviour of the search (entropy of the
rollouts, divergence from MWKR, agreement, variance and bias of the rollout returns, entropy of the visits at
the root).  `records/` holds the three levels of Section 3.6.4 per run: decisions, rollouts and the sampled
rollout steps.  For RQ5 the line also holds the realised makespans in the 30 fixed scenarios.

`analyse` writes CSV tables and PNG/PDF figures to `<results directory>/analysis/`:

| Experiment | Tables |
|---|---|
| RQ1-RQ4 | `summary` (mean gap over all instances, per size class, per family), `convergence` (gap against iterations and time per decision), `bks_reached` |
| RQ1 | `tests`: MWKR against U at every budget, all instances and each size class; `time_to_target` |
| RQ2 | `friedman` over U, MWKR, LR; `tests`: LR against U and against MWKR; `time_to_target` |
| RQ3 | `tests_same_entropy` (LR-MWKR, AZ-MWKR, LR-AZ at each level), `friedman_levels`, `behaviour` (realised entropy against target, KL, agreement, variance, bias), `relations`, `steps_profile` and `decisions_profile` (by depth and family), `fixed_states` |
| RQ4 | `by_size_ratio`, `tests_other_sizes`, `trend` |
| RQ5 | `summary` (realised gap, its standard deviation, increase over the undisturbed makespan), `friedman`, `tests` |
| rules | extension of RQ1-RQ2 (step `rules`, analysed with the U, MWKR and LR runs of RQ2): `validation` (the fixed distributions ranked on the validation instances at their chosen level, with the entropy reached), `friedman` over U, MWKR, MOR, SPT, FIFO and LR, `tests` (LR against each), `tests_best_fixed` (LR against the best fixed distribution on validation, Holm over the budgets), `tests_rules` (MOR, SPT, FIFO against U and MWKR) |
| Every experiment | `anytime` and `anytime_summary`, with the figures `anytime_<budget>.png/pdf` (see below) |

**Anytime curves (convergence to the BKS).**  Every run also records `anytime`: the best makespan of any
complete schedule found so far (rollouts and the committed schedule), with the wall time since the start of
the run, one point at every improvement, and `time_total`.  `anytime.csv` evaluates these step functions on a
common log-spaced time grid per budget and per group (all instances and each size class): the mean gap to the
BKS over all runs, the share of runs that have reached the BKS, and how many runs already have a schedule
(points are drawn only once every run has one; after a run ends its final value is carried forward).
`anytime_summary.csv` gives the gap and the share at 1 %, 10 % and 100 % of the longest run.  The figure
`anytime_<budget>` shows one line per configuration (colour and marker by configuration, line style by entropy
level: solid 0, dashed 0.5) against log time, with the BKS as the dashed line at 0 %.  Results written before
this was added have no `anytime` and are skipped.

Conventions of the test tables:

- The observation is the mean over the runs (or scenarios) of an instance.
- `median_diff` is the median of *first minus second* in percentage points; negative means the first
  configuration has the smaller gaps.  `rank_biserial` is positive when the first configuration is better.
- `p_holm` is the p-value after Holm's procedure; `holm_family` names the comparisons corrected together:
  the budgets of one group of instances in RQ1, the post-hoc comparisons of one Friedman test in RQ2 and
  RQ5, the levels and comparisons of one budget in RQ3, all comparisons of one budget in RQ4.
- Post-hoc comparisons are always listed; `friedman_rejects` says whether the Friedman test allows them to
  be interpreted.
- Zero differences in the Wilcoxon test are split between the two rank sums, as Demsar describes
  (`analysis.zero_method: zsplit`; `wilcox` and `pratt` are available).

## 6. Decisions the thesis leaves open

Each can be changed in `configs/base.yaml`.

**Search**
- A decision with one legal action is executed without a search (`search.skip_forced`).
- The committed action is the most visited one; ties go to the higher mean value, then the lower job.
- An operation with zero processing time (orb07) is a point that no other operation on its machine may
  contain, as in the disjunctive formulation.

**Entropy levels**
- Steps with a single candidate have no entropy and are left out of every average.
- The temperature is found by bisection to within 0.01, on rollouts that start at ten evenly spaced depths
  of a schedule, as the rollouts of the search do (`entropy.calibration`).
- The temperature of MWKR is set on the 15 x 15 validation instances, and in RQ3 on those of the size of the
  instance; the temperature of a network on the validation instances of its training size.
- One entropy level per configuration is chosen for use outside RQ3, with the 15 x 15 networks.

**Time budgets**
- T50 and T200 are one time per instance: the mean time per searched decision of U
  (`time_budget.profile: mean`).  Early decisions, whose rollouts are long, then get fewer iterations than
  late ones.  `per_decision` gives every decision the time that U needs at that depth instead.
- An iteration that has started is completed, so a configuration whose single rollout is longer than the
  budget overruns it; the actual time per decision is stored with every run.
- Times are measured per host and stored in `artifacts/time_budgets.json`.

**Recording**
- In RQ3 all three levels are recorded in the runs with 100 iterations; the runs at T200 record decisions
  and rollouts only, so that recording does not take time from the search (`recording.timed_level`).
- Recording uses its own random numbers: a run gives the same schedule with and without it.

**Training**
- The ranked-reward threshold is fixed for one generation, so episodes can be played in parallel with the
  same result.  Generation 0 (searches of U) is ranked against its own results, and so is the first
  generation that searches with the networks; after that the window holds only results of the networks
  (`training.rank_against_generation0`).  With the results of U in the window, no episode of the following
  six generations is rewarded.
- Inside the PUCT tree a complete schedule is valued by the ranked reward of its makespan; priors are the
  softmax of the logits at temperature 1; unvisited children start at Q = 0 as in AlphaZero
  (`training.q_init: parent` starts them at the mean value of their parent instead).
- Only decisions with at least two candidates are stored as records.
- The training budget in steps of the dynamics model is not set (`training.budget_model_steps: null`); fix
  it after the pilot study.  `training.min_generations` can delay early stopping.
- Tuning runs on 15 x 15 with 30 generations per trial (`tuning`).
- **Warm start (optional, off by default).**  `training.warm_start` first trains the policy network to copy a
  dispatching rule (imitation learning, `training/imitation.py`), for example
  `--set 'training.warm_start={rule: FIFO, instances: 300, epochs: 8, learning_rate: 0.001}'`.
  Generation 0 then searches with PUCT instead of U, the rule's results seed the ranking window, and the
  copied network is the first candidate for the best checkpoint, so self-play has to improve on it.  Copying
  MWKR would make the learned distribution start out as MWKR, which matters for RQ3.

**Stochastic setting**
- Networks for RQ5 are trained separately for sigma 0.1 and 0.2.
- The search draws its scenarios from the disturbance model with its own seed; the 30 evaluation scenarios
  of an instance are fixed and the same for every configuration.

**Statistics**
- The Friedman p-value is the chi-square one; the statistic of Iman and Davenport is reported next to it.
- `time_to_target` interpolates the mean gap linearly between the budgets that were run.

**Networks**
- The GIN is built with `GINConv` of PyTorch Geometric and trained with it.  In the rollouts the trained
  weights are evaluated in NumPy, and only the node embeddings that a step changes are recomputed.  The
  tests show that this gives the same logits (`inference_backend: torch` evaluates the PyTorch module).

**Fast rollout networks (optional; Section 4.1 of the thesis)**
- After AlphaGo's fast rollout policy: six cheap features per candidate operation (work remaining, operations
  remaining, processing time, start delay, waiting time, completion delay), computed from the partial
  schedule.  Fast LR: policy 6 -> 16 (ReLU) -> 1 shared by all candidates (129 weights, any size) and value
  6 -> 16 (ReLU) per candidate, mean over the candidates, -> 1 (tanh).  Fast AZ: the features of all jobs as one
  vector; policy 6n -> 32 (ReLU) -> n (3 407 weights on 15 x 15) and a separate value network 6n -> 32 -> 1
  (tanh).  `training.networks.fast_lr/fast_az: {hidden: 0}` gives the linear versions.
- Training is the same as for LR and AZ: `train` (or `train-fast`) runs PUCT self-play with the policy and the
  value as targets (U in generation 0), the ranked reward, the replay buffer, early stopping and the training
  budget.  In the experiments only the policy is used: it samples the rollouts of UCT (set a configuration's
  `model` to a fast network).  A model that lists a `teacher` is trained by copying that rule or network
  instead.  `training/fast_trainer.py` (UCT self-play with the network in the rollouts, no value output) is an
  earlier alternative and is no longer used by the commands.
- The pilot study times the fast networks as rollouts on every size (`FastLR`, `FastAZ` in `timing.csv`) and
  costs their PUCT training (`training.csv`).

**Comparison with U on fixed states** (`fixed-states`, Section 3.6.4).  Besides the entropy and the
divergence from MWKR, every distribution is compared with U, the reference: `kl_u` = D_KL(p || U) = ln k - H(p)
(and `kl_u_norm` = 1 - normalised entropy), `tv_u` = total variation distance from U (the share of probability
moved away from uniform), and, on the first `fixed_states.search_per_instance` states, the agreement of the
search: UCT from the state with the distribution's rollouts and with U's, `fixed_states.search_iterations`
iterations each; `agree_u_search` = same committed operation, `visit_overlap_u` = overlap of the root visit
distributions.  U against U with another seed is the baseline.  The command also writes
`analysis/fixed_states_summary.csv` (mean and standard error per distribution and level) and the figure
`analysis/fixed_states_vs_u.png/pdf`.  An agreement with U as "same most probable operation" is not defined,
because U has no most probable operation.

**Rollout policies of several experiments, instance by instance** (`compare`).

    python -m mctsjssp compare --config configs/experiments/rq2.yaml --config configs/experiments/fast.yaml

names every configuration `<experiment>:<label>` and compares them on the same instance and budget (disturbed
runs only at the same sigma).  It writes to `results/compare/` (`--out`): `per_instance.csv` (runs, mean, best
and standard deviation of the gap, time, iterations and rank per instance, budget and configuration),
`gap_by_instance_<budget>.csv` (instances as rows, configurations as columns), `vs_reference.csv` (every
configuration against U, or `--reference`: instances better / equal / worse, Wilcoxon with Holm, median
difference, rank-biserial correlation), `mean_rank.csv` (mean rank over the instances all configurations
share) and the heatmaps `compare_instances_<budget>` (gap minus that of U per instance and configuration;
blue = better than U).

**States visited by the rollouts, across experiments** (`compare-states`).  Set `recording.rollout_sequences`
(for example `--set recording.rollout_sequences=0.01`) when running an experiment: the complete job sequence
of that share of the rollouts is stored in `records/<run>.sequences.npz`, with where the tree path ends and
the rollout begins.  Every state a rollout visited is a prefix of its sequence and is rebuilt exactly.  Then

    python -m mctsjssp compare-states --config configs/experiments/rq2.yaml --config configs/experiments/rq3.yaml

rebuilds the states (identified by their partial schedule, so two orders that build the same schedule are one
state) and writes, per experiment, configuration, budget, instance and depth (share of the operations
scheduled): `distinct_share` (diversity of the rollout states), `partial_ratio` (partial makespan / LB0, the
quality of the states) and `overlap_u` (Jaccard overlap with the states of U on the same instance and budget),
plus every pair of configurations in `visited_overlap.csv`, and the figures `visited_states_<budget>`.
Configurations are compared only at the same budget and instance; disturbed runs (RQ5) are left out.
`--part all` includes the states of the tree path, `--instance` and `--max-sequences` limit the work.

**Validation**
- The second check runs SPT, MWKR, MOR and FIFO as non-delay rules, which is how the published gaps were
  obtained; `data/published_rule_gaps.csv` holds the values of Ho et al. (Table 10).

## 7. Cost of the rollouts

The pilot study measures the cost of one search iteration for every configuration and instance size.  The
figures below come from the machine on which the code was written (2 cores, one process, untrained
networks; the cost of a rollout does not depend on the weights).  They will differ on other hardware, but the
ratios are a property of the method: a rollout of LR or AZ calls a network at every step.

Time per iteration at the first decision (ms):

| Size | U | MWKR | AZ | LR | LR / U |
|---|---|---|---|---|---|
| 6 x 6 | 0.027 | 0.090 | 1.27 | 2.99 | 112 |
| 10 x 10 | 0.074 | 0.46 | 4.22 | 8.20 | 110 |
| 15 x 15 | 0.15 | 1.29 | 9.55 | 23.0 | 150 |
| 20 x 20 | 0.29 | 2.22 | - | 42.1 | 147 |
| 50 x 10 | 0.56 | 4.60 | - | 54.5 | 97 |

What follows from them:

- **Equal time.**  T200 is the time of 200 iterations of U.  One iteration of LR costs as much as 100-150
  iterations of U, so at T200 LR completes one or two iterations per decision and at T50 exactly one (which
  overruns the budget).  AZ completes about four at T200.  With one iteration the committed action is the
  single child that was expanded, chosen at random.  RQ4 and RQ5 use T200 only.  A budget `{time: N}` with a
  larger N gives the learned rollouts room to search; every N is supported.
- **Hours.**  The full design of Table 3.6 was costed at about 195 core hours on that machine, of which 147
  are the iteration budgets of LR in RQ2.  One generation of self-play for the 15 x 15 GIN takes about five
  minutes in one process.  Run `pilot` on the machine that will run the experiments before trusting either
  figure.
- **Implementation.**  The dynamics model is plain Python tuned for speed (about 0.5 microseconds per step).
  A step of an LR rollout costs 55-75 microseconds for any instance size, because only the node embeddings
  that the step changes are recomputed; a full evaluation of the GIN would cost 170 microseconds on 15 x 15
  and 380 on 50 x 10.

## 8. Tests

`pytest` runs 194 tests in under a minute; `pytest -m "not slow"` leaves out the end-to-end ones.

| File | What it checks |
|---|---|
| `test_instances.py` | parser and its error checks, Table 3.3 (82 instances, sizes, 35/20/27 size classes), subsets of RQ3 and RQ5, BKS, LB0, generator, disturbance model, fixed scenarios |
| `test_model.py` | insertion against a slow reference, a hand-checked schedule, active schedules, copies, execution with and without disturbances, the independent checker, the environment |
| `test_observation.py` | features and arcs, incremental update against recomputation |
| `test_distributions.py` | softmax, entropy, KL divergence, MWKR rule and distribution, identical decisions with and without recording, bisection of the temperature |
| `test_networks.py` | architectures of Table 3.4, equality of PyTorch Geometric, single-state and NumPy evaluation, incremental evaluation, masking, gradients, checkpoints |
| `test_mcts.py` | untried operations first, one node per iteration, backup, UCT formula, subtree reuse, budgets, determinism, recording, PUCT priors and Dirichlet noise |
| `test_training.py` | replay buffer, ranked reward, self-play episodes, trainer, stopping rules, reproducibility, parallel self-play, a network learning from informative targets, the self-play search with an informative value, the warm start, random search, entropy-level selection, offline assessment |
| `test_fast.py` | features by hand, sizes of the fast networks, NumPy against PyTorch, fast rollouts, copying, self-play from scratch, committed-action targets, the `train-fast` command in an experiment |
| `test_stats.py` | Wilcoxon against SciPy and by hand, effect sizes, Holm by hand, Friedman, comparison tables |
| `test_analysis.py` | gaps, summaries, convergence, time to target, the tables of RQ1-RQ5 on synthetic results, report files |
| `test_experiments.py` | configuration files, run counts of Table 3.6, seeds, validation checks (published rule gaps reproduced), pilot study, and end-to-end runs: resume, reproducibility across processes, stochastic evaluation, fixed states |

## 9. Observations from the first runs

Made while the code was written, on a 2-core machine.  They are not results of the study, but they bear on
how it is run.

**Validation (Section 3.7).**  Checks 1 and 2 pass.  All 3 936 schedules of the first check are feasible and
execution without disturbances reproduces every makespan.  Run as non-delay rules, SPT, MWKR, MOR and FIFO
reproduce the 24 published mean gaps of Ho et al. to within 0.10 percentage points.  In the MDP of the
thesis, where every unfinished job is a legal action, the same rules are weaker, except FIFO:

| Mean gap (%) | ft | la | abz | orb | swv | yn |
|---|---|---|---|---|---|---|
| MWKR, non-delay (published: 19.6, 12.6, 16.6, 25.1, 28.4, 19.7) | 19.6 | 12.6 | 16.6 | 25.2 | 28.4 | 19.7 |
| MWKR, every unfinished job a candidate | 28.3 | 18.4 | 25.9 | 31.3 | 35.3 | 29.9 |
| FIFO, either way | 28.9 | 18.9 | 24.6 | 29.7 | 37.4 | 25.9 |

The MWKR rollout distribution of Section 3.2.2 works on the second set of candidates, where MWKR is about as
good as FIFO.

**Equal time.**  See section 7: at T200 the learned rollouts complete one to four iterations per decision.

**Exploration constant.**  With c = 1 and returns of -Cmax / LB0, the visits at the root are close to
uniform at 100 and 200 iterations: in single runs of U on ft06, la16 and la36 the normalised entropy of the
visits was 0.997-0.999 and the most visited action received 19 %, 12 % and 8 % of the visits against 17 %,
11 % and 8 % for a uniform split.  `configs/experiments/sensitivity_c.yaml` measures the effect of c on the
gap.

**Self-labelling (used since update 4).**  Every iteration draws 64 new instances; the current policy samples
32 schedules of each at temperature 1, the shortest becomes the label, and the policy is fitted by
cross-entropy to its decisions (labels of the last 3 iterations, 4 epochs, Adam 1e-3); no value network,
search or reward model (Corsini et al., NeurIPS 2024; `training/self_label.py`, `training.self_label`).
The best network has the lowest mean of its greedy and sampled Cmax / LB0 on 100 validation instances.
GIN and fast LR work for every size: 6 x 6 starts from random weights, 10 x 10 from the best 6 x 6 network and
15 x 15 from 10 x 10 (`init: curriculum`); from random weights the 15 x 15 GIN did not learn (12 iterations,
sampled schedules as uniform), while the 6 x 6 network already gave greedy 1.59 on 15 x 15 (FIFO 1.59, uniform
1.77).  AZ and fast AZ depend on the size and did not learn from random weights on 10 x 10 and 15 x 15 (greedy
worse than uniform after 10-15 iterations, with and without job relabelling), so they first copy the greedy
schedules of LR / fast LR of their size and keep them in every fit (`teacher`, `teacher_keep`); every epoch
renumbers their jobs at random (`augment`).  Networks for disturbances start from their deterministic twin.
Sharper training samples (`temperature: auto`) gave better labels but not better networks.  The check stops
the pipeline only if a network starting with `lr_` or `flr_` has not learned (`stop_if_not_learned`).
Even with 1000 instances and 60 epochs the 15 x 15 networks of AZ copied LR only partly (greedy 1.85 against
LR 1.56), so expect AZ to stay near or below uniform on 15 x 15.

**Comparison with self-play (update 5).**  The step `self_play` trains lr_6x6, lr_10x10, lr_15x15, az_6x6,
az_10x10 and az_15x15 by the AlphaZero-style self-play of the original design (Table 3.4: PUCT with 100
iterations per decision, Dirichlet noise, ranked reward, replay buffer), from random weights, untuned and
without warm start, into `artifacts/self_play` (`self_play` in study.yaml); early stopping (5 generations
without improvement, not before generation 10, at most 50) ends it soon when it does not learn, and it never
stops the pipeline.  `compare_training` then judges both styles in the same way: the best checkpoint of each
network dispatches the 100 validation instances greedily and with 4 samples at temperature 1 (the same seeds),
next to uniform random dispatching, MWKR and FIFO, with the entropy of the policy and the minutes and steps
of training: `results/training/compare_training.csv`, `.md` and `.png`.

**Self-play training (earlier runs).**  The third check did not pass in test runs on 6 x 6 with the settings of Table 3.4
(learning rates 5e-5 and 5e-4, 12 to 26 generations).  The greedy policy of the best network, AZ at 5e-4,
went from 1.72 to 1.64-1.66 (Cmax / LB0) against 1.57 for uniform random dispatching; the GIN stayed between
1.65 and 1.77.  The self-play search itself improved from 1.61 to between 1.50 and 1.57 over 20 to 25
generations.  In every run the validation score failed to improve for five generations within the first
seven, so the stopping rule of Table 3.4 would have ended training there.

Two tests separate the parts of the loop.  A network fitted to the decisions of a dispatching rule reaches
the quality of that rule, and the PUCT search with an informative value dispatches at 1.34 against 1.56 with
a constant value (the search of U: 1.31).  The slow part is therefore the value network, which has to learn
from one label of +1 or -1 per episode, while the visit counts that serve as policy targets stay close to
the priors until it does.

**Warm start.**  Both networks were trained on 6 x 6 with the settings of Table 3.4 for 20 generations,
after first copying FIFO from 300 instances (about 10 400 decisions).  Mean Cmax / LB0 of the greedy policy on
the validation instances (lower is better):

| | After copying FIFO | Best after self-play | FIFO itself | Random |
|---|---|---|---|---|
| LR (GIN) | 1.374 | 1.374 (generation 0) | 1.396 | 1.550 |
| AZ | 1.445 | 1.425 (generation 17) | 1.396 | 1.572 |

The warm start gets both networks past the third check at once, and the GIN's copy is slightly better than
FIFO itself.  Self-play then added little in 20 generations: the GIN never improved on its copy (it drifted to
about 1.41 and came back to 1.378), and AZ improved by 0.02 without reaching FIFO.  The value networks did
start to learn: their loss fell from about 1.05 to 0.58-0.61, below the 0.75-0.8 of a network that only
predicts how often episodes win.  With the stopping rule of Table 3.4 (five generations without
improvement), the GIN would have stopped after generation 5; a warm start therefore needs a longer patience or
`training.min_generations`.  The searches of self-play (PUCT with the networks, 100 iterations) reached
1.36-1.50, against 1.31 for the search of U with uniform rollouts.

**Fast rollout networks.**  Measured per rollout step on 6 x 6, 10 x 10 and 15 x 15 (50 x 10 for LR):

| | U | MWKR | LR (GIN) | Fast LR | Fast LR, linear | AZ | Fast AZ |
|---|---|---|---|---|---|---|---|
| Microseconds per step | 0.7-1.2 | 2.4-9.0 | 80-103 | 14-52 | 6-37 | 34-41 | 17-27 |

Fast LR is 4-6 times faster than the GIN up to 15 x 15 and 2 times on 50 x 10; fast AZ 1.5-2 times faster
than AZ.  Most of what remains is computing the six features for every candidate in Python.

Trained to copy FIFO on 6 x 6, the fast networks dispatched better than the full ones (fast LR 1.348, fast AZ
1.391, LR 1.375, AZ 1.449 Cmax / LB0; FIFO 1.396).  Trained from scratch by self-play (Table 3.4 settings,
learning rate 5e-4, 20 generations), they learned almost nothing: the visit distributions at the root had a
normalised entropy of 0.998 in every generation, nearly uniform.  The best greedy policies reached 1.544
(fast LR), 1.616 (linear) and 1.652 (fast AZ) against 1.58-1.60 for random dispatching.

In the search on ft06 and 19 generated 6 x 6 instances (entropy level 0.5, two runs each), no network trained
from scratch differed significantly from U or MWKR at 100 iterations (all 1.30-1.35).  At T200 every one was
significantly worse than U (1.299) and MWKR (1.297), the fast ones by less: fast LR 1.341, fast LR linear
1.337, fast AZ 1.375, against 1.418 for the GIN of LR and 1.388 for AZ.

Rescaling the returns of each search to [0, 1] (as MuZero does) lowered the entropy of the visits only to
0.982, but gave better schedules (1.305 against 1.333 for U's search on 15 validation instances).

