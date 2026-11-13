# hybrid-quantum-resource-mapper

[![tests](https://github.com/abenehra21/hybrid-quantum-resource-mapper/actions/workflows/tests.yml/badge.svg)](https://github.com/abenehra21/hybrid-quantum-resource-mapper/actions/workflows/tests.yml)

**AI-driven task analysis and resource mapping for hybrid quantum–classical computing.**

Quantum processors are good at a narrow set of hard sub-problems, but they are small, noisy and slow to access.
Classical HPC handles everything else. Today, experts decide by hand which part of an application goes where.
This project automates that decision. It profiles an incoming task (qubits, gates, precision, hardware noise,
deadline) and routes it to **CLASSICAL** execution, a **HYBRID** quantum–classical workflow, or **REDESIGN** when
the task cannot work as posed. Every decision comes with a trace explaining why. A task that cannot run as posed
is then split automatically into parts that can, and scheduled as a task DAG.

**Visual demo:** [`docs/demo.html`](docs/demo.html) runs seven applications through all four phases, step by step,
with the partition, the task DAG and its schedule drawn out (open it in a browser; it works offline).

**Study module:** [`docs/handbook.html`](docs/handbook.html) explains every phase from first principles, with live
calculators, demo walkthroughs and a professor Q&A bank.

| Phase | Goal | Status |
|---|---|---|
| 1 · Rule-based prototype | Transparent decision rules, baseline runs, Phase 2 dataset | ✅ this repo |
| 2 · Resource prediction | ML models predict gate count, depth, shots, runtime, energy, error | ✅ this repo |
| 3 · AI task allocation | Learned allocator replaces or augments the rules | ✅ this repo (simulation study) |
| 4 · Automatic partitioning | Split applications into classical/quantum subtasks (task DAG) | ✅ this repo (simulation study + executed checks) |
| 5 · QPU integration | Run selected workloads on real hardware, collect feedback | planned |
| 6 · Federated learning | Aggregate model updates across sites without sharing raw data | planned |

```mermaid
flowchart LR
    A[Application] --> F1[F1 Characterize<br/>TaskProfile]
    F1 --> F2[F2 Suitability]
    F2 --> F3[F3 Partition<br/>Phase 4 planner]
    F3 --> F4[F4 Resource check<br/>fits the QPU?]
    F4 -- infeasible --> R[REDESIGN]
    F4 --> F5[F5 Execute<br/>CPU/GPU ⇄ QPU]
    F5 --> F6[F6 Feedback<br/>time · energy · error]
    F6 -.-> FL[Federated learning<br/>Phase 6]
    P2[Phase 2 predictor<br/>ms, no transpile] -. predicted circuit fields .-> F4
    R -. Phase 4: split, parts re-enter .-> F3
```

## Quickstart

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest

# Phase 1
.venv/bin/python -m hybrid_mapper decide --preset fes_sqd              # rule trace for a named profile
.venv/bin/python -m hybrid_mapper profile --workload maxcut --size 16  # build + transpile + profile
.venv/bin/python -m hybrid_mapper baselines --quick                    # ~1.5 min (full run ~25 min), appends to logs/decisions.csv

# Phase 2
.venv/bin/python -m hybrid_mapper predict --workload maxcut --size 80 --reps 2 --backend nighthawk_r2 --compare
.venv/bin/python -m hybrid_mapper dataset                              # regenerate data/resources.csv (~15 min)

# Phase 3
.venv/bin/python -m hybrid_mapper allocate --workload vqe --size 100 --reps 2 --compare   # rules vs AI vs truth
.venv/bin/python -m hybrid_mapper tasks                                # regenerate data/tasks.csv (~5 s)

# Phase 4
.venv/bin/python -m hybrid_mapper partition --workload maxcut --size 120 --target-ratio 0.9   # all 4 phases, task DAG
.venv/bin/python -m hybrid_mapper partition --workload clustered --size 26 --blocks 3 --precision 0.05
.venv/bin/python -m hybrid_mapper workflow --name screening_pipeline   # multi-kernel application as one DAG
.venv/bin/python -m hybrid_mapper plans                                # regenerate data/plans.csv (~1.5 min)

.venv/bin/python -m hybrid_mapper demo                                 # guided walkthrough of all 4 phases (~30 s)
.venv/bin/python -m hybrid_mapper demo --html                          # refresh the data in docs/demo.html (~1 min)
.venv/bin/python -m hybrid_mapper report                               # write results/*.md (~2 min)
```

Common workload flags: `--workload {maxcut,vqe,random}`, `--size`, `--backend {heron_r3,nighthawk_r2}`,
`--reps` (QAOA p / ansatz layers / random-circuit depth), `--entanglement` (VQE ansatz), `--seed`.
`allocate` and `partition` add operating conditions: `--precision`, `--target-ratio`, `--deadline`, `--queue`,
`--budget`, `--samples`. `partition` also takes `--workload clustered` with `--blocks` and `--links`.

---

## Phase 1: the rule-based decision engine

### The TaskProfile

| Tier | Fields | Cost to obtain |
|------|--------|---|
| Problem | `task_id`, `problem_class` {classical, chem, opt, dyn, samp}, `algorithm` {none, vqe, dyn, qaoa, sqd, samp}, `precision_target`, `deadline_min`, `classical_meets_target`, `low_entanglement` | free |
| Circuit | `n_qubits`, `depth`, `two_qubit_gates`, `sum_abs_coeffs`, `measurement_groups`, `samples_needed` | build + transpile (Phase 2 predicts these) |
| Hardware | `target_qpu`, `eplg`, `throughput_per_s`, `queue_wait_min`, `shot_budget` | backend calibration |

### Derived quantities (`rules.py`)

- Estimated fidelity: `F = exp(two_qubit_gates · ln(1 − eplg))`
- Algorithm kind: vqe, dyn → `expect`; qaoa, samp → `sample`; sqd → `recover`
- Shots: `expect` → `(sum_abs_coeffs / precision)² / F²`; `sample`/`recover` → `samples_needed`
- QPU seconds: `shots / throughput`. Statevector bytes: `16 · 2^n_qubits`

### The rules

The rules are checked in order and the first one that fires decides, so cheap checks come first. Every threshold
is in [`config/rules.yaml`](config/rules.yaml) and the QPU specs are in
[`config/backends.yaml`](config/backends.yaml). A test proves that editing the YAML changes decisions without
any code change.

| Rule | Condition | Decision | Reasoning |
|------|-----------|----------|-----------|
| R1 | `problem_class == classical` or `algorithm == none` | CLASSICAL | No quantum algorithm applies, so there is nothing to map. |
| R2 | statevector fits in `statevector_memory_gib` (16 GiB ⇔ n ≤ 30) | CLASSICAL | Exact simulation on a workstation beats any noisy QPU run. |
| R3 | `low_entanglement` (entanglement bound ≤ 12 ebits) | CLASSICAL | Tensor-network or MPS methods handle weakly entangled states efficiently. The profiler bounds entanglement from the circuit itself: each two-qubit gate crossing a cut of the qubit line can at most double the Schmidt rank there. |
| R4 | `classical_meets_target` | CLASSICAL | If a classical method already reaches the target, the QPU adds cost but no value. |
| R5 | `n_qubits` > backend qubits | REDESIGN | The circuit does not fit: it needs cutting, a smaller active space, or fragmentation. |
| R6 | `F < fidelity_min[kind]` (expect 0.05, sample 0.01, recover 1e-4) | REDESIGN | The output would be mostly noise. The tolerance depends on how much signal the algorithm can recover. |
| R7 | `shots > shot_budget` | REDESIGN | The measurement cost is unaffordable: group terms, lower precision, or switch algorithm. |
| R8 | `queue_wait_min + qpu_seconds/60 > deadline_min` | CLASSICAL | The QPU cannot answer in time, so fall back to classical. |
| R9 | otherwise | HYBRID | The task is feasible. The quantum/classical split comes from `templates.py`. |

### Same circuit, opposite decision: `fes_sqd` vs `fes_vqe`

Both are spec-only profiles based on the published IBM–RIKEN SQD experiment ([4Fe-4S], 77 qubits,
3,500 two-qubit gates on heron_r3, so F ≈ 5.35e-4). No circuit is built for them.

```
$ python -m hybrid_mapper decide --preset fes_sqd
DECISION: HYBRID  (fired R9)
  algorithm kind recover | F 0.0005351 | shots 3e+06 | QPU seconds 750
  R1  pass  task has a quantum algorithm candidate
  R2  pass  77 qubits: statevector 2.25e+15 GiB exceeds 16 GiB
  R3  pass  entanglement not flagged as low
  R4  pass  no classical method known to meet the target
  R5  pass  77 qubits fit on heron_r3 (156)
  R6  pass  F=0.000535 >= F_min[recover]=0.0001
  R7  pass  shots 3e+06 <= budget 1e+07
  R8  pass  queue+QPU 72.5 min <= deadline 480 min
  R9  FIRE  all checks passed: run sqd as hybrid workflow on heron_r3

$ python -m hybrid_mapper decide --preset fes_vqe
DECISION: REDESIGN  (fired R6)
  algorithm kind expect | F 0.0005351 | shots 8.732e+15 | QPU seconds 2.183e+12
  R1–R5 pass
  R6  FIRE  F=0.000535 < F_min[expect]=0.05
```

SQD only needs noisy samples and recovers the signal classically, so a fidelity of about 5e-4 is usable. VQE needs
accurate expectation values, and at the same fidelity the 1/F² mitigation overhead pushes the shot count to about 10¹⁶.

All presets are pytest cases: `h2_vqe` and `maxcut20` → R2, `maxcut120` → R4, `maxcut120_open` → R9,
`fes_sqd` → R9, `fes_vqe` → R6, `lin_solve` → R1, `deep_random` → R6.

### Baselines and the proposal's metric table

`baselines` runs small instances (≤ 8-qubit VQE, ≤ 12-node Max-Cut) under each strategy:

- **classical**: exact diagonalisation (VQE Hamiltonians) or brute-force Max-Cut.
- **hybrid**: VQE/QAOA optimised with calibrated SPSA on qiskit-aer, using the FakeFez noise model.
  The *projected* runtime replaces simulation time with estimated QPU time (shots ÷ throughput).

Each run appends one row per (workload, strategy) to `logs/decisions.csv`: every profile field, the decision,
and the measured runtime, estimated energy, QPU time, QPU round trips, shots, error and success. `report` turns
this into the proposal's comparison table. It adds an **oracle** label (the cheapest strategy that meets the
accuracy target) and reports the rule engine's decision accuracy and regret.

**Results** (full run, 7 workloads; details in [`results/phase1_baselines.md`](results/phase1_baselines.md)):

| Metric (mean per workload) | Classical-only | Hybrid (always) | Rule-based mapper | Oracle |
|---|---|---|---|---|
| Runtime, projected (s) | < 0.01 | 54.7 | < 0.01 | < 0.01 |
| Energy, estimated (J) | ≈ 1 | 1.37e6 | ≈ 1 | ≈ 1 |
| Accuracy (mean relative error) | 0 | 0.106 | 0 | 0 |
| QPU time, estimated (s) | 0 | 54.6 | 0 | 0 |
| Number of shots | 0 | 2.18e5 | 0 | 0 |
| Success rate (rel. error ≤ 5%) | 100% | 57% | 100% | 100% |
| Communication overhead (QPU round trips) | 0 | 106 | 0 | 0 |

The rule engine sends all seven small workloads to classical execution via R2 and matches the oracle every time
(100% decision accuracy, zero regret). At this size that is the expected result, not a trivial one: exact
simulation takes milliseconds, while noisy hybrid runs take about a minute of QPU time. Hybrid QAOA found the
optimal cut on every graph, but hybrid VQE missed the target by 8–37% under the capped SPSA budget and the device
noise. Workloads large enough for the decision to become contested cannot be simulated to get an oracle label.
That gap is what Phase 5 (real QPU feedback) closes.

---

## Phase 2: resource prediction

**Why predict when you can transpile?** Circuit-level fields cost a build plus a transpile, which takes about 1–2 s
per task and can reach ~30 s when a layout has no clean embedding. Phase 4 will need to score many candidate
partitions quickly. Predicting from problem-level fields lets the framework reject hopeless options before
building anything.

**Data.** `dataset` sweeps Max-Cut/QAOA (p = 1–3), EfficientSU2 VQE (linear, circular and full entanglement,
1–4 layers) and random circuits from 6 to 120 qubits on both backends. Each row is labelled with the exact
transpiled two-qubit gate count, depth and measurement groups (`data/resources.csv`, 608 rows). The labels
cost no QPU time.

**Models** (`predict.py`), one per target:

| Model | What it is | Expected behaviour |
|---|---|---|
| `analytic` | closed-form count before routing: 2·p·\|E\| (QAOA), reps·pairs (EfficientSU2) | the baseline to beat; no SWAP overhead |
| `power-law` | log-linear ridge per (family, backend, entanglement): log y ~ log n + log reps + log analytic | learns the routing overhead, and extrapolates |
| `gbt` | gradient-boosted trees on all features | strong inside the training range, but cannot extrapolate |

**Results** (details in [`results/phase2_prediction.md`](results/phase2_prediction.md)). The headline test is
*extrapolation*: train on ≤ 40 qubits and predict 50–120 qubits. The interpolation split holds out whole
circuits on both backends at once; a row-wise split would let the same circuit's other-backend row leak its
labels into training.

| Target (extrapolation, MAPE) | analytic | power-law | gbt |
|---|---|---|---|
| two-qubit gates | 51% | **14.5%** | 41% |
| depth | — | **19%** | 37% |
| measurement groups | (no grouping) | **1.5%** | 3.7% |

| Predicted fields → Phase 1 rules (extrapolation) | analytic | power-law | gbt |
|---|---|---|---|
| decision agrees with exact-transpile decision | 69% | **95.8%** | 90% |
| median error in log₁₀ F | 0.79 | **0.14** | 0.27 |

- **The power-law model beats the analytic baseline** on every target. It learns the routing (SWAP) overhead
  that 2·p·|E| misses.
- **Trees are as good as the power-law inside the training range** (100% decision agreement for both when 25%
  of circuits are held out), but they fall behind on larger sizes. This is the expected tree failure mode, and it is the
  reason to prefer a model family that extrapolates.
- **Speed:** predicting and deciding takes about 0.1 ms per task in a batch. Exact transpilation takes a median of 0.7 s,
  and 20 of 608 took over 10 s, up to 30 s.
- **Execution outcomes** (runtime, energy, error) are fitted with leave-one-out on the 7 hybrid runs. The
  power-law beats a mean predictor on all three (for example, runtime MAE is 11.7 s vs 36.8 s), but n = 7 means
  these numbers only show that the pipeline works. They grow with every `baselines` run.

**Deciding without transpiling:**

```
$ python -m hybrid_mapper predict --workload maxcut --size 80 --reps 2 --backend nighthawk_r2 --compare
PREDICTED with power-law model in 3.0 ms (no transpile)
DECISION: HYBRID  (fired R9)
  R6  pass  F=0.0123 >= F_min[sample]=0.01
COMPARE with exact transpile (518 ms):
  two_qubit_gates      predicted     1465   exact     1519
  depth                predicted      658   exact      708
  decision             predicted HYBRID (R9)   exact HYBRID (R9)

$ python -m hybrid_mapper predict --workload maxcut --size 80 --reps 2 --backend heron_r3 --compare
PREDICTED with power-law model in 2.3 ms (no transpile)
DECISION: REDESIGN  (fired R6)
  R6  FIRE  F=0.00678 < F_min[sample]=0.01
COMPARE with exact transpile (367 ms):
  two_qubit_gates      predicted     2320   exact     2448
  decision             predicted REDESIGN (R6)   exact REDESIGN (R6)
```

The same 80-node problem is feasible on Nighthawk's square lattice but not on Heron's heavy-hex. The difference
comes from routing overhead that the analytic formula (2·p·|E| = 480 gates on both) cannot see.

---

## Phase 3: AI task allocation

The proposal: *use ML predictions to replace or augment fixed resource-mapping decisions*. Judging any
allocator requires knowing the right answer for tasks too large to simulate, and that needs real execution
feedback (Phase 5). Until then Phase 3 is a **simulation study**. `environment.py` is a ground-truth model of
what would actually happen, built from effects the Phase 1 rules ignore, each taken from the literature:

| Effect | What the rules assume | What the ground truth knows |
|---|---|---|
| Tensor networks (Vidal 2003) | > 30 qubits cannot be simulated | weakly entangled circuits run exactly as an MPS at any width (bond dimension ≤ 2^ebits across the worst cut) |
| Classical heuristics (Goemans–Williamson 1995) | `classical_meets_target` is a hand-set flag | 0.878 Max-Cut guarantee vs QAOA worst cases of 0.692 / 0.756 / 0.792 for p = 1/2/3 (Farhi 2014; Wurtz & Love 2021) |
| Exact Max-Cut | not considered | small graphs are solved exactly by enumerating 2^(n−1) cuts (30 nodes ≈ 25 s, 40 nodes ≈ 9 h) |
| HPC node | 16 GiB workstation | 512 GiB node → statevector up to 35 qubits |
| Readout error | ignored | (1 − 0.0076)^n on sampling tasks (FakeFez median) |
| Shots-for-fidelity | hard F_min cut-off | a sampling task needs ~1,000 clean samples: shots = max(N, 1000/F) within budget |
| Queue | known exactly | realised wait is log-normal around the nominal value |

`tasks` pairs each of the 608 transpiled workloads with 4 sampled operating conditions (precision, ratio
target, deadline, queue, shot budget, samples) → 2,432 labelled tasks: CLASSICAL 73%, REDESIGN 16%,
HYBRID 11%. Allocators see only pre-execution information: problem-level features, conditions and **Phase 2
predicted** circuit fields. The hidden quantities reach them only through the labels.

**Results** on 30% of workloads never seen in training
([`results/phase3_allocation.md`](results/phase3_allocation.md) has everything):

| | Classical-only | Quantum-only | Rules (Phase 1) | Rules, tuned thresholds | Rules + R3 entanglement check | **AI: decision tree** | **AI: gradient boosting** |
|---|---|---|---|---|---|---|---|
| Decision accuracy vs oracle | 71% | 13% | 58% | 61% | 79% | 95% | **97%** |
| Mean regret (cost units) | 1,043 | 75,746 | 2,727 | 2,328 | 1,269 | **334** | 428 |
| Unnecessary QPU use | 0% | 71% | 6.2% | 3.6% | 4.8% | **0%** | **0%** |

Over 5 different random workload splits, gradient boosting averages **98.1% ± 0.8%** accuracy and regret
**213 ± 159**. The decision tree is nearly as accurate (96.9% ± 1.2%) but its regret varies a lot (411 ± 525), so
one split is not the whole story.

- **Replace beats augment.** Feeding the rules Phase 2 predictions (58%) or tuning their thresholds (61%)
  barely helps, because no threshold can say "weakly entangled circuits are classically easy". That needs
  features the rules don't have. 140 of the rules' 306 mistakes are exactly that case.
- **Giving the rules that feature is the fairest comparison.** "Rules (Phase 1)" is the rule list as originally
  specified, with R3 never firing. With R3 fed the circuit's entanglement bound, the rules reach 79% and halve
  their regret, but the AI still wins clearly. The remaining mistakes are interactions no single threshold
  expresses: the Max-Cut heuristic against the accuracy target, and low-fidelity sampling rescued by extra shots.
- **The tree is interpretable.** `allocate` prints the path a task takes through it, and the report prints the
  whole tree as if-then rules, ready to compare side by side with R1–R9.
- **Honest failure mode: extrapolation.** Trained only on ≤ 40 qubits, the AI is still far more accurate on
  larger tasks (78% vs 33%). But its errors there are expensive (QPU jobs that blow the shot budget), so its
  regret (5,907) is worse than always choosing classical (1,613). Choosing the action with the lowest expected
  regret instead of the most likely one barely helps (5,781): the model is confidently wrong out of range.
- **Range guard.** A learned policy is therefore only trusted up to the size that at least 5% of its training
  tasks reach; beyond it the rules with the R3 check decide. Out of range that cuts regret from 5,907 to
  2,324, and inside the training range it changes nothing on this split (5-split average 97.3% vs 98.1%).
- **Feedback fixes it (Function 6).** Feeding back 10% of the large workloads cuts gradient-boosting regret to
  1,970. At 30% it reaches **465**, beating classical-only (1,613), the static rules (5,340) and the rules
  with the entanglement check (2,289), none of which can learn. With the range guard the curve never rises
  above the rules on the way: 2,289 → 1,185 → 575. This is the proposal's research hypothesis, shown in
  simulation.

```
$ python -m hybrid_mapper allocate --workload vqe --size 100 --reps 2 --compare
Phase 1 rules (on predicted fields):  REDESIGN via R7: shots 2e+09 > budget 1e+07
  + R3 entanglement check (2 ebits): CLASSICAL via R3
Phase 3 AI (gradient boosting):       CLASSICAL  P(CLASSICAL)=1.00
GROUND TRUTH (... entanglement bound 2 ebits ...)
  classical: feasible  (mps in 1 s)
  oracle:    CLASSICAL
```

---

## Phase 4: automatic partitioning

Phases 1–3 answer REDESIGN when a task cannot run as posed. Phase 4 does the redesign. It splits the task,
sends every part back through Phase 2 (predicted resources) and Phase 3 (allocation) as an ordinary task, and
schedules the accepted split as a **task DAG** on classical nodes and the QPU (`partition.py`).

One splitter finds the modules of a weighted graph (Louvain communities, bisected until they fit). It is used
in two ways:

| | What is split | What a part is | What splitting costs | How it is checked |
|---|---|---|---|---|
| **Max-Cut** | the problem graph | a graph of ≤ 20 nodes, solved exactly | edges between parts are only decided in the merge step | **really executed**: every plan returns a cut and a certified ratio, cut / (sum of part optima + crossing edges) |
| **Gate cutting** | the circuit's qubit-interaction graph | a narrower circuit fragment | each cut CZ becomes 6 local circuits and needs 9× the shots (Mitarai & Fujii 2021) | `cut_expectation` rebuilds the uncut ⟨H⟩ from the fragments alone, to 1e-14 on small circuits |

The subtasks reuse everything from earlier phases by changing their conditions: a Max-Cut part has ratio target 1
(exact), and a circuit fragment needs precision ε / 3^cuts, which is the 9^cuts shot overhead in the units the
rules already use. Sampling tasks are not partitioned (cutting rebuilds expectation values, not samples), and a
fully entangled circuit has no modules to cut.

**New workload.** `clustered` circuits are fully entangled blocks joined by a few CZ gates, the shape circuit
cutting was proposed for (Peng et al. 2020). As one task they are too wide to simulate and too noisy to run.

**Data.** `plans` executes every candidate partition of every application in the ground-truth model: 3,288
plans for the 2,432 Phase 3 tasks plus 384 clustered tasks (`data/plans.csv`).

**What partitioning changes in the ground truth** ([`results/phase4_partitioning.md`](results/phase4_partitioning.md)):

- **351 of the 749 tasks that are REDESIGN as one task have a working partition (47%)**: all 149 stuck Max-Cut
  tasks (certified ratios 0.89–1.0, median 0.94, against the heuristic's 0.878 guarantee), and 202 of 353
  clustered ones. The rest need too many circuit variants for their deadline.
- No task that already works gets worse. Chain ansätze are never partitioned (an MPS simulator already handles
  them), and fully entangled 36- and 40-qubit circuits stay REDESIGN.
- **No part is ever sent to the QPU.** In this model the QPU's share of the work is sampling, which cannot be
  partitioned. Cutting turns oversized optimisation and modular chemistry tasks into classical work.

**Planners**, scored on 30% of workloads never seen in training. The plan oracle executes every candidate and
takes the cheapest one that works:

| | Rules + R3, no partitioning | Phase 3 AI alone | Partition whenever stuck | Rule planner | **Learned planner** | Plan oracle |
|---|---|---|---|---|---|---|
| Tasks that get a valid result | 59% | 70% | 81% | 77% | **84%** | 86% |
| Failed attempts | 12.7% | 4.6% | 8.5% | 5.1% | **1.4%** | 0% |
| REDESIGN tasks rescued | 0% | 0% | 40% | 26% | **51%** | 51% |
| Mean regret (cost units) | 1,615 | 928 | 2,875 | 717 | **288** | 0 |

Over 5 random workload splits the learned planner delivers a valid result for **84.6% ± 0.7%** of tasks (Phase 3
alone: 70.8% ± 1.6%; oracle: 86.0% ± 0.8%) with regret 530 ± 359 (Phase 3 alone: 903 ± 127).

- **Knowing when not to cut matters.** Partitioning whenever Phase 3 is stuck delivers more results than not
  partitioning, but its regret is three times worse (2,875 vs 928): many clustered plans miss their deadline.
- **The rule planner** accepts a split when every part is allocatable by Phase 3 and no fragment has more than
  2 cuts. It is safe and conservative (26% rescued).
- **The learned planner** is a classifier trained on executed plans ("does this candidate work and beat the task
  as it is?"). On this split it rescues every task the oracle rescues. Scoring a candidate takes about 1 ms.
- **Extrapolation, honestly.** When every test application is larger than anything in training, the range
  guard hands both learned components over to their rule versions. Partitioning still helps (valid results
  40% → 56%, regret 3,072 → 2,835), but the oracle reaches 77%.

```
$ python -m hybrid_mapper partition --workload maxcut --size 120 --target-ratio 0.9 --backend nighthawk_r2
Phase 1 rules:      HYBRID via R9: all checks passed: run qaoa as hybrid workflow on nighthawk_r2
Phase 3 AI:         REDESIGN  P(CLASSICAL)=0.00  P(HYBRID)=0.00  P(REDESIGN)=1.00
  ground truth as one task: REDESIGN  (classical: heuristic misses accuracy target; quantum: QAOA ratio 0.502 < target 0.9)

Phase 4 candidates (fewest cuts first):
  1. 10 parts (19, 17, 16, 15, 13, 10, ... qubits), 53 cuts
       parts through Phases 2+3: {'CLASSICAL': 10}   planner: ACCEPT
       ground truth: works (all 10 parts run, merged in time), certified ratio 0.942

DECISION: PARTITION (candidate 1)     plan oracle: PARTITION
  schedule: 5 s, 0 QPU-s, 12 kJ, cost J 17.01
```

**Applications of several kernels.** `workflow` plans every kernel of an application from
[`config/applications.yaml`](config/applications.yaml), joins the kernel DAGs along their dependencies and
schedules them together. The example pipeline becomes 21 steps on 4 classical nodes and one QPU: a Max-Cut
split into 10 parts, a 78-qubit modular circuit cut into 3 fragments, a sampling job on the QPU, and an MPS
simulation, finishing in 49 minutes of an 8-hour deadline. With a 30-minute deadline one kernel has no working
plan and is reported as blocked.

---

## Repository layout

```
hybrid_mapper/
  profile.py     TaskProfile dataclass + JSON
  backends.py    QPU specs from config/backends.yaml, transpile targets
  profiler.py    F1: build, transpile (optimization_level=3), fill circuit fields
  rules.py       F2+F4: derived quantities, ordered rules, decision trace
  templates.py   F3: fixed classical/quantum split per algorithm
  workloads.py   Max-Cut/QAOA, synthetic-Hamiltonian VQE, random circuits
  presets.py     named reference profiles (also test cases)
  baselines.py   classical vs hybrid runs -> logs/decisions.csv
  dataset.py     Phase 2 sweep -> data/resources.csv
  predict.py     Phase 2 models, evaluation, predicted profiles
  environment.py Phase 3 ground-truth execution model (never read by the rules)
  allocate.py    Phase 3 tasks, allocators (tuned rules, decision tree, boosting, range guard), evaluation
  partition.py   Phase 4 splitter, Max-Cut divide-and-merge, gate cutting, task DAG, schedule, planners
  demo.py        guided walkthrough of all four phases
  visual.py      refreshes the pipeline output embedded in docs/demo.html
  report.py      results/phase{1,2,3,4}_*.md
  cli.py         python -m hybrid_mapper {decide,profile,baselines,dataset,predict,tasks,allocate,plans,
                 partition,workflow,demo,report}
config/          backends.yaml, rules.yaml (rule thresholds), environment.yaml (ground-truth and partitioning
                 assumptions), applications.yaml (multi-kernel example applications)
data/            resources.csv (Phase 2 training data), tasks.csv (Phase 3 labelled tasks),
                 clustered.csv and plans.csv (Phase 4 applications and executed candidate plans)
docs/            demo.html (visual demo), handbook.html (study module)
results/         generated reports
tests/           pytest suite (presets, config-driven thresholds, formulas, models, cutting and scheduling checks)
```

## Limitations

- **Fidelity model.** `F = (1 − eplg)^N₂q` ignores single-qubit, readout, crosstalk, idling and leakage errors,
  so it is optimistic. The residual against measured fidelity is a natural Phase 2/5 learning target.
- **Mitigation overhead.** 1/F² is a rough estimate. Real PEC/ZNE/PEA costs depend on the method and the noise.
- **Thresholds are hypotheses.** The 30-qubit line depends on the simulating machine, and F_min values are
  starting points. All of them live in `rules.yaml`, and learning them is the point of Phase 3.
- **Hand-supplied flag.** `classical_meets_target` is not inferred and defaults to False. `low_entanglement` is
  inferred from an entanglement bound, which is an upper bound: a circuit can be easier to simulate than it says.
- **Topology proxy.** nighthawk_r2 is transpiled to a 10×12 square grid, not a calibrated device model.
- **Synthetic chemistry.** VQE Hamiltonians are seeded random bounded-weight Pauli sums (`synthetic_hamiltonian`),
  not molecular data.
- **Simulated hybrid runs.** Runtimes are projected (QPU time from throughput), and energy uses *assumed* power
  figures in `rules.yaml` (QPU power attributed by share of QPU time). Hybrid accuracy reflects a capped SPSA
  budget and the FakeFez noise model.
- **Small execution dataset.** Runtime, energy and error models train on a handful of simulated runs. The report
  prints n next to every number. Transpiler-labelled resources (608 rows) are the solid part of Phase 2.
- **Prediction scope.** Predictors cover the three workload families in the sweep. A new family needs new
  training rows.
- **Phase 3 is a simulation study.** Its labels come from a ground-truth *model* whose effects come from the
  literature, so the AI is shown to recover those effects from pre-execution features, not to beat real
  hardware. The same pipeline retrains unchanged on real execution feedback once Phase 5 exists.
- **Worst-case vs typical.** The ground truth compares worst-case guarantees on both sides (GW 0.878 vs QAOA
  bounds). Typical-instance performance of both is better, and the p = 3 bound is conditional on a conjecture.
- **Phase 4 fragments are classical in this model.** No fragment of any working plan is worth a QPU here, so
  the partitioner is shown to move work to classical resources, not to make quantum advantage. A fragment
  that is too wide and too entangled to simulate yet shallow enough to survive noise would change that.
- **Cutting costs are upper bounds.** A fragment is charged 6^cuts circuit variants and 9^cuts shots. Better
  decompositions and classical communication between fragments lower both; wire cutting is not modelled.
- **Max-Cut parts are capped at 20 nodes** so that every plan in the evaluation is really solved by enumeration
  on a laptop. Larger parts would raise the certified ratio. The certificate understates the true ratio.
- **Clustered circuits are outside the Phase 2 and 3 training families.** The predictor is far off on the whole
  circuit (it is accurate on the fragments, which are in-family), and the allocator sees it as a wide fully
  entangled ansatz.
- **The schedule is a simple list schedule** on a fixed number of classical nodes and one QPU with one queue
  wait. It is not an optimal scheduler and does not model data movement between steps.
