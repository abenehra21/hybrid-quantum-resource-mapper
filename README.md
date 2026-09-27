# hybrid-quantum-resource-mapper

AI-driven task analysis and resource mapping for hybrid quantum–classical computing.

**Phase 1 (this repo)** is a transparent, rule-based baseline. It profiles a computational task and decides
whether it should run **CLASSICAL**, as a **HYBRID** quantum–classical workflow, or needs a **REDESIGN**.
Every decision comes with a trace showing which rule fired and why. Phase 1 also includes a dataset generator
(`baselines`) that runs small workloads under each strategy and logs the outcomes. Phase 2 (not in this repo yet)
will replace the hand-computed estimates with ML predictions trained on that dataset.

## How it works

A `TaskProfile` has three tiers of fields:

| Tier | Fields |
|------|--------|
| Problem | `task_id`, `problem_class` {classical, chem, opt, dyn, samp}, `algorithm` {none, vqe, dyn, qaoa, sqd, samp}, `precision_target`, `deadline_min`, `classical_meets_target`, `low_entanglement` |
| Circuit | `n_qubits`, `depth`, `two_qubit_gates`, `sum_abs_coeffs`, `samples_needed` |
| Hardware | `target_qpu`, `eplg`, `throughput_per_s`, `queue_wait_min`, `shot_budget` |

`rules.py` derives these quantities:

- Estimated fidelity: `F = exp(two_qubit_gates · ln(1 − eplg))`
- Algorithm kind: vqe, dyn → `expect`; qaoa, samp → `sample`; sqd → `recover`
- Shots: `expect` → `(sum_abs_coeffs / precision)² / F²`; `sample`/`recover` → `samples_needed`
- QPU seconds: `shots / throughput`
- Statevector bytes: `16 · 2^n_qubits`

The rules are then checked in order, and the first one that fires decides. Every threshold is in
[`config/rules.yaml`](config/rules.yaml), and the QPU specs are in [`config/backends.yaml`](config/backends.yaml).

| Rule | Condition | Decision | Reasoning |
|------|-----------|----------|-----------|
| R1 | `problem_class == classical` or `algorithm == none` | CLASSICAL | No quantum algorithm applies, so there is nothing to map. |
| R2 | statevector fits in `statevector_memory_gib` (16 GiB ⇔ n ≤ 30) | CLASSICAL | Exact simulation on a workstation beats any noisy QPU run. |
| R3 | `low_entanglement` | CLASSICAL | Tensor-network or MPS methods handle weakly entangled states efficiently. |
| R4 | `classical_meets_target` | CLASSICAL | If a classical method already reaches the precision target, the QPU adds cost but no value. |
| R5 | `n_qubits` > backend qubits | REDESIGN | The circuit does not fit on the device, so it needs cutting, compression or a different encoding. |
| R6 | `F < fidelity_min[kind]` (expect 0.05, sample 0.01, recover 1e-4) | REDESIGN | The output would be mostly noise. The tolerance depends on how much signal the algorithm can recover from noise. |
| R7 | `shots > shot_budget` | REDESIGN | Error-mitigation overhead or the sample count is unaffordable. |
| R8 | `queue_wait_min + qpu_seconds/60 > deadline_min` | CLASSICAL | The QPU result would arrive too late, so the task falls back to classical. |
| R9 | otherwise | HYBRID | The task is feasible. The quantum/classical split comes from `templates.py`. |

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest
.venv/bin/python -m hybrid_mapper decide --preset fes_sqd
.venv/bin/python -m hybrid_mapper profile --workload maxcut --size 16 --backend heron_r3
.venv/bin/python -m hybrid_mapper baselines --quick     # ~30 s; appends to logs/decisions.csv
```

Presets: `h2_vqe`, `maxcut20`, `maxcut120`, `maxcut120_open`, `fes_sqd`, `fes_vqe`, `lin_solve`, `deep_random`.
The `profile` workloads are `maxcut` (random 3-regular graph + QAOA), `vqe` (efficient-SU2 ansatz on a
**synthetic**, seeded Pauli Hamiltonian) and `random` (`qiskit.circuit.random.random_circuit`). Circuits are
transpiled at `optimization_level=3`, either to `FakeFez` (heron_r3) or to a 10×12 grid proxy (nighthawk_r2).

`baselines` runs small instances (≤12 qubits) two ways. The classical strategy uses exact diagonalisation or
brute-force Max-Cut. The hybrid strategy uses SPSA-optimised VQE/QAOA on qiskit-aer with the FakeFez noise model,
with the iteration cap set in `rules.yaml`. Each run appends one row per (workload, strategy) to `logs/decisions.csv`.
The row contains every profile field, the decision, the fired rule, the runtime, the result, the exact answer,
the error and the shots. This CSV is the Phase 2 dataset.

## Same circuit, opposite decision: fes_sqd vs fes_vqe

Both are spec-only profiles based on the published IBM–RIKEN SQD experiment: 77 qubits, 3,500 two-qubit
gates on heron_r3, so F ≈ 5.35e-4. No circuit is built for them.

```
$ python -m hybrid_mapper decide --preset fes_sqd
DECISION: HYBRID  (fired R9)

Derived:
  algorithm kind     recover
  est. fidelity F    0.0005351
  shots              3e+06
  QPU seconds        750
  statevector bytes  2.418e+24

Trace:
  R1  pass  task has a quantum algorithm candidate
  R2  pass  77 qubits: statevector 2.25e+15 GiB exceeds 16 GiB
  R3  pass  entanglement not flagged as low
  R4  pass  no classical method known to meet the target
  R5  pass  77 qubits fit on heron_r3 (156)
  R6  pass  F=0.000535 >= F_min[recover]=0.0001
  R7  pass  shots 3e+06 <= budget 1e+07
  R8  pass  queue+QPU 72.5 min <= deadline 480 min
  R9  FIRE  all checks passed: run sqd as hybrid workflow on heron_r3
```

```
$ python -m hybrid_mapper decide --preset fes_vqe
DECISION: REDESIGN  (fired R6)

Derived:
  algorithm kind     expect
  est. fidelity F    0.0005351
  shots              8.732e+15
  QPU seconds        2.183e+12
  statevector bytes  2.418e+24

Trace:
  R1  pass  task has a quantum algorithm candidate
  R2  pass  77 qubits: statevector 2.25e+15 GiB exceeds 16 GiB
  R3  pass  entanglement not flagged as low
  R4  pass  no classical method known to meet the target
  R5  pass  77 qubits fit on heron_r3 (156)
  R6  FIRE  F=0.000535 < F_min[expect]=0.05
```

SQD only needs noisy samples and recovers the signal classically, so a fidelity of about 5e-4 is usable.
VQE needs expectation values, and at the same fidelity the 1/F² mitigation overhead pushes the shot count to about 10¹⁶.

## Limitations

- The fidelity formula only counts two-qubit gate errors (via EPLG). It ignores single-qubit, readout, crosstalk,
  idling and leakage errors, so it is optimistic.
- The 1/F² mitigation overhead is a rough estimate. Real PEC/ZNE costs depend on the method and the noise model.
- Every threshold in `rules.yaml` is a tunable hypothesis, not an established cutoff. Phase 2 is meant to learn them.
- `classical_meets_target` and `low_entanglement` are supplied by hand. The profiler cannot infer them and defaults both to False.
- The nighthawk_r2 transpile target is a square-grid topology proxy, not a calibrated device model.
- VQE Hamiltonians are synthetic (`synthetic_hamiltonian`) and are not molecular data. The hybrid baselines use a
  capped SPSA optimiser, so their errors reflect that budget rather than the best achievable result.
