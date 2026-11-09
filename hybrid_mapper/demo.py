"""python -m hybrid_mapper demo [--phase N]: a guided walkthrough of Phases 1-3 for a live presentation."""
import time

import pandas as pd

from hybrid_mapper.presets import PRESETS
from hybrid_mapper.rules import decide, load_rules


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def summary(name: str, d) -> str:
    return (f"  {name:<12} {d.decision:<9} via {d.fired_rule}   F={d.derived['fidelity']:.3g}  "
            f"shots={d.derived['shots']:.3g}   {d.trace[-1]['reason']}")


def phase1() -> None:
    banner("PHASE 1: transparent rules. Every decision names the rule that fired.")
    for name in ("lin_solve", "h2_vqe", "maxcut120", "fes_sqd", "fes_vqe", "deep_random"):
        print(summary(name, decide(PRESETS[name])))
    print("\n  Same 77-qubit, 3,500-gate circuit: SQD tolerates F~5e-4 (classical recovery repairs noisy"
          "\n  samples), VQE needs F>=0.05 for its 1/F^2 mitigation overhead. Opposite decisions.")
    cfg = load_rules()
    stricter = cfg | {"fidelity_min": cfg["fidelity_min"] | {"recover": 1e-3}}
    print("\n  Thresholds live in config/rules.yaml. Raise F_min[recover] from 1e-4 to 1e-3:")
    print(summary("fes_sqd", decide(PRESETS["fes_sqd"], cfg=stricter)))


def phase2() -> None:
    from hybrid_mapper.dataset import build, label, problem_features
    from hybrid_mapper.predict import ResourcePredictor, load_dataset, profile_from_features

    banner("PHASE 2: predict circuit resources from problem-level features (no transpile).")
    predictor = ResourcePredictor("power-law").fit(load_dataset())
    w = build("maxcut", 80, 2)
    for backend in ("nighthawk_r2", "heron_r3"):
        t0 = time.perf_counter()
        row = problem_features("maxcut", 80, 2, workload=w) | {"backend": backend}
        p = predictor.predict(pd.DataFrame([row])).iloc[0]
        d_pred = decide(profile_from_features(row, p.two_qubit_gates, p.depth, p.measurement_groups, "pred"))
        ms_pred = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        ex = label(w, backend)
        d_ex = decide(profile_from_features(row, ex["two_qubit_gates"], ex["depth"], ex["measurement_groups"], "ex"))
        ms_ex = (time.perf_counter() - t0) * 1e3
        print(f"  Max-Cut 80 nodes, p=2 on {backend:<13} analytic 2p|E| = {row['analytic_2q']}")
        print(f"    predicted {p.two_qubit_gates:6.0f} gates -> {d_pred.decision:<8} ({ms_pred:5.1f} ms)")
        print(f"    exact     {ex['two_qubit_gates']:6d} gates -> {d_ex.decision:<8} ({ms_ex:5.0f} ms transpile)")
    print("\n  Routing overhead (SWAPs) differs by topology: heavy-hex needs more than the square grid,"
          "\n  so the same problem is feasible on one QPU and not the other. The analytic count misses it.")


SHOWCASE = [
    ("Weakly entangled 100-qubit VQE: the rules see 100 qubits and a huge shot count; an MPS simulator\n"
     "  sees bond dimension 2^2.", ("vqe", 100, 2, "reverse_linear", 0, "heron_r3"), {}),
    ("100-node Max-Cut: the rules approve QAOA; the classical heuristic (0.878 guarantee) beats noisy QAOA.",
     ("maxcut", 100, 1, "none", 0, "nighthawk_r2"), {}),
    ("Deep 50-qubit sampling circuit: F=1e-4 fails R6, but a 1e8-shot budget buys ~20,000 clean samples.",
     ("random", 50, 20, "none", 0, "heron_r3"), {"shot_budget": 1e8, "samples_needed": 1e4}),
]


def phase3() -> None:
    from hybrid_mapper.allocate import allocate_one, evaluate_split, load_tasks, splits

    banner("PHASE 3: AI allocation learned from (simulated) execution outcomes.")
    base = {"precision_target": 1e-2, "maxcut_ratio_target": 0.85, "deadline_min": 480, "queue_wait_min": 30,
            "shot_budget": 1e7, "samples_needed": 1e5}
    for i, (story, spec, overrides) in enumerate(SHOWCASE, 1):
        print(f"\n--- Example {i}. {story}\n")
        print(allocate_one(*spec, base | overrides, compare=True))
    print("\n--- Held-out evaluation (30% of workloads never seen in training)\n")
    tasks = load_tasks()
    train, test = next(iter(splits(tasks).values()))
    r = evaluate_split(train, test)
    cols = ["Decision accuracy (vs oracle)", "Success rate", "Mean regret (cost units)", "QPU time (s, mean)",
            "Unnecessary QPU use"]
    table = pd.DataFrame(r["metrics"]).T[cols]
    print(table.to_string(float_format=lambda v: f"{v:,.3g}"))
    print("\n  Full tables, the learned tree, extrapolation and the feedback curve: results/phase3_allocation.md")


def run(phase: int | None = None) -> None:
    for n, fn in ((1, phase1), (2, phase2), (3, phase3)):
        if phase in (None, n):
            fn()
